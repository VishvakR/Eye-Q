"""FastAPI application factory with full component lifecycle management.

The lifespan owns **every** runtime component — database, ring buffer, video
capture, Stage 1 watchdog, Stage 2 VLM worker, and the alert sink — because
splitting that ownership between here and ``main.py`` previously let the
lifespan overwrite components the orchestrator had already wired up, silently
severing rule delivery to the VLM.

Startup order:   DB → buffer → capture → Stage 1 → Stage 2 → alert sink
Shutdown order:  reverse (PLAN.md §8)

Usage:
    >>> from app.api.server import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn app.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import CameraCreate, RuleCreate, RuleUpdate, ws_manager
from app.core.database import Database
from app.core.labels import COCO_CLASSES, labels_in_prompt, scope_warning
from app.core.roi import RegionOfInterest

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_ID = "default"


# ── Rule Synchronisation ─────────────────────────────────────────


def _parse_labels(rule: dict) -> list[str]:
    """Decode a rule's ``target_labels`` JSON column into a list."""
    try:
        labels = json.loads(rule["target_labels"])
        return labels if isinstance(labels, list) else []
    except (TypeError, ValueError):
        return []


def _parse_roi(rule: dict) -> RegionOfInterest | None:
    """Decode a rule's ``roi_json`` polygon, or None for full-frame."""
    try:
        vertices = json.loads(rule["roi_json"])
    except (TypeError, ValueError):
        return None

    if not isinstance(vertices, list) or len(vertices) < 3:
        return None

    try:
        return RegionOfInterest(
            vertices=[(float(x), float(y)) for x, y in vertices],
            name=rule.get("name", "rule"),
        )
    except (TypeError, ValueError) as exc:
        logger.warning(f"Rule {rule.get('id')} has an invalid ROI polygon: {exc}")
        return None


async def sync_rules(app_obj: FastAPI) -> None:
    """Push the enabled rules from SQLite into both pipeline stages.

    Called on startup and after every rule create/update/delete. Without this
    the VLM never sees the user's plain-English prompts and the watchdog
    triggers on every COCO class.

    Stage 1 gets the *coarsest* settings across all rules (label union, minimum
    confidence, minimum cooldown) so it never filters out something a rule
    still wants. Stage 2 then applies each rule's own prompt and ROI, since a
    single watchdog cannot carry N distinct polygons.
    """
    db: Database = app_obj.state.db
    rules = [r for r in await db.list_rules() if r["enabled"]]

    # ── Stage 2: per-rule prompts and ROIs ───────────────────────
    vlm_worker = getattr(app_obj.state, "vlm_worker", None)
    if vlm_worker is not None:
        vlm_worker.clear_rules()
        for rule in rules:
            roi = _parse_roi(rule)
            for label in _parse_labels(rule) or ["*"]:
                vlm_worker.add_rule(
                    label=label,
                    prompt=rule["prompt"],
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    roi=roi,
                )

    # ── Stage 1: coarsest filter that satisfies every rule ───────
    watchdog = getattr(app_obj.state, "watchdog", None)
    if watchdog is None:
        return

    settings = app_obj.state.settings

    label_union: set[str] = set()
    wildcard = False
    for rule in rules:
        labels = _parse_labels(rule)
        if labels:
            label_union.update(labels)
        else:
            # A rule with no label filter must see everything.
            wildcard = True

    # None = every label triggers, empty set = none do. With no enabled rules
    # there is no question to ask, so nothing should wake the VLM — the
    # detector keeps running and the live feed keeps drawing, but Stage 2
    # stays asleep instead of burning inferences nobody asked for.
    if wildcard:
        scope = None
    elif label_union:
        scope = label_union
    else:
        scope = set()

    watchdog.update_target_labels(scope)
    watchdog.update_confidence(
        min((r["confidence"] for r in rules), default=settings.yolo_confidence)
    )
    watchdog.update_cooldown(
        min((r["cooldown_s"] for r in rules), default=settings.detection_cooldown_s)
    )

    if scope is None:
        scope_desc = "all"
    elif scope:
        scope_desc = str(sorted(scope))
    else:
        scope_desc = "none — Stage 2 idle until a rule is enabled"

    logger.info(f"Synced {len(rules)} enabled rule(s) to the pipeline (triggers on: {scope_desc}).")


# ── Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop every system component in dependency order."""
    from app.config.settings import get_settings

    settings = get_settings()

    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.buffer = None
    app.state.watchdog = None
    app.state.vlm_worker = None
    app.state.captures = []
    app.state.publisher = None
    app.state.alert_sink = None

    # ── Database ─────────────────────────────────────────────────
    db = await Database.create(settings.database_path)
    await db.initialize_tables()
    app.state.db = db

    if settings.enable_pipeline:
        await _start_pipeline(app, settings, db)

    # Rules must reach the workers before the first frame is judged.
    await sync_rules(app)

    logger.info("Eye-Q control plane ready.")

    yield

    # ── Shutdown (reverse order) ─────────────────────────────────
    logger.info("Shutting down Eye-Q components...")

    if app.state.alert_sink is not None:
        await app.state.alert_sink.stop()
    if app.state.vlm_worker is not None:
        app.state.vlm_worker.stop()
    if app.state.watchdog is not None:
        app.state.watchdog.stop()
    for capture in app.state.captures:
        capture.stop()
    if app.state.publisher is not None:
        app.state.publisher.close()
    if app.state.buffer is not None:
        app.state.buffer.close()
        app.state.buffer.unlink()

    await db.close()
    logger.info("Eye-Q shut down cleanly.")


async def _start_pipeline(app: FastAPI, settings, db: Database) -> None:
    """Construct and start buffer, capture, Stage 1, Stage 2, and the sink."""
    from app.core.buffer import RollingFrameBuffer
    from app.core.capture import VideoCapture
    from app.core.ipc import EventPublisher
    from app.pipelines.detector import YOLOWatchdog
    from app.services.alert_sink import AlertSink

    logger.info("=" * 60)
    logger.info("  Eye-Q — Open-Vocabulary VLM Retrofit Appliance")
    logger.info("=" * 60)
    logger.info(f"  Device:     {settings.inference_device}")
    logger.info(f"  Source:     {settings.default_source}")
    logger.info(f"  Resolution: {settings.frame_width}×{settings.frame_height}")
    logger.info(f"  Buffer:     {settings.buffer_capacity} frames")
    logger.info(f"  YOLO:       {settings.yolo_model}")
    logger.info(f"  VLM:        {settings.vlm_backend} / {settings.vlm_model}")
    logger.info("=" * 60)

    # Register the configured source so the API reflects reality. The schema
    # stays multi-camera-shaped even though one stream runs today.
    await db.create_camera(
        camera_id=DEFAULT_CAMERA_ID,
        name="Default Camera",
        source=str(settings.default_source),
        resolution=f"{settings.frame_width}x{settings.frame_height}",
        fps=settings.target_fps,
    )

    buffer = RollingFrameBuffer(
        capacity=settings.buffer_capacity,
        frame_height=settings.frame_height,
        frame_width=settings.frame_width,
        buffer_id="main",
    )
    app.state.buffer = buffer

    publisher = EventPublisher(settings.zmq_endpoint)
    app.state.publisher = publisher

    capture = VideoCapture(
        source=settings.video_source,
        buffer=buffer,
        camera_id=DEFAULT_CAMERA_ID,
        target_fps=settings.target_fps,
    )
    capture.start()
    app.state.captures = [capture]

    watchdog = YOLOWatchdog(
        buffer=buffer,
        publisher=publisher,
        device=settings.inference_device,
        model_name=settings.yolo_model,
        confidence_threshold=settings.yolo_confidence,
        imgsz=settings.yolo_imgsz,
        cooldown_seconds=settings.detection_cooldown_s,
        camera_id=DEFAULT_CAMERA_ID,
        target_fps=settings.watchdog_fps,
        track_enabled=settings.track_enabled,
        track_iou_threshold=settings.track_iou_threshold,
        track_max_age_frames=settings.track_max_age_frames,
    )
    watchdog.start()
    app.state.watchdog = watchdog

    if settings.enable_vlm:
        from app.pipelines.vlm_backends import build_backend
        from app.pipelines.vlm_worker import VLMWorker

        vlm_worker = VLMWorker(
            buffer=buffer,
            zmq_endpoint=settings.zmq_endpoint,
            result_endpoint=settings.zmq_result_endpoint,
            backend=build_backend(settings),
            staleness_threshold_s=settings.vlm_staleness_s,
            crop_padding=settings.vlm_crop_padding,
            inference_timeout_s=settings.vlm_timeout_s,
            snapshot_dir=settings.snapshot_dir,
            default_prompt=settings.vlm_default_prompt,
            focus_dim=settings.vlm_focus_dim,
            verdict_ttl_s=settings.vlm_verdict_ttl_s,
            rejudge_iou=settings.vlm_rejudge_iou,
            min_rejudge_s=settings.vlm_min_rejudge_s,
            max_instances_per_batch=settings.vlm_max_instances_per_batch,
        )
        vlm_worker.start()
        app.state.vlm_worker = vlm_worker

        # The sink turns verdicts into stored, broadcast, webhooked alerts.
        sink = AlertSink(db, ws_manager, settings)
        await sink.start()
        app.state.alert_sink = sink
    else:
        logger.warning("VLM worker disabled — Stage 1 only, no verdicts will be produced.")


# ── MJPEG Preview ────────────────────────────────────────────────


def _render_preview_frame(app: FastAPI, quality: int, max_width: int) -> bytes | None:
    """Grab the latest frame, annotate it, and JPEG-encode it.

    Runs in a worker thread — ``cv2.imencode`` is CPU-bound and would stall
    the event loop (taking the REST API and WebSocket down with it) if it ran
    inline.

    Returns:
        Encoded JPEG bytes, or None if no frame is available yet.
    """
    from app.core.overlay import draw_detection, draw_roi

    buffer = app.state.buffer
    if buffer is None:
        return None

    try:
        frame, _ts, _idx = buffer.read_latest()
    except RuntimeError:
        return None  # Buffer still empty

    watchdog = getattr(app.state, "watchdog", None)
    if watchdog is not None:
        roi = watchdog.roi
        if len(roi.vertices) > 3 or roi.vertices != [(0, 0), (1, 0), (1, 1), (0, 1)]:
            draw_roi(frame, roi)
        for event, triggered in watchdog.recent_detections(max_age_s=1.0):
            draw_detection(frame, event.bbox, event.label, event.confidence, triggered)

    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)))

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else None


async def _mjpeg_generator(app: FastAPI):
    """Yield ``multipart/x-mixed-replace`` parts at the configured frame rate.

    Each part is preceded by CRLF because RFC 2046 defines the delimiter as
    ``CRLF--boundary``. Emitting the first boundary without it is tolerated by
    Chrome and Firefox but leaves Safari unable to parse the stream at all.
    """
    settings = app.state.settings
    interval = 1.0 / settings.stream_fps

    while True:
        started = time.monotonic()

        jpeg = await asyncio.to_thread(
            _render_preview_frame, app, settings.stream_quality, settings.stream_width
        )

        if jpeg is not None:
            yield (
                b"\r\n--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg
            )

        elapsed = time.monotonic() - started
        await asyncio.sleep(max(0.0, interval - elapsed))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Eye-Q",
        description="Open-vocabulary VLM retrofit appliance for IP camera surveillance",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Serve frontend static files ──────────────────────────────
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.middleware("http")
    async def no_cache_dashboard_assets(request: Request, call_next):
        """Serve the dashboard's HTML/JS/CSS uncached.

        A cached ``app.js`` paired with a freshly served ``index.html`` puts the
        page in a state neither version was written for — which is exactly how
        the live feed once rendered as an empty element. The dashboard is a
        handful of KB served from localhost, so caching buys nothing and costs
        a whole class of "works after a hard reload" bugs.
        """
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response

    # Alert evidence JPEGs. Created eagerly so the mount never 500s on a
    # fresh checkout that has not yet produced an alert.
    snapshot_dir = Path("data/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/snapshots", StaticFiles(directory=str(snapshot_dir)), name="snapshots")

    # ── Routes ───────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        """Serve the main dashboard HTML."""
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return HTMLResponse(
            "<h1>Eye-Q</h1><p>Frontend not found. Place index.html in frontend/</p>"
        )

    # ── Live Preview ─────────────────────────────────────────────

    @app.get("/api/stream/{camera_id}.mjpg")
    async def stream_camera(request: Request, camera_id: str):
        """Live MJPEG preview with detection boxes drawn on each frame."""
        if request.app.state.buffer is None:
            raise HTTPException(503, "Video pipeline is not running")

        return StreamingResponse(
            _mjpeg_generator(request.app),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/stream/{camera_id}/frame.jpg")
    async def stream_frame(request: Request, camera_id: str):
        """Single annotated frame as a plain JPEG.

        The universal fallback for the live view: every browser renders a
        JPEG, whereas ``multipart/x-mixed-replace`` support in ``<img>`` is
        uneven (Safari in particular). The dashboard polls this when the MJPEG
        stream fails to load.
        """
        state = request.app.state
        if state.buffer is None:
            raise HTTPException(503, "Video pipeline is not running")

        settings = state.settings
        jpeg = await asyncio.to_thread(
            _render_preview_frame, request.app, settings.stream_quality, settings.stream_width
        )
        if jpeg is None:
            raise HTTPException(503, "No frame available yet")

        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # ── Rules CRUD ───────────────────────────────────────────────

    @app.get("/api/rules")
    async def list_rules(request: Request, camera_id: str | None = Query(None)):
        db: Database = request.app.state.db
        return await db.list_rules(camera_id=camera_id)

    @app.post("/api/rules", status_code=201)
    async def create_rule(request: Request, rule: RuleCreate):
        db: Database = request.app.state.db
        rule_id = await db.create_rule(
            camera_id=rule.camera_id,
            name=rule.name,
            prompt=rule.prompt,
            target_labels=rule.target_labels,
            roi_json=rule.roi_json,
            confidence=rule.confidence,
            cooldown_s=rule.cooldown_s,
            enabled=rule.enabled,
        )
        await sync_rules(request.app)

        created = await db.get_rule(rule_id)
        # Non-blocking: an unscoped rule is legitimate for scene-level
        # questions, so this informs rather than refuses.
        warning = scope_warning(rule.prompt, rule.target_labels)
        if warning:
            logger.warning(f"Rule {rule_id} ('{rule.name}'): {warning}")
            created = {**created, "warning": warning}
        return created

    @app.delete("/api/rules")
    async def delete_all_rules(request: Request):
        """Delete every user-defined rule at once.

        Alert history survives — the rows are reassigned to the sentinel rule
        rather than cascading away with their rule.
        """
        db: Database = request.app.state.db
        deleted = await db.delete_all_rules()
        await sync_rules(request.app)
        return {"deleted": deleted}

    @app.get("/api/rules/{rule_id}")
    async def get_rule(request: Request, rule_id: int):
        db: Database = request.app.state.db
        rule = await db.get_rule(rule_id)
        if rule is None:
            raise HTTPException(404, f"Rule {rule_id} not found")
        return rule

    @app.put("/api/rules/{rule_id}")
    async def update_rule(request: Request, rule_id: int, rule: RuleUpdate):
        db: Database = request.app.state.db
        existing = await db.get_rule(rule_id)
        if existing is None:
            raise HTTPException(404, f"Rule {rule_id} not found")
        fields = rule.model_dump(exclude_unset=True)
        updated = await db.update_rule(rule_id, **fields)
        await sync_rules(request.app)
        return updated

    @app.delete("/api/rules/{rule_id}", status_code=204)
    async def delete_rule(request: Request, rule_id: int):
        db: Database = request.app.state.db
        deleted = await db.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(404, f"Rule {rule_id} not found")
        await sync_rules(request.app)

    # ── Alerts ───────────────────────────────────────────────────

    @app.get("/api/alerts")
    async def list_alerts(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        rule_id: int | None = Query(None),
    ):
        db: Database = request.app.state.db
        alerts = await db.list_alerts(limit=limit, offset=offset, rule_id=rule_id)
        total = await db.count_alerts(rule_id=rule_id)
        return {"alerts": alerts, "total": total, "limit": limit, "offset": offset}

    @app.get("/api/alerts/{alert_id}")
    async def get_alert(request: Request, alert_id: int):
        db: Database = request.app.state.db
        alert = await db.get_alert(alert_id)
        if alert is None:
            raise HTTPException(404, f"Alert {alert_id} not found")
        return alert

    # ── Cameras ──────────────────────────────────────────────────

    @app.get("/api/cameras")
    async def list_cameras(request: Request):
        db: Database = request.app.state.db
        return await db.list_cameras()

    @app.post("/api/cameras", status_code=201)
    async def create_camera(request: Request, camera: CameraCreate):
        db: Database = request.app.state.db
        await db.create_camera(
            camera_id=camera.id,
            name=camera.name,
            source=camera.source,
            resolution=camera.resolution,
            fps=camera.fps,
            enabled=camera.enabled,
        )
        return await db.get_camera(camera.id)

    @app.delete("/api/cameras/{camera_id}", status_code=204)
    async def delete_camera(request: Request, camera_id: str):
        db: Database = request.app.state.db
        deleted = await db.delete_camera(camera_id)
        if not deleted:
            raise HTTPException(404, f"Camera {camera_id} not found")

    # ── Status ───────────────────────────────────────────────────

    @app.get("/api/status")
    async def get_status(request: Request):
        state = request.app.state
        db: Database = state.db
        db_stats = await db.get_stats()

        uptime = time.monotonic() - getattr(state, "start_time", 0)

        def _stats_of(component):
            """Read a component's stats snapshot, tolerating a dead thread."""
            if component is None:
                return None
            try:
                return component.stats
            except Exception:
                return None

        watchdog_stats = _stats_of(getattr(state, "watchdog", None))
        vlm_stats = _stats_of(getattr(state, "vlm_worker", None))
        buffer_stats = _stats_of(getattr(state, "buffer", None))
        capture_stats = [_stats_of(c) for c in getattr(state, "captures", [])]

        settings = state.settings

        return {
            "uptime_seconds": uptime,
            "inference_device": settings.inference_device,
            "pipeline_enabled": settings.enable_pipeline,
            "buffer": {
                "occupancy": buffer_stats.occupancy if buffer_stats else 0,
                "frames_written": buffer_stats.frames_written if buffer_stats else 0,
                "capacity": buffer_stats.capacity if buffer_stats else 0,
            },
            "cameras": [
                {
                    "camera_id": c.camera_id,
                    "connected": c.is_connected,
                    "fps": c.current_fps,
                    "frames_captured": c.frames_captured,
                }
                for c in capture_stats
                if c is not None
            ],
            "watchdog": {
                "is_running": watchdog_stats.is_running if watchdog_stats else False,
                "fps": watchdog_stats.current_fps if watchdog_stats else 0,
                "frames_processed": watchdog_stats.frames_processed if watchdog_stats else 0,
                "detections_published": (
                    watchdog_stats.detections_published if watchdog_stats else 0
                ),
                "tracks_active": watchdog_stats.tracks_active if watchdog_stats else 0,
                "cooldown_keys": watchdog_stats.cooldown_keys if watchdog_stats else 0,
            },
            "vlm": {
                "is_running": vlm_stats.is_running if vlm_stats else False,
                "model_loaded": vlm_stats.model_loaded if vlm_stats else False,
                "device": vlm_stats.device if vlm_stats else "n/a",
                "triggers_processed": vlm_stats.triggers_processed if vlm_stats else 0,
                "triggers_stale": vlm_stats.triggers_stale if vlm_stats else 0,
                "triggers_coalesced": vlm_stats.triggers_coalesced if vlm_stats else 0,
                "verdicts_positive": vlm_stats.verdicts_positive if vlm_stats else 0,
                # Whether per-subject judging is paying for itself: memoised
                # should climb far faster than judged once a scene settles, and
                # a rising shed count means the crowd outpaces the VLM.
                "instances_judged": vlm_stats.instances_judged if vlm_stats else 0,
                "instances_memoised": vlm_stats.instances_memoised if vlm_stats else 0,
                "instances_shed": vlm_stats.instances_shed if vlm_stats else 0,
                "avg_inference_ms": vlm_stats.avg_inference_ms if vlm_stats else 0,
            },
            "database": db_stats,
            "websocket_clients": ws_manager.active_count,
        }

    # ── WebSocket ────────────────────────────────────────────────

    @app.websocket("/ws/events")
    async def websocket_events(ws: WebSocket):
        await ws_manager.connect(ws)
        try:
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    return app


# Module-level app instance for `uvicorn app.api.server:app`
app = create_app()
