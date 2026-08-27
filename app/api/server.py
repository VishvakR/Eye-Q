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
import functools
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
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

    Co-occurrence requirements compose under that same rule, which has a
    consequence worth knowing: a pair requirement is a *narrowing*, and there
    is no coarsest narrowing. One plain "person" rule alongside one
    "person + cell phone" rule means the only composition that keeps the plain
    rule firing is no gate at all, so the pair requirement is dropped and every
    person still wakes the VLM. That is correct but invisible, so the drop is
    logged as a warning naming the rule responsible.
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

    # ── Co-occurrence requirements ───────────────────────────────
    # Composed under the same "never filter out something a rule still wants"
    # rule that governs labels above. A label carries a pair requirement only
    # if *every* enabled rule targeting it asks for one; a single unpaired rule
    # on that label drops the gate, because enforcing it would silently blind
    # that rule. Paying in wasted inference is recoverable, missing the event
    # the appliance exists to catch is not.
    pair_req: dict[str, set[str]] = {}
    unpaired: dict[str, str] = {}
    paired_by: dict[str, str] = {}
    wildcard_rule: str | None = None

    for rule in rules:
        labels = _parse_labels(rule)
        pair = (rule.get("pair_label") or "").strip()
        if not labels:
            # A rule with no label filter must see everything, so it suppresses
            # every gate for the same reason it forces the label scope open.
            wildcard_rule = wildcard_rule or rule["name"]
            continue
        for label in labels:
            if pair:
                pair_req.setdefault(label, set()).add(pair)
                paired_by.setdefault(label, rule["name"])
            else:
                unpaired.setdefault(label, rule["name"])

    if wildcard_rule is not None and pair_req:
        logger.warning(
            f"Rule '{wildcard_rule}' has no label filter, so it must see every "
            f"detection — the pair requirement(s) on {sorted(pair_req)} cannot "
            f"be enforced at Stage 1 and every detection will still wake the VLM."
        )
        pair_req.clear()

    for label in sorted(set(unpaired) & set(pair_req)):
        partners = " or ".join(sorted(pair_req[label]))
        logger.warning(
            f"Rule '{unpaired[label]}' targets '{label}' with no pair_label, so "
            f"the '{label} + {partners}' gate on rule '{paired_by[label]}' cannot "
            f"be enforced at Stage 1 — every {label} will still wake the VLM."
        )
        pair_req.pop(label)

    watchdog.update_pair_requirements(pair_req or None)

    if scope is None:
        scope_desc = "all"
    elif scope:
        scope_desc = str(sorted(scope))
    else:
        scope_desc = "none — Stage 2 idle until a rule is enabled"

    pair_desc = (
        "; pairs: " + ", ".join(f"{k}+{'/'.join(sorted(v))}" for k, v in sorted(pair_req.items()))
        if pair_req
        else ""
    )
    logger.info(
        f"Synced {len(rules)} enabled rule(s) to the pipeline "
        f"(triggers on: {scope_desc}{pair_desc})."
    )


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
        # The same floor Stage 2 applies, moved to where it is cheapest. One
        # setting for both stages on purpose: they call the same
        # BoundingBox.pixel_size on the same fixed-shape ring buffer, so the
        # predicate cannot drift. Expect vlm.instances_too_small to read zero
        # from here on — that is Stage 1 shedding first, not a broken counter.
        min_subject_px=settings.vlm_min_subject_px,
        pair_overlap=settings.yolo_pair_overlap,
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
            max_crop_edge=settings.vlm_max_crop_edge,
            min_subject_px=settings.vlm_min_subject_px,
            batch_size=settings.vlm_batch_size,
            square_edge=settings.vlm_square_edge,
            probe=settings.vlm_probe,
            probe_threshold=settings.vlm_probe_threshold,
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
        # Keep only the newest box per subject. recent_detections() returns
        # every frame from the last second, so without this a stationary person
        # is drawn ~30 times over — and the smearing gets worse the busier the
        # scene, which is exactly when the operator needs to read it.
        latest: dict[tuple[str, int], tuple] = {}
        for event, triggered in watchdog.recent_detections(max_age_s=1.0):
            latest[(event.label, event.track_id)] = (event, triggered)

        for event, triggered in latest.values():
            draw_detection(
                frame,
                event.bbox,
                event.label,
                event.confidence,
                triggered,
                track_id=event.track_id,
            )

    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)))

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else None


@functools.lru_cache(maxsize=8)
def _placeholder_jpeg(width: int, quality: int, message: str) -> bytes:
    """A dark card carrying ``message``, encoded once and reused.

    The stream must always have something to send. See
    :func:`_mjpeg_generator` for why silence is worse than a placeholder.
    """
    height = max(90, int(width * 9 / 16))
    frame = np.full((height, width, 3), 15, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, width / 1280)
    thickness = max(1, int(scale * 1.5))
    (text_w, text_h), _ = cv2.getTextSize(message, font, scale, thickness)

    cv2.putText(
        frame,
        message,
        ((width - text_w) // 2, (height + text_h) // 2),
        font,
        scale,
        (120, 120, 130),
        thickness,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else b""


async def _mjpeg_generator(app: FastAPI):
    """Yield ``multipart/x-mixed-replace`` parts at the configured frame rate.

    Each part is preceded by CRLF because RFC 2046 defines the delimiter as
    ``CRLF--boundary``. Emitting the first boundary without it is tolerated by
    Chrome and Firefox but leaves Safari unable to parse the stream at all.

    Two rules keep the panel alive, both learned from it going blank:

    **Never go silent.** A stream carrying no parts is not an error — the HTTP
    response is open and healthy, it simply has nothing in it. The browser's
    ``<img>`` therefore never fires ``onerror``, the snapshot fallback never
    engages, and the operator gets an empty box with no indication anything is
    wrong. That happens routinely: for the seconds before the first frame is
    captured, and again whenever a camera drops and ``read_latest`` has nothing
    to hand back. So when there is no frame, say so in a placeholder rather
    than sending nothing.

    **Never die.** A single exception here ends the response for good, and an
    ``<img>`` that has already started rendering will not reconnect. One
    corrupt frame must not cost the operator their camera feed until they
    happen to reload the page.
    """
    settings = app.state.settings
    interval = 1.0 / settings.stream_fps
    consecutive_failures = 0

    while True:
        started = time.monotonic()

        try:
            jpeg = await asyncio.to_thread(
                _render_preview_frame, app, settings.stream_quality, settings.stream_width
            )
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            jpeg = None
            # Log the first failure in a run at full volume, then back off:
            # a persistently broken camera should not fill the disk.
            if consecutive_failures == 1 or consecutive_failures % 100 == 0:
                logger.error(
                    f"Preview render failed ({consecutive_failures}x): {exc}",
                    exc_info=consecutive_failures == 1,
                )

        if jpeg is None:
            jpeg = _placeholder_jpeg(
                settings.stream_width,
                settings.stream_quality,
                "Camera unavailable" if consecutive_failures else "Waiting for camera...",
            )

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

    # Alert evidence JPEGs, served from wherever the VLM worker writes them.
    # This has to follow ``snapshot_dir`` rather than hardcode the default:
    # the worker honours the setting, so a mount pinned to "data/snapshots"
    # silently serves 404s to every thumbnail the moment EYEQ_SNAPSHOT_DIR
    # points somewhere else. Created eagerly so the mount never 500s on a
    # fresh checkout that has not yet produced an alert.
    from app.config.settings import get_settings

    snapshot_dir = Path(get_settings().snapshot_dir)
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

    @app.get("/api/labels")
    async def list_labels(prompt: str | None = Query(None)):
        """The classes Stage 1 can detect, and which ones a prompt is about.

        The dashboard uses ``suggested`` to catch the common misconfiguration
        where a rule asks about people but watches all eighty classes, and so
        spends its inference budget on pallets and trucks.
        """
        return {
            "labels": list(COCO_CLASSES),
            "suggested": labels_in_prompt(prompt) if prompt else [],
        }

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
            pair_label=rule.pair_label,
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

        warning = scope_warning(
            updated.get("prompt", ""),
            json.loads(updated.get("target_labels") or "[]"),
        )
        if warning:
            logger.warning(f"Rule {rule_id}: {warning}")
            updated = {**updated, "warning": warning}
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
                # Detections dropped for carrying too few captured pixels to
                # answer about. Rising means the camera is too far for the
                # floor, not that the scene is quiet — lower vlm_min_subject_px
                # rather than accept the blind spot, since a skipped subject is
                # never judged at all. Its Stage 2 twin, vlm.instances_too_small,
                # correctly reads zero while Stage 1 sheds first.
                "detections_filtered_size": (
                    watchdog_stats.detections_filtered_size if watchdog_stats else 0
                ),
                # Detections held back because a rule's pair_label was absent
                # from the frame or not overlapping. This is the gate working.
                # A value that tracks detections_total means the partner class
                # is never detected at all — check yolo_confidence, which small
                # classes like 'cell phone' rarely clear at the 0.25 default.
                "detections_filtered_pair": (
                    watchdog_stats.detections_filtered_pair if watchdog_stats else 0
                ),
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
                "instances_too_small": (
                    vlm_stats.instances_too_small if vlm_stats else 0
                ),
                # Whether Stage 2 has a working endpoint at all. A backend
                # that is down looks identical to a quiet scene in every
                # counter above — both just stop producing verdicts.
                "backend_unavailable": (
                    vlm_stats.backend_unavailable if vlm_stats else False
                ),
                "batches_dropped_unavailable": (
                    vlm_stats.batches_dropped_unavailable if vlm_stats else 0
                ),
                "avg_inference_ms": vlm_stats.avg_inference_ms if vlm_stats else 0,
                # Capture-to-verdict, the number an operator actually waits
                # through. Compare against avg_inference_ms: when the two are
                # close the appliance is keeping up, and when latency_p95
                # climbs away from it the queue is the problem, not the model.
                "avg_queued_ms": vlm_stats.avg_queued_ms if vlm_stats else 0,
                "latency_p50_ms": vlm_stats.latency_p50_ms if vlm_stats else 0,
                "latency_p95_ms": vlm_stats.latency_p95_ms if vlm_stats else 0,
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
