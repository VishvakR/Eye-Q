"""FastAPI application factory with full component lifecycle management.

This module creates the FastAPI app, wires up all REST/WS routes with
database access via ``app.state``, and manages the startup/shutdown
lifecycle of all system components (buffer, capture, YOLO, VLM, ZMQ).

Usage:
    >>> from app.api.server import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn app.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import (
    CameraCreate,
    ConnectionManager,
    RuleCreate,
    RuleUpdate,
    ws_manager,
)
from app.core.database import Database

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of all system components.

    Startup: initialize DB → buffer → ZMQ → capture → YOLO → VLM.
    Shutdown: VLM → YOLO → capture → ZMQ → buffer → DB (reverse order).
    """
    from app.config.settings import get_settings

    settings = get_settings()

    # ── Database ─────────────────────────────────────────────────
    db = await Database.create(settings.database_path)
    await db.initialize_tables()
    app.state.db = db
    app.state.settings = settings
    app.state.start_time = time.monotonic()

    # Components will be started by main.py orchestrator
    # and attached to app.state there. Here we just handle DB.
    app.state.buffer = None
    app.state.watchdog = None
    app.state.vlm_worker = None
    app.state.captures = []

    logger.info("FastAPI server started.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    await db.close()
    logger.info("FastAPI server shut down.")


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
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir)),
            name="static",
        )

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

    # ── Rules CRUD ───────────────────────────────────────────────

    @app.get("/api/rules")
    async def list_rules(
        request: Request,
        camera_id: Optional[str] = Query(None),
    ):
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
        return await db.get_rule(rule_id)

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
        return updated

    @app.delete("/api/rules/{rule_id}", status_code=204)
    async def delete_rule(request: Request, rule_id: int):
        db: Database = request.app.state.db
        deleted = await db.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(404, f"Rule {rule_id} not found")

    # ── Alerts ───────────────────────────────────────────────────

    @app.get("/api/alerts")
    async def list_alerts(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        rule_id: Optional[int] = Query(None),
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
        db: Database = request.app.state.db
        db_stats = await db.get_stats()

        uptime = time.monotonic() - getattr(request.app.state, "start_time", 0)

        # Gather component stats
        watchdog_stats = None
        if hasattr(request.app.state, "watchdog") and request.app.state.watchdog:
            try:
                watchdog_stats = request.app.state.watchdog.stats
            except Exception:
                pass

        vlm_stats = None
        if hasattr(request.app.state, "vlm_worker") and request.app.state.vlm_worker:
            try:
                vlm_stats = request.app.state.vlm_worker.stats
            except Exception:
                pass

        buffer_stats = None
        if hasattr(request.app.state, "buffer") and request.app.state.buffer:
            try:
                buffer_stats = request.app.state.buffer.stats
            except Exception:
                pass

        settings = request.app.state.settings

        return {
            "uptime_seconds": uptime,
            "inference_device": settings.inference_device,
            "buffer": {
                "occupancy": buffer_stats.occupancy if buffer_stats else 0,
                "frames_written": buffer_stats.frames_written if buffer_stats else 0,
                "capacity": buffer_stats.capacity if buffer_stats else 0,
            },
            "watchdog": {
                "is_running": watchdog_stats.is_running if watchdog_stats else False,
                "fps": watchdog_stats.current_fps if watchdog_stats else 0,
                "frames_processed": watchdog_stats.frames_processed if watchdog_stats else 0,
                "detections_published": watchdog_stats.detections_published if watchdog_stats else 0,
            },
            "vlm": {
                "is_running": vlm_stats.is_running if vlm_stats else False,
                "model_loaded": vlm_stats.model_loaded if vlm_stats else False,
                "triggers_processed": vlm_stats.triggers_processed if vlm_stats else 0,
                "verdicts_positive": vlm_stats.verdicts_positive if vlm_stats else 0,
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
