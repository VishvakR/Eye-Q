"""FastAPI REST and WebSocket routes for the Eye-Q control plane.

Endpoints:
    - ``/api/rules`` — CRUD for alert rules
    - ``/api/alerts`` — Paginated alert history
    - ``/api/cameras`` — Camera registration
    - ``/api/status`` — System health snapshot
    - ``/ws/events`` — WebSocket live event stream
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Request/Response Models ──────────────────────────────────────


class RuleCreate(BaseModel):
    camera_id: str
    name: str
    prompt: str
    target_labels: list[str] = Field(default_factory=list)
    roi_json: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.25
    cooldown_s: float = 5.0
    enabled: bool = True


class RuleUpdate(BaseModel):
    name: str | None = None
    prompt: str | None = None
    target_labels: list[str] | None = None
    roi_json: list[list[float]] | None = None
    confidence: float | None = None
    cooldown_s: float | None = None
    enabled: bool | None = None


class CameraCreate(BaseModel):
    id: str
    name: str
    source: str
    resolution: str = "1920x1080"
    fps: int = 30
    enabled: bool = True


# ── WebSocket Manager ────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections for live event broadcasting."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WebSocket connected. Active: {len(self._connections)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(f"WebSocket disconnected. Active: {len(self._connections)}")

    async def broadcast(self, message: dict) -> None:
        """Broadcast a JSON message to all connected clients."""
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()


# ── Router ───────────────────────────────────────────────────────

router = APIRouter()


# ── Rules ────────────────────────────────────────────────────────


@router.get("/api/rules")
async def list_rules(camera_id: Optional[str] = Query(None)):
    """List all alert rules, optionally filtered by camera."""
    from fastapi import Request
    # Access db from app state (injected at import won't work, we use dependency)
    # This is handled via the app.state pattern in server.py
    pass


@router.post("/api/rules", status_code=201)
async def create_rule(rule: RuleCreate):
    pass


@router.get("/api/rules/{rule_id}")
async def get_rule(rule_id: int):
    pass


@router.put("/api/rules/{rule_id}")
async def update_rule(rule_id: int, rule: RuleUpdate):
    pass


@router.delete("/api/rules/{rule_id}", status_code=204)
async def delete_rule(rule_id: int):
    pass


# ── Alerts ───────────────────────────────────────────────────────


@router.get("/api/alerts")
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    rule_id: Optional[int] = Query(None),
):
    pass


@router.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: int):
    pass


# ── Cameras ──────────────────────────────────────────────────────


@router.get("/api/cameras")
async def list_cameras():
    pass


@router.post("/api/cameras", status_code=201)
async def create_camera(camera: CameraCreate):
    pass


@router.delete("/api/cameras/{camera_id}", status_code=204)
async def delete_camera(camera_id: str):
    pass


# ── Status ───────────────────────────────────────────────────────


@router.get("/api/status")
async def get_status():
    pass


# ── WebSocket ────────────────────────────────────────────────────


@router.websocket("/ws/events")
async def websocket_events(ws: WebSocket):
    """Live event stream for the dashboard."""
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await ws.receive_text()
            # Client can send "ping" for keepalive
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
