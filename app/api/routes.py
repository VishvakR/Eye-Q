"""Shared API types and the WebSocket connection manager.

The REST handlers themselves live in [server.py](server.py), where they can
reach ``app.state`` for the database and pipeline components. This module
holds only what those handlers and the alert sink need in common: request
models and the broadcast hub.

Example:
    >>> await ws_manager.broadcast({"type": "alert", "data": alert_row})
"""

from __future__ import annotations

import logging

from fastapi import WebSocket
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Request Models ───────────────────────────────────────────────


class RuleCreate(BaseModel):
    """Payload for ``POST /api/rules``."""

    camera_id: str
    name: str
    prompt: str
    target_labels: list[str] = Field(default_factory=list)
    roi_json: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.25
    cooldown_s: float = 5.0
    enabled: bool = True


class RuleUpdate(BaseModel):
    """Payload for ``PUT /api/rules/{id}``. Only provided fields are written."""

    name: str | None = None
    prompt: str | None = None
    target_labels: list[str] | None = None
    roi_json: list[list[float]] | None = None
    confidence: float | None = None
    cooldown_s: float | None = None
    enabled: bool | None = None


class CameraCreate(BaseModel):
    """Payload for ``POST /api/cameras``."""

    id: str
    name: str
    source: str
    resolution: str = "1280x720"
    fps: int = 30
    enabled: bool = True


# ── WebSocket Manager ────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections for live event broadcasting.

    A single instance (:data:`ws_manager`) is shared by the ``/ws/events``
    endpoint and the alert sink, so verdicts produced on the pipeline side
    reach every connected dashboard.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        """Accept a connection and add it to the broadcast set."""
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WebSocket connected. Active: {len(self._connections)}")

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a connection from the broadcast set."""
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(f"WebSocket disconnected. Active: {len(self._connections)}")

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected clients, dropping dead ones."""
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
        """Number of currently connected clients."""
        return len(self._connections)


ws_manager = ConnectionManager()
