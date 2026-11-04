"""Shared API types and the WebSocket connection manager.

The REST handlers themselves live in [server.py](server.py), where they can
reach ``app.state`` for the database and pipeline components. This module
holds only what those handlers and the alert sink need in common: request
models and the broadcast hub.

Example:
    >>> await ws_manager.broadcast({"type": "alert", "data": alert_row})
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# How long one client may take to accept a broadcast before it is dropped. A
# sleeping laptop with a full TCP buffer otherwise held every alert — storage
# included, since the sink awaits the broadcast — hostage to its send.
_BROADCAST_SEND_TIMEOUT_S = 2.0


def _validate_roi(vertices: list[list[float]] | None) -> list[list[float]] | None:
    """An ROI is empty (full frame) or a real polygon in normalised coordinates.

    Rejected here rather than downstream: ``sync_rules`` treats a polygon it
    cannot parse as *no* ROI, so a malformed one silently widened the rule to
    the whole frame.
    """
    if not vertices:
        return vertices
    if len(vertices) < 3:
        raise ValueError(f"roi_json needs at least 3 vertices, got {len(vertices)}")
    for vertex in vertices:
        if len(vertex) != 2:
            raise ValueError(f"each roi_json vertex must be [x, y], got {vertex}")
        if not all(0.0 <= coord <= 1.0 for coord in vertex):
            raise ValueError(f"roi_json coordinates are normalised to 0-1, got {vertex}")
    return vertices


# ── Request Models ───────────────────────────────────────────────


class RuleCreate(BaseModel):
    """Payload for ``POST /api/rules``."""

    camera_id: str
    name: str
    prompt: str
    target_labels: list[str] = Field(default_factory=list)
    roi_json: list[list[float]] = Field(default_factory=list)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    cooldown_s: float = Field(default=5.0, ge=0.0)
    enabled: bool = True
    pair_label: str = Field(
        default="",
        description=(
            "Require this COCO class to have an overlapping box before "
            "target_labels trigger — 'cell phone' to ask only about people "
            "actually holding one. Empty disables the requirement."
        ),
    )

    _check_roi = field_validator("roi_json")(_validate_roi)


class RuleUpdate(BaseModel):
    """Payload for ``PUT /api/rules/{id}``. Only provided fields are written.

    ``None`` means "not provided", never "set to null": every rules column is
    NOT NULL, so an explicit null used to reach SQLite and come back as a 500.
    """

    name: str | None = None
    prompt: str | None = None
    target_labels: list[str] | None = None
    roi_json: list[list[float]] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    cooldown_s: float | None = Field(default=None, ge=0.0)
    enabled: bool | None = None
    pair_label: str | None = None

    _check_roi = field_validator("roi_json")(_validate_roi)

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> RuleUpdate:
        nulls = sorted(f for f in self.model_fields_set if getattr(self, f) is None)
        if nulls:
            raise ValueError(f"these fields cannot be null: {', '.join(nulls)}")
        return self


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
        """Send a JSON message to all connected clients, dropping dead ones.

        Concurrent, and each send is bounded. Sequential sends meant one
        stalled client delayed every client after it — and the alert sink
        awaits this, so it delayed storing the next verdict too. Iterates over
        a snapshot because ``disconnect`` can run while a send is suspended.
        """
        clients = list(self._connections)
        if not clients:
            return

        async def send(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_json(message), _BROADCAST_SEND_TIMEOUT_S)
                return None
            except Exception:
                return ws

        for ws in await asyncio.gather(*(send(ws) for ws in clients)):
            if ws is not None:
                self.disconnect(ws)

    @property
    def active_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._connections)


ws_manager = ConnectionManager()
