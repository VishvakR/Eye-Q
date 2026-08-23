"""Alert sink — the bridge from Stage 2 back into the control plane.

This closes the last hop of the cascade. The VLM worker publishes a
``vlm_result`` event on its ZMQ endpoint; without a consumer those verdicts
went nowhere, so the ``alerts`` table stayed empty and the dashboard's live
feed could never populate.

For each verdict this task:
    1. Persists an ``alerts`` row (both violations *and* clears — the history
       is only trustworthy if negatives are recorded too)
    2. Broadcasts the stored row over the ``/ws/events`` WebSocket
    3. Fires the webhook for confirmed violations only

It runs as an asyncio task on the FastAPI event loop, so SQLite writes,
WebSocket sends, and webhook POSTs all stay non-blocking.

Example:
    >>> sink = AlertSink(db, ws_manager, settings)
    >>> await sink.start()
    >>> # ... alerts flow into SQLite and out to connected dashboards ...
    >>> await sink.stop()
"""

from __future__ import annotations

import asyncio
import logging

from app.core.database import DEFAULT_RULE_ID, DEFAULT_RULE_NAME
from app.core.ipc import AsyncEventSubscriber
from app.schemas.events import VLMResult
from app.services.webhook import WebhookDispatcher

logger = logging.getLogger(__name__)


__all__ = ["AlertSink", "DEFAULT_RULE_ID", "DEFAULT_RULE_NAME"]


class AlertSink:
    """Consumes ``vlm_result`` events and turns them into alerts.

    Args:
        db: Connected :class:`~app.core.database.Database`.
        ws_manager: WebSocket connection manager used to broadcast alerts.
        settings: Application settings (endpoint + webhook config).
    """

    def __init__(self, db, ws_manager, settings) -> None:
        self._db = db
        self._ws = ws_manager
        self._settings = settings
        self._webhook = WebhookDispatcher(
            settings.webhook_url, settings.webhook_timeout_s
        )

        self._subscriber: AsyncEventSubscriber | None = None
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

        self._alerts_stored = 0
        self._violations = 0

    @property
    def alerts_stored(self) -> int:
        """Number of alerts persisted since startup."""
        return self._alerts_stored

    async def start(self) -> None:
        """Seed the sentinel rule, connect the subscriber, and begin consuming."""
        await self._db.ensure_default_rule()

        self._subscriber = AsyncEventSubscriber(
            self._settings.zmq_result_endpoint,
            topic="vlm_result",
        )
        self._stopping.clear()
        self._task = asyncio.create_task(self._consume_loop(), name="AlertSink")

        logger.info(
            f"AlertSink listening on {self._settings.zmq_result_endpoint} "
            f"(topic='vlm_result')."
        )

    async def stop(self) -> None:
        """Stop consuming and release the subscriber, webhook client, and task."""
        self._stopping.set()

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None

        await self._webhook.close()

        logger.info(f"AlertSink stopped. Alerts stored: {self._alerts_stored}.")


    async def _consume_loop(self) -> None:
        """Receive verdicts until cancelled, isolating per-message failures."""
        assert self._subscriber is not None

        while not self._stopping.is_set():
            try:
                message = await self._subscriber.receive(timeout_ms=500)
                if message is None:
                    continue

                _topic, payload = message
                await self._handle_result(payload)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One malformed message must never kill the sink.
                logger.error(f"AlertSink failed to handle a result: {exc}", exc_info=True)

    async def _handle_result(self, payload: dict) -> None:
        """Persist, broadcast, and optionally webhook a single verdict."""
        result = VLMResult(**payload)
        detection = result.query.detection

        rule_id = result.query.rule_id
        if await self._db.get_rule(rule_id) is None:
            # Rule was deleted between trigger and verdict — keep the evidence.
            logger.debug(f"Rule {rule_id} no longer exists; filing under sentinel rule.")
            rule_id = DEFAULT_RULE_ID

        alert_id = await self._db.create_alert(
            rule_id=rule_id,
            camera_id=detection.camera_id,
            timestamp=detection.timestamp,
            verdict=result.verdict,
            raw_response=result.raw_response,
            inference_ms=result.inference_time_ms,
            bbox_json=detection.bbox.model_dump(),
            frame_snapshot=result.snapshot_path,
            track_id=detection.track_id,
            label=detection.label,
            confidence=result.confidence,
        )

        self._alerts_stored += 1
        if result.verdict:
            self._violations += 1

        # Broadcast the stored row, not the VLMResult: the dashboard renders
        # live cards and paginated history with the same code path, and that
        # code reads DB column names (verdict as 0/1, created_at, ...).
        alert_row = await self._db.get_alert(alert_id)
        if alert_row is not None:
            alert_row["rule_name"] = result.query.rule_name
            alert_row["label"] = detection.label
            # Two different confidences meet here and they are not
            # interchangeable. ``confidence`` is the VLM's p(yes), read off the
            # stored row, and it is what decides whether an alert was a close
            # call. ``detection_confidence`` is YOLO's box score — how sure the
            # detector was that this was a person at all. This line previously
            # published the detector's score under the name ``confidence``,
            # so every dashboard card and webhook payload carried a number that
            # meant something entirely different from what it claimed.
            alert_row["detection_confidence"] = detection.confidence
            await self._ws.broadcast({"type": "alert", "data": alert_row})

        if result.verdict and self._webhook.enabled:
            # Fire-and-forget so a slow endpoint cannot stall the sink.
            asyncio.create_task(
                self._webhook.dispatch(
                    {
                        "alert_id": alert_id,
                        "rule_id": rule_id,
                        "rule_name": result.query.rule_name,
                        "prompt": result.query.prompt,
                        "camera_id": detection.camera_id,
                        "label": detection.label,
                        "track_id": detection.track_id,
                        # See the note above: the VLM's p(yes) and the
                        # detector's box score are different questions, and
                        # this payload used to send the latter as "confidence".
                        "confidence": result.confidence,
                        "detection_confidence": detection.confidence,
                        "verdict": result.verdict,
                        "raw_response": result.raw_response,
                        "inference_ms": result.inference_time_ms,
                        "model": result.model_name,
                    }
                )
            )
