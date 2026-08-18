"""Tests for the Stage 2 → control-plane alert sink.

This is the regression suite for the defect that made Eye-Q non-functional:
the VLM worker published ``vlm_result`` events that nothing consumed, so no
alert was ever stored, broadcast, or webhooked. Every test here asserts that
a published verdict actually lands somewhere observable.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.config.settings import Settings
from app.core.database import Database
from app.core.ipc import EventPublisher
from app.schemas.events import BoundingBox, DetectionEvent, VLMQuery, VLMResult
from app.services.alert_sink import DEFAULT_RULE_ID, AlertSink

# Ports well away from the defaults so a running instance cannot interfere.
TEST_RESULT_ENDPOINT = "tcp://127.0.0.1:5591"


class FakeWSManager:
    """Stands in for the WebSocket ConnectionManager, recording broadcasts."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def active_count(self) -> int:
        return 0


def make_result(
    verdict: bool = True,
    rule_id: int = DEFAULT_RULE_ID,
    raw_response: str = "Yes, a person is blocking the fire exit.",
    snapshot_path: str | None = None,
) -> VLMResult:
    """Build a VLMResult exactly as the worker would publish it."""
    detection = DetectionEvent(
        timestamp=time.monotonic(),
        camera_id="test_cam",
        label="person",
        confidence=0.91,
        bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.4, y2=0.6),
        frame_index=3,
    )
    return VLMResult(
        query=VLMQuery(
            detection=detection,
            rule_id=rule_id,
            rule_name="Fire exit check",
            prompt="Is a person blocking the fire exit?",
        ),
        verdict=verdict,
        raw_response=raw_response,
        inference_time_ms=812.5,
        model_name="moondream2",
        snapshot_path=snapshot_path,
    )


@pytest.fixture
async def db(tmp_path):
    """A fresh SQLite database per test."""
    database = await Database.create(str(tmp_path / "test.db"))
    await database.initialize_tables()
    yield database
    await database.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at the test endpoint with webhooks disabled."""
    return Settings(
        database_path=str(tmp_path / "test.db"),
        zmq_result_endpoint=TEST_RESULT_ENDPOINT,
        webhook_url=None,
    )


async def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it is true or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def _publish_until_consumed(publisher, result: VLMResult, predicate) -> bool:
    """Publish repeatedly until the sink reacts.

    ZMQ PUB/SUB is a slow joiner: messages sent before the subscriber's
    subscription registers are silently dropped. Retrying removes that race
    from the test without a fixed sleep.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        publisher.publish("vlm_result", result.model_dump())
        if await _wait_for(predicate, timeout=0.3):
            return True
    return False


class TestAlertSink:
    """The sink must persist, broadcast, and correctly attribute verdicts."""

    async def test_verdict_is_persisted_and_broadcast(self, db, settings):
        """A published verdict lands in SQLite *and* on the WebSocket.

        This is the exact path that was severed: worker → ZMQ → sink → DB/WS.
        """
        ws = FakeWSManager()
        sink = AlertSink(db, ws, settings)
        await sink.start()

        publisher = EventPublisher(TEST_RESULT_ENDPOINT)
        try:
            delivered = await _publish_until_consumed(
                publisher, make_result(), lambda: sink.alerts_stored > 0
            )
            assert delivered, "AlertSink never consumed the published verdict"

            alerts = await db.list_alerts()
            assert len(alerts) >= 1
            assert alerts[0]["verdict"] == 1
            assert "fire exit" in alerts[0]["raw_response"].lower()
            assert alerts[0]["inference_ms"] == pytest.approx(812.5)

            assert ws.messages, "No WebSocket broadcast was emitted"
            message = ws.messages[0]
            assert message["type"] == "alert"
            # The dashboard renders live cards and history with one code path,
            # so the broadcast must carry DB column names.
            assert message["data"]["verdict"] == 1
            assert "created_at" in message["data"]
            assert message["data"]["rule_name"] == "Fire exit check"
        finally:
            publisher.close()
            await sink.stop()

    async def test_clear_verdicts_are_recorded_too(self, db, settings):
        """Negative verdicts are stored — history is useless without them."""
        ws = FakeWSManager()
        sink = AlertSink(db, ws, settings)
        await sink.start()

        publisher = EventPublisher(TEST_RESULT_ENDPOINT)
        try:
            result = make_result(verdict=False, raw_response="No, the exit is clear.")
            delivered = await _publish_until_consumed(
                publisher, result, lambda: sink.alerts_stored > 0
            )
            assert delivered

            alerts = await db.list_alerts()
            assert alerts[0]["verdict"] == 0
        finally:
            publisher.close()
            await sink.stop()

    async def test_snapshot_path_is_stored(self, db, settings):
        """The evidence JPEG path survives into the alerts row."""
        ws = FakeWSManager()
        sink = AlertSink(db, ws, settings)
        await sink.start()

        publisher = EventPublisher(TEST_RESULT_ENDPOINT)
        try:
            result = make_result(snapshot_path="data/snapshots/abc123.jpg")
            delivered = await _publish_until_consumed(
                publisher, result, lambda: sink.alerts_stored > 0
            )
            assert delivered

            alerts = await db.list_alerts()
            assert alerts[0]["frame_snapshot"] == "data/snapshots/abc123.jpg"
        finally:
            publisher.close()
            await sink.stop()

    async def test_deleted_rule_falls_back_to_sentinel(self, db, settings):
        """A verdict for a since-deleted rule is kept, not lost to the FK.

        ``alerts.rule_id`` references ``rules``, so a verdict arriving after
        its rule was deleted would otherwise be rejected mid-insert.
        """
        ws = FakeWSManager()
        sink = AlertSink(db, ws, settings)
        await sink.start()

        publisher = EventPublisher(TEST_RESULT_ENDPOINT)
        try:
            result = make_result(rule_id=99999)  # Never existed
            delivered = await _publish_until_consumed(
                publisher, result, lambda: sink.alerts_stored > 0
            )
            assert delivered

            alerts = await db.list_alerts()
            assert alerts[0]["rule_id"] == DEFAULT_RULE_ID
        finally:
            publisher.close()
            await sink.stop()

    async def test_sentinel_rule_is_seeded_and_disabled(self, db, settings):
        """Startup seeds the fallback rule, but never as an active rule."""
        sink = AlertSink(db, FakeWSManager(), settings)
        await sink.start()
        try:
            sentinel = await db.get_rule(DEFAULT_RULE_ID)
            assert sentinel is not None
            # Disabled, so it never inflates the "active rules" stat.
            assert sentinel["enabled"] == 0
        finally:
            await sink.stop()

    async def test_malformed_message_does_not_kill_the_sink(self, db, settings):
        """One bad payload must not stop later verdicts from being stored."""
        ws = FakeWSManager()
        sink = AlertSink(db, ws, settings)
        await sink.start()

        publisher = EventPublisher(TEST_RESULT_ENDPOINT)
        try:
            publisher.publish("vlm_result", {"not": "a valid VLMResult"})
            await asyncio.sleep(0.2)

            delivered = await _publish_until_consumed(
                publisher, make_result(), lambda: sink.alerts_stored > 0
            )
            assert delivered, "Sink stopped consuming after a malformed message"
        finally:
            publisher.close()
            await sink.stop()
