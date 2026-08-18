"""Unit and integration tests for ZeroMQ PUB/SUB transport.

Tests cover:
    - Basic publish/subscribe round-trip with JSON payload
    - Topic-based filtering (subscriber only receives matching topics)
    - Non-blocking receive with timeout (returns None on timeout)
    - Multiple messages in sequence
    - Drain method for batch message retrieval
    - Message statistics tracking
    - Context manager usage
    - Malformed/edge-case payloads
"""

from __future__ import annotations

import time

from app.core.ipc import EventPublisher, EventSubscriber

# ── Helpers ──────────────────────────────────────────────────────

def _unique_endpoint() -> str:
    """Generate a unique ZMQ endpoint to avoid port conflicts between tests."""
    import random
    port = random.randint(15000, 25000)
    return f"tcp://127.0.0.1:{port}"


# ── Basic Round-Trip ─────────────────────────────────────────────

class TestBasicPubSub:
    """Test fundamental publish/subscribe message flow."""

    def test_single_message_roundtrip(self):
        """A single published message should be received by a subscriber."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="test") as sub:
                # Give ZMQ a moment to establish the connection
                time.sleep(0.2)

                payload = {"key": "value", "number": 42}
                pub.publish("test", payload)

                result = sub.receive(timeout_ms=2000)

                assert result is not None
                topic, data = result
                assert topic == "test"
                assert data["key"] == "value"
                assert data["number"] == 42

    def test_multiple_messages(self):
        """Multiple published messages should all be received in order."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="seq") as sub:
                time.sleep(0.2)

                for i in range(5):
                    pub.publish("seq", {"index": i})

                received = []
                for _ in range(5):
                    msg = sub.receive(timeout_ms=2000)
                    if msg:
                        received.append(msg[1]["index"])

                assert received == [0, 1, 2, 3, 4]


# ── Topic Filtering ──────────────────────────────────────────────

class TestTopicFiltering:
    """Test that subscribers only receive messages matching their topic."""

    def test_subscriber_filters_by_topic(self):
        """Subscriber should only receive messages on its subscribed topic."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="alpha") as sub:
                time.sleep(0.2)

                pub.publish("alpha", {"msg": "yes"})
                pub.publish("beta", {"msg": "no"})
                pub.publish("alpha", {"msg": "also_yes"})

                time.sleep(0.1)

                results = sub.drain(max_messages=10)

                topics = [t for t, _ in results]
                payloads = [d["msg"] for _, d in results]

                assert "beta" not in topics
                assert "yes" in payloads
                assert "also_yes" in payloads

    def test_subscribe_all_topics(self):
        """Subscribing to empty string should receive all topics."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="") as sub:
                time.sleep(0.2)

                pub.publish("topic_a", {"a": 1})
                pub.publish("topic_b", {"b": 2})

                time.sleep(0.1)

                results = sub.drain(max_messages=10)
                topics = {t for t, _ in results}

                assert "topic_a" in topics
                assert "topic_b" in topics


# ── Timeout Behavior ─────────────────────────────────────────────

class TestTimeout:
    """Test non-blocking receive with timeout."""

    def test_receive_timeout_returns_none(self):
        """receive() should return None when no message arrives within timeout."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as _pub:
            with EventSubscriber(endpoint, topic="none") as sub:
                start = time.monotonic()
                result = sub.receive(timeout_ms=200)
                elapsed = time.monotonic() - start

                assert result is None
                # Should have waited roughly 200ms (±100ms tolerance)
                assert 0.1 <= elapsed <= 0.5

    def test_receive_zero_timeout(self):
        """receive(timeout_ms=0) should return immediately if no message."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as _pub:
            with EventSubscriber(endpoint, topic="imm") as sub:
                start = time.monotonic()
                result = sub.receive(timeout_ms=0)
                elapsed = time.monotonic() - start

                assert result is None
                assert elapsed < 0.05  # Should be nearly instant


# ── Drain ────────────────────────────────────────────────────────

class TestDrain:
    """Test bulk message retrieval with drain()."""

    def test_drain_multiple_messages(self):
        """drain() should return all pending messages."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="bulk") as sub:
                time.sleep(0.2)

                for i in range(10):
                    pub.publish("bulk", {"i": i})

                time.sleep(0.1)

                results = sub.drain(max_messages=100)

                assert len(results) == 10
                indices = [d["i"] for _, d in results]
                assert indices == list(range(10))

    def test_drain_empty(self):
        """drain() on empty queue should return empty list."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as _pub:
            with EventSubscriber(endpoint, topic="empty") as sub:
                time.sleep(0.1)
                results = sub.drain()
                assert results == []


# ── Statistics ───────────────────────────────────────────────────

class TestStatistics:
    """Test message counters."""

    def test_publisher_message_count(self):
        """Publisher should track the number of sent messages."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            assert pub.messages_sent == 0

            for i in range(5):
                pub.publish("stats", {"i": i})

            assert pub.messages_sent == 5

    def test_subscriber_message_count(self):
        """Subscriber should track the number of received messages."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="cnt") as sub:
                time.sleep(0.2)

                for i in range(3):
                    pub.publish("cnt", {"i": i})

                time.sleep(0.1)

                for _ in range(3):
                    sub.receive(timeout_ms=500)

                assert sub.messages_received == 3


# ── Complex Payloads ─────────────────────────────────────────────

class TestComplexPayloads:
    """Test various payload types and edge cases."""

    def test_nested_payload(self):
        """Nested dict/list payloads should survive serialization."""
        endpoint = _unique_endpoint()

        payload = {
            "detection": {
                "label": "person",
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                "tags": ["alert", "zone_a"],
            },
            "confidence": 0.95,
        }

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="nested") as sub:
                time.sleep(0.2)

                pub.publish("nested", payload)
                result = sub.receive(timeout_ms=2000)

                assert result is not None
                _, data = result
                assert data["detection"]["label"] == "person"
                assert data["detection"]["bbox"]["x1"] == 0.1
                assert data["detection"]["tags"] == ["alert", "zone_a"]
                assert data["confidence"] == 0.95

    def test_empty_payload(self):
        """Empty dict payload should work."""
        endpoint = _unique_endpoint()

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="empty") as sub:
                time.sleep(0.2)

                pub.publish("empty", {})
                result = sub.receive(timeout_ms=2000)

                assert result is not None
                _, data = result
                assert data == {}


# ── Repr ─────────────────────────────────────────────────────────

class TestRepr:
    """Test string representations."""

    def test_publisher_repr(self):
        endpoint = _unique_endpoint()
        with EventPublisher(endpoint) as pub:
            r = repr(pub)
            assert "EventPublisher" in r
            assert "active" in r

    def test_subscriber_repr(self):
        endpoint = _unique_endpoint()
        with EventPublisher(endpoint) as _pub:
            with EventSubscriber(endpoint, topic="repr") as sub:
                r = repr(sub)
                assert "EventSubscriber" in r
                assert "repr" in r
