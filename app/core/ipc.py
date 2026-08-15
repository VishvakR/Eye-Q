"""ZeroMQ PUB/SUB transport for inter-component event messaging.

This module provides thin wrappers around ``pyzmq`` for publishing and
subscribing to JSON events over ZMQ multipart messages. The protocol uses
two-frame messages: ``[topic_bytes, json_payload_bytes]``.

Design choices:
    - ``tcp://`` transport for maximum portability (works across processes
      and machines, with ~30μs latency on localhost — negligible vs VLM
      inference times of 500ms+).
    - Non-blocking sends with ``zmq.NOBLOCK`` — if the subscriber's queue
      is full, messages are dropped rather than blocking the publisher.
    - ``zmq.Poller`` for non-blocking receives with configurable timeout.
    - ``zmq.Context.instance()`` singleton to share ZMQ I/O threads across
      all publishers/subscribers in the same process.

Example:
    >>> with EventPublisher("tcp://127.0.0.1:5557") as pub:
    ...     pub.publish("detection", {"timestamp": 1723.456, "label": "person"})
    ...
    >>> with EventSubscriber("tcp://127.0.0.1:5557", topic="detection") as sub:
    ...     event = sub.receive(timeout_ms=1000)
    ...     if event:
    ...         topic, payload = event
"""

from __future__ import annotations

import json
import logging
import time

import zmq

logger = logging.getLogger(__name__)


class EventPublisher:
    """ZeroMQ PUB socket wrapper with topic-based multipart messaging.

    Binds a PUB socket on the given endpoint. Messages are sent as
    two-frame multipart messages: ``[topic, json_payload]``.

    Args:
        endpoint: ZMQ bind address (e.g., ``"tcp://127.0.0.1:5557"``).
        high_water_mark: Maximum queued outbound messages before dropping.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        high_water_mark: int = 1000,
    ) -> None:
        self._endpoint = endpoint
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)

        # Configure socket options
        self._socket.setsockopt(zmq.SNDHWM, high_water_mark)
        self._socket.setsockopt(zmq.LINGER, 0)  # Don't block on close

        self._socket.bind(endpoint)
        self._closed = False
        self._messages_sent = 0

        logger.info(f"ZMQ Publisher bound to {endpoint}")

        # Allow subscribers to connect before we start sending
        # (ZMQ PUB/SUB has a brief connection setup period)
        time.sleep(0.1)

    def publish(self, topic: str, payload: dict) -> bool:
        """Publish a JSON event on a named topic.

        Non-blocking: if the high-water mark is reached, the message
        is silently dropped rather than blocking the caller.

        Args:
            topic: Topic string for subscriber filtering (e.g., ``"detection"``).
            payload: Dictionary to serialize as JSON. Must be JSON-serializable.

        Returns:
            True if the message was queued for sending, False if dropped.
        """
        if self._closed:
            logger.warning("Attempted to publish on a closed socket.")
            return False

        try:
            topic_bytes = topic.encode("utf-8")
            payload_bytes = json.dumps(payload, default=str).encode("utf-8")

            self._socket.send_multipart(
                [topic_bytes, payload_bytes],
                flags=zmq.NOBLOCK,
            )
            self._messages_sent += 1
            return True

        except zmq.Again:
            logger.debug(f"ZMQ send dropped (HWM reached) on topic '{topic}'")
            return False
        except zmq.ZMQError as exc:
            logger.error(f"ZMQ publish error on topic '{topic}': {exc}")
            return False

    @property
    def messages_sent(self) -> int:
        """Total number of messages successfully queued since creation."""
        return self._messages_sent

    def close(self) -> None:
        """Close the PUB socket and release resources."""
        if self._closed:
            return

        self._closed = True
        try:
            self._socket.close()
        except zmq.ZMQError as exc:
            logger.warning(f"Error closing ZMQ PUB socket: {exc}")

        logger.info(
            f"ZMQ Publisher closed ({self._endpoint}). "
            f"Total messages sent: {self._messages_sent}"
        )

    def __enter__(self) -> EventPublisher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return (
            f"EventPublisher(endpoint='{self._endpoint}', "
            f"status={status}, sent={self._messages_sent})"
        )


class EventSubscriber:
    """ZeroMQ SUB socket wrapper with non-blocking receive via Poller.

    Connects a SUB socket to the publisher's endpoint and subscribes
    to a specific topic (or all topics if ``topic=""``).

    Args:
        endpoint: ZMQ connect address (e.g., ``"tcp://127.0.0.1:5557"``).
        topic: Topic filter string. Empty string subscribes to all topics.
        high_water_mark: Maximum queued inbound messages before dropping.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        topic: str = "",
        high_water_mark: int = 1000,
    ) -> None:
        self._endpoint = endpoint
        self._topic = topic
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)

        # Configure socket options
        self._socket.setsockopt(zmq.RCVHWM, high_water_mark)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.LINGER, 0)

        self._socket.connect(endpoint)

        # Poller for non-blocking receive with timeout
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        self._closed = False
        self._messages_received = 0

        logger.info(
            f"ZMQ Subscriber connected to {endpoint}, "
            f"topic filter: '{topic or '(all)'}'"
        )

    def receive(self, timeout_ms: int = 1000) -> tuple[str, dict] | None:
        """Receive a single message with timeout.

        Uses ``zmq.Poller`` to wait for a message up to ``timeout_ms``
        milliseconds. Returns immediately if a message is available.

        Args:
            timeout_ms: Maximum wait time in milliseconds. Use 0 for
                        non-blocking (return immediately if no message).
                        Use -1 for infinite blocking.

        Returns:
            Tuple of ``(topic, payload_dict)`` if a message was received,
            or ``None`` if the timeout elapsed with no message.
        """
        if self._closed:
            logger.warning("Attempted to receive on a closed socket.")
            return None

        try:
            events = dict(self._poller.poll(timeout_ms))

            if self._socket in events:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)

                if len(parts) != 2:
                    logger.warning(
                        f"Received malformed multipart message with {len(parts)} parts, "
                        f"expected 2. Discarding."
                    )
                    return None

                topic_str = parts[0].decode("utf-8")
                payload = json.loads(parts[1].decode("utf-8"))

                self._messages_received += 1
                return topic_str, payload

        except zmq.Again:
            pass  # No message available (non-blocking mode)
        except json.JSONDecodeError as exc:
            logger.warning(f"Received non-JSON payload: {exc}")
        except zmq.ZMQError as exc:
            logger.error(f"ZMQ receive error: {exc}")

        return None

    def drain(self, max_messages: int = 100) -> list[tuple[str, dict]]:
        """Drain all pending messages from the socket (non-blocking).

        Useful for clearing a backlog or processing a burst of events.

        Args:
            max_messages: Maximum messages to drain in one call.

        Returns:
            List of ``(topic, payload)`` tuples.
        """
        messages: list[tuple[str, dict]] = []
        for _ in range(max_messages):
            msg = self.receive(timeout_ms=0)
            if msg is None:
                break
            messages.append(msg)
        return messages

    @property
    def messages_received(self) -> int:
        """Total number of messages successfully received since creation."""
        return self._messages_received

    def close(self) -> None:
        """Close the SUB socket and release resources."""
        if self._closed:
            return

        self._closed = True
        try:
            self._poller.unregister(self._socket)
        except (zmq.ZMQError, KeyError):
            pass
        try:
            self._socket.close()
        except zmq.ZMQError as exc:
            logger.warning(f"Error closing ZMQ SUB socket: {exc}")

        logger.info(
            f"ZMQ Subscriber closed ({self._endpoint}). "
            f"Total messages received: {self._messages_received}"
        )

    def __enter__(self) -> EventSubscriber:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return (
            f"EventSubscriber(endpoint='{self._endpoint}', "
            f"topic='{self._topic}', status={status}, "
            f"received={self._messages_received})"
        )
