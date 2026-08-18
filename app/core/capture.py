"""Non-blocking video capture running in a dedicated daemon thread.

This module wraps OpenCV's ``cv2.VideoCapture`` in a daemon thread that
continuously grabs frames from a video source (RTSP, file, or webcam)
and writes them into the shared-memory ring buffer. It is designed to
never block — if the buffer is full, the oldest frame is overwritten.

Features:
    - Supports RTSP streams, MP4/AVI files, and USB/webcam device indices
    - Exponential backoff reconnection for RTSP streams (1s → 2s → 4s → 8s → 16s)
    - Configurable target FPS with frame-skip logic
    - Runtime statistics: captured frames, dropped frames, current FPS
    - Graceful shutdown via ``threading.Event``

Example:
    >>> buffer = RollingFrameBuffer(capacity=300, frame_height=720, frame_width=1280)
    >>> capture = VideoCapture(source="rtsp://192.168.1.100:554/stream1",
    ...                        buffer=buffer, camera_id="cam_1")
    >>> capture.start()
    >>> # ... later ...
    >>> capture.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2

from app.core.buffer import RollingFrameBuffer

logger = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    """Snapshot of video capture statistics for monitoring."""

    camera_id: str
    source: str
    is_running: bool
    is_connected: bool
    frames_captured: int
    frames_dropped: int
    current_fps: float
    target_fps: int
    uptime_seconds: float
    reconnect_attempts: int


class VideoCapture:
    """Non-blocking video capture in a dedicated daemon thread.

    Continuously grabs frames from the source and writes them into
    the shared-memory ring buffer. Designed to never block the main
    thread or any downstream consumer.

    Args:
        source: Video source — RTSP URI (str), file path (str), or device index (int).
        buffer: Shared-memory ring buffer to write frames into.
        camera_id: Unique identifier for this camera stream.
        target_fps: Maximum frame rate to capture. Frames exceeding this
                    rate are grabbed (to keep the stream current) but not decoded.
        reconnect_max_retries: Maximum consecutive reconnection attempts.
        reconnect_base_delay: Initial delay between reconnections (seconds).
                              Doubles with each attempt (exponential backoff).
    """

    # Minimum wall-clock duration a connection must last before a drop is
    # treated as a genuine mid-stream failure rather than an immediate EOF.
    _MIN_SESSION_S = 0.5

    def __init__(
        self,
        source: str | int,
        buffer: RollingFrameBuffer,
        camera_id: str = "default",
        target_fps: int = 30,
        reconnect_max_retries: int = 5,
        reconnect_base_delay: float = 1.0,
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._camera_id = camera_id
        self._target_fps = target_fps
        self._reconnect_max_retries = reconnect_max_retries
        self._reconnect_base_delay = reconnect_base_delay

        # Frame timing
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics (accessed from main thread, written from capture thread)
        self._frames_captured = 0
        self._frames_dropped = 0
        self._reconnect_attempts = 0
        self._is_connected = False
        self._start_time: float = 0.0
        self._last_write_time: float = 0.0
        self._fps_counter = _FPSCounter(window_size=30)

        # OpenCV capture handle (created in capture thread)
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        """Whether the video source is currently connected."""
        return self._is_connected

    def start(self) -> None:
        """Start the capture thread.

        The thread runs as a daemon — it will be killed automatically
        when the main process exits.

        Raises:
            RuntimeError: If the capture is already running.
        """
        if self.is_running:
            raise RuntimeError(
                f"VideoCapture '{self._camera_id}' is already running."
            )

        self._shutdown_event.clear()
        self._start_time = time.monotonic()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"VideoCapture-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            f"VideoCapture '{self._camera_id}' started: "
            f"source={self._source}, target_fps={self._target_fps}"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the capture thread to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        if not self.is_running:
            return

        logger.info(f"Stopping VideoCapture '{self._camera_id}'...")
        self._shutdown_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"VideoCapture '{self._camera_id}' did not stop within "
                    f"{timeout}s — thread may be stuck."
                )

        self._thread = None
        logger.info(f"VideoCapture '{self._camera_id}' stopped.")

    @property
    def stats(self) -> CaptureStats:
        """Current capture statistics snapshot."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        return CaptureStats(
            camera_id=self._camera_id,
            source=str(self._source),
            is_running=self.is_running,
            is_connected=self._is_connected,
            frames_captured=self._frames_captured,
            frames_dropped=self._frames_dropped,
            current_fps=self._fps_counter.fps,
            target_fps=self._target_fps,
            uptime_seconds=uptime,
            reconnect_attempts=self._reconnect_attempts,
        )

    # ── Internal Capture Loop ────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture loop running in a dedicated thread.

        Handles connection, frame grabbing, and reconnection logic.
        """
        while not self._shutdown_event.is_set():
            # Attempt to connect/reconnect
            if not self._connect():
                logger.error(
                    f"VideoCapture '{self._camera_id}': all reconnection "
                    f"attempts exhausted for source '{self._source}'. Stopping."
                )
                break

            session_start = time.monotonic()

            # Continuous frame capture
            self._run_capture()

            # If we exit _run_capture and haven't been told to shut down,
            # we lost the connection — loop back to reconnect
            if self._shutdown_event.is_set():
                break

            session_duration = time.monotonic() - session_start

            if session_duration < self._MIN_SESSION_S:
                # A session this short means the source ended almost
                # immediately — a finished video file being replayed, or a
                # stream that accepts the connection then drops it. Without a
                # pause here the loop reopens the source hundreds of times a
                # second and pins a CPU core.
                logger.info(
                    f"VideoCapture '{self._camera_id}': source ended after "
                    f"{session_duration:.2f}s, restarting in "
                    f"{self._MIN_SESSION_S:.1f}s..."
                )
                self._shutdown_event.wait(timeout=self._MIN_SESSION_S)
            else:
                logger.warning(
                    f"VideoCapture '{self._camera_id}': "
                    f"connection lost, attempting reconnect..."
                )

        # Cleanup
        self._release_capture()

    def _connect(self) -> bool:
        """Attempt to open the video source with exponential backoff.

        Returns:
            True if successfully connected, False if all retries exhausted.
        """
        delay = self._reconnect_base_delay

        for attempt in range(self._reconnect_max_retries):
            if self._shutdown_event.is_set():
                return False

            self._release_capture()

            logger.info(
                f"VideoCapture '{self._camera_id}': connecting to "
                f"'{self._source}' (attempt {attempt + 1}/{self._reconnect_max_retries})"
            )

            try:
                self._cap = cv2.VideoCapture(self._source)

                # Set buffer size to minimize latency for RTSP streams
                if isinstance(self._source, str) and self._source.startswith("rtsp"):
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if self._cap.isOpened():
                    self._is_connected = True
                    self._reconnect_attempts = 0

                    # Log source properties
                    w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self._cap.get(cv2.CAP_PROP_FPS)
                    logger.info(
                        f"VideoCapture '{self._camera_id}': connected. "
                        f"Source resolution: {w}×{h} @ {fps:.1f} FPS"
                    )
                    return True

            except Exception as exc:
                logger.warning(
                    f"VideoCapture '{self._camera_id}': connection error: {exc}"
                )

            self._reconnect_attempts += 1
            self._is_connected = False

            # Exponential backoff (interruptible)
            logger.info(
                f"VideoCapture '{self._camera_id}': retrying in {delay:.1f}s..."
            )
            self._shutdown_event.wait(timeout=delay)
            delay = min(delay * 2, 30.0)  # Cap at 30 seconds

        return False

    def _run_capture(self) -> None:
        """Continuously grab and decode frames from the connected source.

        Implements frame-rate control: if the source produces frames
        faster than ``target_fps``, excess frames are grabbed (to keep
        the RTSP buffer current) but not decoded or stored.
        """
        if self._cap is None:
            return

        while not self._shutdown_event.is_set():
            try:
                # Always grab the latest frame (keeps RTSP stream current)
                grabbed = self._cap.grab()
                if not grabbed:
                    # Stream ended or disconnected
                    self._is_connected = False
                    return

                now = time.monotonic()
                elapsed = now - self._last_write_time

                # Frame-rate control: skip decode if too fast
                if elapsed < self._frame_interval:
                    self._frames_dropped += 1
                    continue

                # Decode the grabbed frame
                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    self._frames_dropped += 1
                    continue

                # Validate frame
                if frame.size == 0 or frame.ndim != 3:
                    self._frames_dropped += 1
                    continue

                # Write to ring buffer
                timestamp = now
                self._buffer.write(frame, timestamp)

                self._frames_captured += 1
                self._fps_counter.tick()
                self._last_write_time = now

            except Exception as exc:
                logger.error(
                    f"VideoCapture '{self._camera_id}': frame error: {exc}"
                )
                self._frames_dropped += 1

                # Small delay to prevent tight error loops
                self._shutdown_event.wait(timeout=0.01)

    def _release_capture(self) -> None:
        """Release the OpenCV capture handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._is_connected = False

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        connected = "connected" if self._is_connected else "disconnected"
        return (
            f"VideoCapture(camera_id='{self._camera_id}', "
            f"source='{self._source}', {status}, {connected}, "
            f"captured={self._frames_captured})"
        )


class _FPSCounter:
    """Sliding-window FPS calculator.

    Tracks the timestamps of the last ``window_size`` frames and
    computes the average frame rate over that window.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record the current time as a frame timestamp."""
        now = time.monotonic()
        self._timestamps.append(now)

        # Keep only the last N timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]

    @property
    def fps(self) -> float:
        """Current frames per second over the sliding window."""
        if len(self._timestamps) < 2:
            return 0.0

        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / duration
