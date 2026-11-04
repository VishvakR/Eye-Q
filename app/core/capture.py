"""Non-blocking video capture running in a dedicated daemon thread.

This module wraps OpenCV's ``cv2.VideoCapture`` in a daemon thread that
continuously grabs frames from a video source (RTSP, file, or webcam)
and writes them into the shared-memory ring buffer. It is designed to
never block — if the buffer is full, the oldest frame is overwritten.

Features:
    - Supports RTSP streams, MP4/AVI files, and USB/webcam device indices
    - Exponential backoff reconnection for RTSP streams (1s → 2s → 4s → 8s → 16s)
    - Configurable target FPS with frame-skip logic
    - File sources play at their native frame rate and loop seamlessly, so a
      recorded clip behaves like a live camera instead of being decoded as
      fast as the CPU allows
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
from pathlib import Path

import cv2

from app.core.buffer import RollingFrameBuffer

logger = logging.getLogger(__name__)

# Schemes that denote a live network stream rather than a file on disk.
_STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_file_source(source: str | int) -> bool:
    """Whether ``source`` names a video file on disk.

    Device indices and network stream URIs are live sources: they arrive in
    real time on their own and must never be paced or rewound. Only a path
    that exists on disk gets the recorded-clip treatment.
    """
    if not isinstance(source, str):
        return False
    if source.lower().startswith(_STREAM_SCHEMES):
        return False
    try:
        return Path(source).is_file()
    except OSError:
        return False


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
    source_fps: float = 0.0
    loops_completed: int = 0


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
        reconnect_max_retries: Maximum consecutive reconnection attempts, or
                               None to keep trying for as long as the
                               appliance runs. An appliance that gives up on a
                               camera rebooting for 40 seconds stays blind
                               until someone restarts the process, so the
                               default is None.
        reconnect_base_delay: Initial delay between reconnections (seconds).
                              Doubles with each attempt (exponential backoff).
        loop_file: Rewind file sources to frame 0 on EOF instead of ending the
                   session. Ignored for live sources.
        realtime_file: Pace file sources at their recorded frame rate. Without
                       it a clip is decoded as fast as the CPU allows, which
                       turns a 17-second recording into a sub-second blur and
                       hands Stage 1 frames that jump seconds apart.
    """

    # Minimum wall-clock duration a connection must last before a drop is
    # treated as a genuine mid-stream failure rather than an immediate EOF.
    _MIN_SESSION_S = 0.5

    # Per-frame exceptions in a row before the session is abandoned and the
    # source reopened. A handle that fails every frame will not recover by
    # being asked again a hundred times a second.
    _MAX_CONSECUTIVE_ERRORS = 50

    def __init__(
        self,
        source: str | int,
        buffer: RollingFrameBuffer,
        camera_id: str = "default",
        target_fps: int = 30,
        reconnect_max_retries: int | None = None,
        reconnect_base_delay: float = 1.0,
        loop_file: bool = True,
        realtime_file: bool = True,
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._camera_id = camera_id
        self._target_fps = target_fps
        self._reconnect_max_retries = reconnect_max_retries
        self._reconnect_base_delay = reconnect_base_delay
        self._loop_file = loop_file
        self._realtime_file = realtime_file
        self._is_file = is_file_source(source)

        # Frame timing
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        # Playback state for file sources
        self._source_fps: float = 0.0
        self._loops_completed = 0
        self._frames_since_rewind = 0

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics (accessed from main thread, written from capture thread)
        self._frames_captured = 0
        self._frames_dropped = 0
        self._reconnect_attempts = 0
        self._is_connected = False
        self._start_time: float = 0.0
        self._next_write_at: float = 0.0
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
            source_fps=self._source_fps,
            loops_completed=self._loops_completed,
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

        attempt = 0
        while (
            self._reconnect_max_retries is None
            or attempt < self._reconnect_max_retries
        ):
            attempt += 1
            if self._shutdown_event.is_set():
                return False

            self._release_capture()

            logger.info(
                f"VideoCapture '{self._camera_id}': connecting to "
                f"'{self._source}' (attempt {attempt}/{self._reconnect_max_retries or '∞'})"
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
                    # A file's declared FPS drives playback pacing. Some
                    # containers report 0 or NaN; those fall back to target_fps.
                    self._source_fps = fps if fps and fps > 0 and fps < 1000 else 0.0
                    self._frames_since_rewind = 0
                    logger.info(
                        f"VideoCapture '{self._camera_id}': connected. "
                        f"Source resolution: {w}×{h} @ {fps:.1f} FPS"
                        + (
                            f" (file playback @ {1.0 / self._playback_interval():.1f} FPS"
                            f"{', looping' if self._loop_file else ''})"
                            if self._playback_interval() > 0.0
                            else ""
                        )
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

        Two rates are in play and they do different jobs:

        - **Grab rate.** Live sources are grabbed as fast as they arrive, which
          keeps the RTSP buffer current. File sources are grabbed on a clock
          derived from their recorded FPS, so a 17-second clip takes 17
          seconds rather than being decoded at whatever rate the CPU manages.
        - **Write rate.** Independently of the above, frames arriving faster
          than ``target_fps`` are grabbed but not decoded or stored, so a
          30 FPS file still writes at 2 FPS when asked to — in real time, by
          skipping frames rather than by playing in slow motion.

        On EOF a looping file seeks back to frame 0 and keeps the same
        session, so the buffer never goes dark between passes.
        """
        if self._cap is None:
            return

        grab_interval = self._playback_interval()
        next_grab = time.monotonic()
        consecutive_errors = 0
        # Half a frame of slack on the write deadline. Frames from a source at
        # exactly target_fps arrive with jitter either side of the interval,
        # and a strict `elapsed < interval` test against the previous write
        # rejected every early one — a 29.97 FPS clip under target_fps=30
        # left 0.03 ms of margin and wrote in bursts at 15-20 FPS.
        write_slack = self._frame_interval / 2.0

        while not self._shutdown_event.is_set():
            try:
                if grab_interval > 0.0:
                    # Pace file playback. wait() rather than sleep() so a
                    # shutdown lands immediately instead of after the frame.
                    delay = next_grab - time.monotonic()
                    if delay > 0 and self._shutdown_event.wait(timeout=delay):
                        return
                    next_grab += grab_interval
                    # Never bank a catch-up burst: if decode fell behind (or
                    # the process was suspended), resume from now instead of
                    # replaying the backlog at full speed.
                    next_grab = max(next_grab, time.monotonic())

                # Always grab the latest frame (keeps RTSP stream current)
                grabbed = self._cap.grab()
                if not grabbed:
                    if self._rewind():
                        continue
                    # Stream ended or disconnected
                    self._is_connected = False
                    return

                self._frames_since_rewind += 1
                now = time.monotonic()

                # Frame-rate control: skip decode if too fast. Paced against a
                # deadline that advances by exactly one interval per write, so
                # the long-run rate is target_fps rather than drifting under it.
                if now < self._next_write_at - write_slack:
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
                consecutive_errors = 0
                # Never bank a burst: after a stall, resume from now.
                self._next_write_at = max(
                    self._next_write_at + self._frame_interval,
                    now + write_slack,
                )

            except Exception as exc:
                self._frames_dropped += 1
                consecutive_errors += 1
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"VideoCapture '{self._camera_id}': {consecutive_errors} "
                        f"consecutive frame errors (last: {exc}); reopening the source."
                    )
                    self._is_connected = False
                    return
                if consecutive_errors == 1:
                    logger.error(
                        f"VideoCapture '{self._camera_id}': frame error: {exc}"
                    )

                # Small delay to prevent tight error loops
                self._shutdown_event.wait(timeout=0.01)

    def _playback_interval(self) -> float:
        """Seconds between grabs for a paced file source, or 0.0 for live ones.

        Returns 0.0 whenever the source is live or pacing is disabled, which
        is the caller's signal to grab as fast as frames arrive.
        """
        if not (self._is_file and self._realtime_file):
            return 0.0

        fps = self._source_fps if self._source_fps > 0 else float(self._target_fps)
        return 1.0 / fps if fps > 0 else 0.0

    def _rewind(self) -> bool:
        """Seek a looping file source back to its first frame.

        Returns:
            True if the source was rewound and capture should continue.
            False for live sources, for non-looping files, and when the
            previous rewind yielded no frames — a container whose seek
            silently fails would otherwise spin here forever, so it is handed
            to the reconnect path, which reopens the file from scratch.
        """
        if self._cap is None or not (self._is_file and self._loop_file):
            return False

        if self._frames_since_rewind == 0:
            logger.warning(
                f"VideoCapture '{self._camera_id}': rewind produced no frames "
                f"for '{self._source}' — reopening the source instead."
            )
            return False

        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return False

        self._frames_since_rewind = 0
        self._loops_completed += 1
        logger.debug(
            f"VideoCapture '{self._camera_id}': looped "
            f"'{self._source}' ({self._loops_completed} pass(es))."
        )
        return True

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
