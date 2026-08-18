"""Tests for the video capture thread.

Uses a synthetic MP4 written with ``cv2.VideoWriter`` so the tests need no
webcam, no network, and no fixture binaries.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.core.buffer import RollingFrameBuffer
from app.core.capture import VideoCapture

FRAME_W, FRAME_H = 320, 240


@pytest.fixture
def sample_video(tmp_path):
    """Write a short MP4 whose frames have distinguishable content."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_W, FRAME_H)
    )
    if not writer.isOpened():
        pytest.skip("No MP4 encoder available in this OpenCV build")

    for i in range(60):
        frame = np.full((FRAME_H, FRAME_W, 3), i * 4 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return str(path)


@pytest.fixture
def buffer():
    """A small ring buffer matched to the synthetic video's dimensions."""
    # macOS caps POSIX shared-memory names at 31 characters, so the unique
    # suffix has to stay short.
    buf = RollingFrameBuffer(
        capacity=30,
        frame_height=FRAME_H,
        frame_width=FRAME_W,
        buffer_id=f"cap{int(time.monotonic() * 1000) % 100000}",
    )
    yield buf
    buf.close()
    buf.unlink()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll a predicate until true or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TestVideoCaptureLifecycle:
    """Start/stop semantics that the orchestrator depends on."""

    def test_not_running_before_start(self, buffer):
        capture = VideoCapture(source="nonexistent.mp4", buffer=buffer)
        assert not capture.is_running
        assert not capture.is_connected

    def test_double_start_raises(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer)
        capture.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                capture.start()
        finally:
            capture.stop()

    def test_stop_is_idempotent(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer)
        capture.start()
        capture.stop()
        capture.stop()  # Must not raise or hang
        assert not capture.is_running

    def test_repr_reports_state(self, buffer):
        capture = VideoCapture(source="test.mp4", buffer=buffer, camera_id="cam_1")
        text = repr(capture)
        assert "cam_1" in text
        assert "stopped" in text


class TestVideoCaptureIngestion:
    """Frames must actually reach the ring buffer."""

    def test_frames_land_in_the_buffer(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=30)
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.frames_captured > 0), (
                "Capture thread never wrote a frame"
            )

            frame, timestamp, _index = buffer.read_latest()
            assert frame.shape == (FRAME_H, FRAME_W, 3)
            assert timestamp > 0
        finally:
            capture.stop()

    def test_stats_are_populated(self, sample_video, buffer):
        capture = VideoCapture(
            source=sample_video, buffer=buffer, camera_id="cam_x", target_fps=30
        )
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.frames_captured >= 5)

            stats = capture.stats
            assert stats.camera_id == "cam_x"
            assert stats.is_running
            assert stats.uptime_seconds > 0
            assert stats.target_fps == 30
        finally:
            capture.stop()

    def test_fps_cap_limits_writes(self, sample_video, buffer):
        """A low target FPS must throttle buffer writes, not the decode loop."""
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=2)
        capture.start()
        try:
            time.sleep(1.0)
            captured = capture.stats.frames_captured
        finally:
            capture.stop()

        # At 2 FPS, ~1s of wall clock should yield only a couple of frames —
        # generously bounded to stay reliable on a loaded CI machine.
        assert captured <= 8, f"Expected throttling to ~2 FPS, got {captured} frames"


class TestVideoCaptureResilience:
    """A dead source must fail loudly and finitely, never hang the process."""

    def test_bad_source_gives_up_after_retries(self, buffer):
        capture = VideoCapture(
            source="/nonexistent/path/video.mp4",
            buffer=buffer,
            reconnect_max_retries=2,
            reconnect_base_delay=0.05,
        )
        capture.start()

        assert _wait_until(lambda: not capture.is_running, timeout=5.0), (
            "Capture thread did not exit after exhausting retries"
        )
        assert capture.stats.reconnect_attempts >= 1
        assert not capture.is_connected

    def test_stop_interrupts_reconnect_backoff(self, buffer):
        """Shutdown must not wait out a long backoff sleep."""
        capture = VideoCapture(
            source="/nonexistent/path/video.mp4",
            buffer=buffer,
            reconnect_max_retries=10,
            reconnect_base_delay=5.0,
        )
        capture.start()
        time.sleep(0.2)

        started = time.monotonic()
        capture.stop(timeout=3.0)
        elapsed = time.monotonic() - started

        assert elapsed < 3.0, f"stop() blocked on the backoff sleep for {elapsed:.1f}s"
        assert not capture.is_running
