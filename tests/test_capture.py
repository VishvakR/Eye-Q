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
from app.core.capture import VideoCapture, is_file_source

FRAME_W, FRAME_H = 320, 240
SAMPLE_FPS = 30.0
SAMPLE_FRAMES = 60  # 2 seconds at SAMPLE_FPS


@pytest.fixture
def sample_video(tmp_path):
    """Write a short MP4 whose frames have distinguishable content."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_FPS, (FRAME_W, FRAME_H)
    )
    if not writer.isOpened():
        pytest.skip("No MP4 encoder available in this OpenCV build")

    for i in range(SAMPLE_FRAMES):
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

    def test_default_never_gives_up_on_a_source(self, buffer):
        """A camera rebooting for longer than the backoff must still come back."""
        capture = VideoCapture(
            source="/nonexistent/path/video.mp4",
            buffer=buffer,
            reconnect_base_delay=0.01,
        )
        capture.start()
        try:
            time.sleep(0.5)  # several capped-backoff attempts
            assert capture.is_running, "Capture gave up on a source with no retry limit"
            assert capture.stats.reconnect_attempts >= 3
        finally:
            capture.stop(timeout=3.0)

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


class TestWritePacing:
    """A live source at the target rate must be written at the target rate."""

    class _FakeCap:
        """Delivers frames on a scripted clock: each grab advances time."""

        def __init__(self, capture, arrivals: list[float]) -> None:
            self._capture = capture
            self._arrivals = iter(arrivals)
            self.now = 0.0

        def grab(self) -> bool:
            try:
                self.now = next(self._arrivals)
            except StopIteration:
                self._capture._shutdown_event.set()
                return False
            return True

        def retrieve(self):
            return True, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    def _run(self, capture, monkeypatch, arrivals: list[float]) -> None:
        cap = self._FakeCap(capture, arrivals)
        monkeypatch.setattr("app.core.capture.time.monotonic", lambda: cap.now)
        capture._cap = cap
        capture._run_capture()

    def test_jittered_frames_at_target_fps_are_all_written(self, buffer, monkeypatch):
        """30 FPS frames with ±2 ms of jitter under target_fps=30 lose nothing.

        The old test compared elapsed time against the last write with no
        slack, so every frame arriving a hair early was dropped.
        """
        capture = VideoCapture(source=0, buffer=buffer, target_fps=30)
        interval = 1.0 / 30.0
        jitter = [0.002, -0.002, 0.0015, -0.0015]
        self._run(
            capture,
            monkeypatch,
            [100.0 + i * interval + jitter[i % len(jitter)] for i in range(120)],
        )

        assert capture.stats.frames_dropped == 0
        assert capture.stats.frames_captured == 120

    def test_over_rate_source_is_held_to_target(self, buffer, monkeypatch):
        capture = VideoCapture(source=0, buffer=buffer, target_fps=10)
        # 10 s at 60 FPS.
        self._run(capture, monkeypatch, [100.0 + i / 60.0 for i in range(600)])

        assert 98 <= capture.stats.frames_captured <= 102

    def test_repeated_frame_errors_hand_over_to_reconnect(self, buffer):
        capture = VideoCapture(source=0, buffer=buffer)

        class _BrokenCap:
            def grab(self):
                raise RuntimeError("decoder wedged")

        capture._cap = _BrokenCap()
        capture._run_capture()  # must return rather than spin forever

        assert capture.stats.frames_dropped == VideoCapture._MAX_CONSECUTIVE_ERRORS


class TestFileSourceClassification:
    """Only real files on disk may be paced and rewound."""

    def test_video_file_is_a_file_source(self, sample_video):
        assert is_file_source(sample_video)

    def test_device_index_is_not_a_file_source(self):
        assert not is_file_source(0)

    def test_rtsp_uri_is_not_a_file_source(self):
        assert not is_file_source("rtsp://192.168.1.100:554/stream1")

    def test_http_uri_is_not_a_file_source(self):
        assert not is_file_source("http://example.com/stream.m3u8")

    def test_missing_path_is_not_a_file_source(self):
        assert not is_file_source("/nonexistent/path/video.mp4")


class TestFilePlayback:
    """A recorded clip must behave like a live camera, not a decode benchmark.

    Without pacing, ``cv2`` decodes a short clip as fast as the CPU allows —
    a 2-second sample replays dozens of times per second, so every frame the
    watchdog reads is seconds away from the one before it.
    """

    def test_playback_runs_at_the_recorded_frame_rate(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=30)
        capture.start()
        try:
            # Let the source open and the first frames land before timing.
            assert _wait_until(lambda: capture.stats.frames_captured > 0)
            baseline = capture.stats.frames_captured
            time.sleep(1.0)
            captured = capture.stats.frames_captured - baseline
        finally:
            capture.stop()

        # ~30 frames in ~1s. The window is wide enough for a loaded machine
        # but nowhere near the hundreds an unpaced decode loop produces.
        assert 12 <= captured <= 55, (
            f"Expected roughly {SAMPLE_FPS:.0f} FPS playback, got {captured} frames in 1s"
        )

    def test_source_fps_is_reported(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer)
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.frames_captured > 0)
            assert capture.stats.source_fps == pytest.approx(SAMPLE_FPS, abs=1.0)
        finally:
            capture.stop()

    def test_target_fps_skips_frames_without_slowing_playback(self, sample_video, buffer):
        """A low target FPS must drop frames, not stretch the clip's duration.

        Playing every frame at 2 FPS would run the 2-second sample in slow
        motion; the clip has to stay in real time and lose the frames between
        writes instead.
        """
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=2)
        capture.start()
        try:
            # 2s of clip at real-time speed is one full pass, so a loop must
            # have happened even though only ~2 frames/s were written.
            assert _wait_until(lambda: capture.stats.loops_completed >= 1, timeout=6.0), (
                "Playback did not reach EOF in real time — it ran in slow motion"
            )
            assert capture.stats.frames_dropped > 0, "Skipped frames were not counted"
        finally:
            capture.stop()


class TestFileLooping:
    """A finished clip must restart seamlessly, without a reconnect cycle."""

    def test_clip_loops_without_reconnecting(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=30)
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.loops_completed >= 1, timeout=8.0), (
                "Capture never looped the sample clip"
            )
            stats = capture.stats
            assert stats.is_connected, "Looping dropped the connection"
            assert stats.reconnect_attempts == 0, (
                "Looping went through the reconnect path instead of seeking"
            )
        finally:
            capture.stop()

    def test_frames_keep_arriving_across_the_loop_point(self, sample_video, buffer):
        capture = VideoCapture(source=sample_video, buffer=buffer, target_fps=30)
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.loops_completed >= 1, timeout=8.0)
            after_loop = capture.stats.frames_captured
            assert _wait_until(
                lambda: capture.stats.frames_captured > after_loop, timeout=3.0
            ), "Capture stalled after rewinding"
        finally:
            capture.stop()

    def test_looping_can_be_disabled(self, sample_video, buffer):
        """``loop_file=False`` keeps the original end-of-session behaviour."""
        capture = VideoCapture(
            source=sample_video, buffer=buffer, target_fps=30, loop_file=False
        )
        capture.start()
        try:
            assert _wait_until(lambda: capture.stats.frames_captured > 0)
            time.sleep(3.0)  # Longer than the 2s clip.
            assert capture.stats.loops_completed == 0
        finally:
            capture.stop()
