"""Unit and integration tests for the Rolling Frame Buffer.

Tests cover:
    - Basic write/read cycle with synthetic frames
    - Read-latest returns the most recently written frame
    - Circular overwrite (buffer overflow) semantics
    - Timestamp-based lookup within and outside tolerance
    - Index-based direct slot access
    - Buffer statistics tracking
    - Error handling (empty buffer, closed buffer, invalid frames)
    - Multi-frame write/read consistency
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.core.buffer import RollingFrameBuffer

# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def small_buffer():
    """Create a small test buffer (5 frames, 64×64×3) and clean up after."""
    buf = RollingFrameBuffer(
        capacity=5,
        frame_height=64,
        frame_width=64,
        frame_channels=3,
        buffer_id=f"test_{int(time.monotonic() * 1000)}",
    )
    yield buf
    buf.close()
    buf.unlink()


@pytest.fixture
def medium_buffer():
    """Create a medium test buffer (20 frames, 128×128×3)."""
    buf = RollingFrameBuffer(
        capacity=20,
        frame_height=128,
        frame_width=128,
        frame_channels=3,
        buffer_id=f"test_med_{int(time.monotonic() * 1000)}",
    )
    yield buf
    buf.close()
    buf.unlink()


def _make_frame(
    height: int = 64, width: int = 64, channels: int = 3, fill: int = 0
) -> np.ndarray:
    """Create a synthetic frame filled with a specific value."""
    return np.full((height, width, channels), fill_value=fill, dtype=np.uint8)


# ── Basic Read/Write ─────────────────────────────────────────────

class TestBasicReadWrite:
    """Test fundamental write and read operations."""

    def test_write_single_frame(self, small_buffer: RollingFrameBuffer):
        """Writing a single frame should succeed and return slot index 0."""
        frame = _make_frame(64, 64, 3, fill=42)
        idx = small_buffer.write(frame, timestamp=100.0)
        assert idx == 0

    def test_read_latest_after_single_write(self, small_buffer: RollingFrameBuffer):
        """read_latest should return the frame we just wrote."""
        frame = _make_frame(64, 64, 3, fill=99)
        small_buffer.write(frame, timestamp=100.0)

        result_frame, result_ts, result_idx = small_buffer.read_latest()

        assert result_idx == 0
        assert result_ts == 100.0
        np.testing.assert_array_equal(result_frame, frame)

    def test_read_latest_returns_copy(self, small_buffer: RollingFrameBuffer):
        """read_latest should return a copy, not a view into shared memory."""
        frame = _make_frame(64, 64, 3, fill=50)
        small_buffer.write(frame, timestamp=100.0)

        result, _, _ = small_buffer.read_latest()

        # Mutating the result should not affect the buffer
        result[:] = 0
        result2, _, _ = small_buffer.read_latest()
        assert result2[0, 0, 0] == 50  # Original value preserved

    def test_write_multiple_read_latest(self, small_buffer: RollingFrameBuffer):
        """read_latest should always return the most recently written frame."""
        for i in range(3):
            frame = _make_frame(64, 64, 3, fill=i * 10)
            small_buffer.write(frame, timestamp=100.0 + i)

        result, ts, idx = small_buffer.read_latest()

        assert idx == 2  # Third frame at slot 2
        assert ts == 102.0
        assert result[0, 0, 0] == 20  # fill=2*10

    def test_auto_timestamp(self, small_buffer: RollingFrameBuffer):
        """Omitting timestamp should auto-assign time.monotonic()."""
        before = time.monotonic()
        frame = _make_frame(64, 64, 3, fill=1)
        small_buffer.write(frame)
        after = time.monotonic()

        _, ts, _ = small_buffer.read_latest()
        assert before <= ts <= after


# ── Circular Overwrite ───────────────────────────────────────────

class TestCircularOverwrite:
    """Test that the buffer wraps around correctly when full."""

    def test_write_beyond_capacity(self, small_buffer: RollingFrameBuffer):
        """Writing more frames than capacity should wrap the write head."""
        capacity = small_buffer.capacity  # 5

        for i in range(capacity + 3):  # Write 8 frames into 5-slot buffer
            frame = _make_frame(64, 64, 3, fill=i)
            small_buffer.write(frame, timestamp=float(i))

        # The latest frame should be frame 7 (fill=7)
        result, ts, idx = small_buffer.read_latest()
        assert result[0, 0, 0] == 7
        assert ts == 7.0
        # Write head should be at (7+1) % 5 = 3, so latest is slot 2
        assert idx == (8 - 1) % capacity

    def test_overwritten_frames_are_gone(self, small_buffer: RollingFrameBuffer):
        """Frames overwritten by circular wrap should be replaced."""
        capacity = small_buffer.capacity  # 5

        # Fill buffer completely
        for i in range(capacity):
            small_buffer.write(_make_frame(fill=i), timestamp=float(i))

        # Overwrite slot 0 with a new frame
        small_buffer.write(_make_frame(fill=99), timestamp=99.0)

        # Slot 0 should now contain fill=99, not fill=0
        frame, ts = small_buffer.read_at_index(0)
        assert frame[0, 0, 0] == 99
        assert ts == 99.0


# ── Timestamp Lookup ─────────────────────────────────────────────

class TestTimestampLookup:
    """Test timestamp-based frame retrieval."""

    def test_exact_timestamp_match(self, small_buffer: RollingFrameBuffer):
        """read_by_timestamp should find an exact timestamp match."""
        for i in range(3):
            small_buffer.write(_make_frame(fill=i * 50), timestamp=100.0 + i)

        result = small_buffer.read_by_timestamp(101.0, tolerance_ms=50.0)

        assert result is not None
        frame, ts, idx = result
        assert ts == 101.0
        assert frame[0, 0, 0] == 50  # fill=1*50

    def test_near_timestamp_within_tolerance(self, small_buffer: RollingFrameBuffer):
        """read_by_timestamp should find the closest frame within tolerance."""
        small_buffer.write(_make_frame(fill=10), timestamp=100.0)
        small_buffer.write(_make_frame(fill=20), timestamp=100.5)
        small_buffer.write(_make_frame(fill=30), timestamp=101.0)

        # Look for 100.4 — closest is 100.5 (0.1s = 100ms away)
        result = small_buffer.read_by_timestamp(100.4, tolerance_ms=200.0)

        assert result is not None
        frame, ts, _ = result
        assert ts == 100.5
        assert frame[0, 0, 0] == 20

    def test_timestamp_outside_tolerance(self, small_buffer: RollingFrameBuffer):
        """read_by_timestamp should return None if no frame is within tolerance."""
        small_buffer.write(_make_frame(fill=10), timestamp=100.0)

        result = small_buffer.read_by_timestamp(200.0, tolerance_ms=50.0)
        assert result is None

    def test_timestamp_empty_buffer(self, small_buffer: RollingFrameBuffer):
        """read_by_timestamp on empty buffer should return None."""
        result = small_buffer.read_by_timestamp(100.0)
        assert result is None


# ── Index-Based Access ───────────────────────────────────────────

class TestIndexAccess:
    """Test direct slot access by index."""

    def test_read_at_valid_index(self, small_buffer: RollingFrameBuffer):
        """read_at_index should return the correct frame."""
        small_buffer.write(_make_frame(fill=11), timestamp=1.0)
        small_buffer.write(_make_frame(fill=22), timestamp=2.0)

        frame, ts = small_buffer.read_at_index(1)
        assert frame[0, 0, 0] == 22
        assert ts == 2.0

    def test_read_at_invalid_index(self, small_buffer: RollingFrameBuffer):
        """read_at_index with out-of-range index should raise IndexError."""
        small_buffer.write(_make_frame(fill=1), timestamp=1.0)

        with pytest.raises(IndexError):
            small_buffer.read_at_index(10)

        with pytest.raises(IndexError):
            small_buffer.read_at_index(-1)


# ── Statistics ───────────────────────────────────────────────────

class TestBufferStats:
    """Test buffer statistics reporting."""

    def test_initial_stats(self, small_buffer: RollingFrameBuffer):
        """Fresh buffer should report zero frames written."""
        stats = small_buffer.stats
        assert stats.capacity == 5
        assert stats.frames_written == 0
        assert stats.occupancy == 0.0
        assert stats.frame_shape == (64, 64, 3)
        assert stats.buffer_id.startswith("test_")

    def test_stats_after_writes(self, small_buffer: RollingFrameBuffer):
        """Stats should track frames written and occupancy correctly."""
        for i in range(3):
            small_buffer.write(_make_frame(fill=i), timestamp=float(i))

        stats = small_buffer.stats
        assert stats.frames_written == 3
        assert stats.occupancy == 3 / 5  # 60%

    def test_stats_occupancy_caps_at_one(self, small_buffer: RollingFrameBuffer):
        """Occupancy should cap at 1.0 even after circular wrap."""
        for i in range(10):
            small_buffer.write(_make_frame(fill=i), timestamp=float(i))

        stats = small_buffer.stats
        assert stats.frames_written == 10
        assert stats.occupancy == 1.0


# ── Error Handling ───────────────────────────────────────────────

class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_read_empty_buffer_raises(self, small_buffer: RollingFrameBuffer):
        """read_latest on empty buffer should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="empty"):
            small_buffer.read_latest()

    def test_write_wrong_channels_raises(self, small_buffer: RollingFrameBuffer):
        """Writing a frame with wrong channel count should raise ValueError."""
        bad_frame = np.zeros((64, 64, 4), dtype=np.uint8)  # 4 channels, expected 3
        with pytest.raises(ValueError, match="channels"):
            small_buffer.write(bad_frame)

    def test_write_after_close_raises(self, small_buffer: RollingFrameBuffer):
        """Writing to a closed buffer should raise RuntimeError."""
        small_buffer.close()
        with pytest.raises(RuntimeError, match="closed"):
            small_buffer.write(_make_frame(), timestamp=1.0)

    def test_read_after_close_raises(self, small_buffer: RollingFrameBuffer):
        """Reading from a closed buffer should raise RuntimeError."""
        small_buffer.write(_make_frame(), timestamp=1.0)
        small_buffer.close()
        with pytest.raises(RuntimeError, match="closed"):
            small_buffer.read_latest()


# ── Frame Resize ─────────────────────────────────────────────────

class TestFrameResize:
    """Test automatic frame resizing on dimension mismatch."""

    def test_frame_resize_on_write(self, small_buffer: RollingFrameBuffer):
        """Writing a frame with different dimensions should resize it."""
        # Buffer expects 64×64, but we write 128×128
        big_frame = _make_frame(128, 128, 3, fill=77)
        idx = small_buffer.write(big_frame, timestamp=1.0)

        assert idx == 0
        result, _, _ = small_buffer.read_latest()
        assert result.shape == (64, 64, 3)


# ── Repr ─────────────────────────────────────────────────────────

class TestRepr:
    """Test string representation."""

    def test_repr(self, small_buffer: RollingFrameBuffer):
        """__repr__ should produce a readable string."""
        r = repr(small_buffer)
        assert "RollingFrameBuffer" in r
        assert "capacity=5" in r
        assert "written=0" in r
