"""Rolling frame buffer backed by POSIX shared memory.

This module provides a zero-copy circular buffer for video frames,
designed for high-throughput producer/consumer pipelines where the
video capture thread writes frames continuously and downstream
consumers (YOLO watchdog, VLM worker) read them without blocking
the producer.

Architecture:
    - Two shared memory segments: pixel data (uint8) and timestamps (float64).
    - Circular write via an atomic shared integer (write head).
    - Writer-side locking only; readers are lock-free (stale-OK semantics).

A torn read (reading a frame while it's being overwritten) is harmless:
YOLO will produce a low-confidence/garbage detection that gets filtered.
The VLM worker validates frame integrity after fetch.

Example:
    >>> buffer = RollingFrameBuffer(capacity=300, frame_height=720, frame_width=1280)
    >>> buffer.write(frame, timestamp=time.monotonic())
    >>> latest_frame, ts, idx = buffer.read_latest()
    >>> buffer.close()
    >>> buffer.unlink()  # Only in the owning process
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Lock, Value
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferStats:
    """Snapshot of rolling buffer state for monitoring."""

    capacity: int
    frames_written: int
    occupancy: float  # 0.0 to 1.0
    frame_shape: tuple[int, int, int]
    memory_bytes: int
    buffer_id: str


class RollingFrameBuffer:
    """Lock-light circular buffer backed by POSIX shared memory.

    Memory Layout:
        SHM Block 1 ("eyeq_frames_{id}"):
            np.ndarray of shape (capacity, height, width, channels), dtype=uint8

        SHM Block 2 ("eyeq_ts_{id}"):
            np.ndarray of shape (capacity,), dtype=float64

        Shared Value: write_head (int32) — current write position
        Shared Value: frame_count (int64) — total frames written (monotonic)

    Args:
        capacity: Number of frames the ring buffer holds.
        frame_height: Height of each frame in pixels.
        frame_width: Width of each frame in pixels.
        frame_channels: Number of color channels (default 3 for BGR/RGB).
        buffer_id: Unique identifier for multi-camera disambiguation.
    """

    def __init__(
        self,
        capacity: int = 300,
        frame_height: int = 1080,
        frame_width: int = 1920,
        frame_channels: int = 3,
        buffer_id: str = "default",
    ) -> None:
        self._capacity = capacity
        self._frame_shape = (frame_height, frame_width, frame_channels)
        self._buffer_id = buffer_id

        # Calculate sizes
        frame_bytes = int(np.prod(self._frame_shape)) * np.dtype(np.uint8).itemsize
        total_frame_bytes = capacity * frame_bytes
        total_ts_bytes = capacity * np.dtype(np.float64).itemsize

        logger.info(
            f"Allocating frame buffer '{buffer_id}': "
            f"{capacity} frames × {frame_height}×{frame_width}×{frame_channels} = "
            f"{total_frame_bytes / (1024**2):.1f} MB"
        )

        # Allocate shared memory — frame pixel data
        self._shm_frames = SharedMemory(
            name=f"eyeq_frames_{buffer_id}",
            create=True,
            size=total_frame_bytes,
        )

        # Allocate shared memory — timestamps
        self._shm_timestamps = SharedMemory(
            name=f"eyeq_ts_{buffer_id}",
            create=True,
            size=total_ts_bytes,
        )

        # Create numpy views over shared memory (zero-copy)
        self._frames: NDArray[np.uint8] = np.ndarray(
            (capacity, *self._frame_shape),
            dtype=np.uint8,
            buffer=self._shm_frames.buf,
        )
        self._timestamps: NDArray[np.float64] = np.ndarray(
            (capacity,),
            dtype=np.float64,
            buffer=self._shm_timestamps.buf,
        )

        # Initialize timestamps to zero (marks slots as unwritten)
        self._timestamps[:] = 0.0

        # Shared atomic counters
        self._write_head: Value = Value("i", 0)  # int32
        self._frame_count: Value = Value("l", 0)  # int64

        # Writer-side lock (readers are lock-free)
        self._write_lock = Lock()

        self._closed = False

        logger.info(
            f"Frame buffer '{buffer_id}' allocated: "
            f"{total_frame_bytes / (1024**2):.1f} MB frames + "
            f"{total_ts_bytes / 1024:.1f} KB timestamps"
        )

    @property
    def capacity(self) -> int:
        """Maximum number of frames the buffer can hold."""
        return self._capacity

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Shape of a single frame: (height, width, channels)."""
        return self._frame_shape

    @property
    def buffer_id(self) -> str:
        """Unique identifier for this buffer instance."""
        return self._buffer_id

    def write(self, frame: NDArray[np.uint8], timestamp: float | None = None) -> int:
        """Write a frame into the next slot in the ring buffer.

        The frame is copied into shared memory. If the buffer is full,
        the oldest frame is silently overwritten (circular semantics).

        Args:
            frame: Video frame as numpy array. Must match ``frame_shape``.
                   If the frame dimensions differ, it will be resized.
            timestamp: Monotonic timestamp (``time.monotonic()``). If None,
                       the current monotonic time is used.

        Returns:
            The slot index where the frame was written.

        Raises:
            ValueError: If the frame has an incompatible number of channels.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed buffer.")

        if timestamp is None:
            timestamp = time.monotonic()

        # Validate and resize if needed
        expected_h, expected_w, expected_c = self._frame_shape
        if frame.ndim != 3 or frame.shape[2] != expected_c:
            raise ValueError(
                f"Frame must have {expected_c} channels, got shape {frame.shape}"
            )

        if frame.shape[0] != expected_h or frame.shape[1] != expected_w:
            # Resize in-place to expected dimensions
            import cv2

            frame = cv2.resize(frame, (expected_w, expected_h))

        with self._write_lock:
            idx = self._write_head.value

            # Copy frame data into shared memory
            np.copyto(self._frames[idx], frame)
            self._timestamps[idx] = timestamp

            # Advance the write head (circular)
            self._write_head.value = (idx + 1) % self._capacity
            self._frame_count.value += 1

        return idx

    def read_latest(self) -> tuple[NDArray[np.uint8], float, int]:
        """Read the most recently written frame.

        Returns a **copy** of the frame to prevent the caller from holding
        a reference to memory that may be overwritten by the next write.

        Returns:
            Tuple of (frame_copy, timestamp, slot_index).

        Raises:
            RuntimeError: If the buffer is empty (no frames written yet).
        """
        if self._closed:
            raise RuntimeError("Cannot read from a closed buffer.")

        if self._frame_count.value == 0:
            raise RuntimeError("Buffer is empty — no frames have been written yet.")

        # The latest frame is at (write_head - 1) mod capacity
        idx = (self._write_head.value - 1) % self._capacity
        frame_copy = self._frames[idx].copy()
        ts = float(self._timestamps[idx])

        return frame_copy, ts, idx

    def read_by_timestamp(
        self,
        target_ts: float,
        tolerance_ms: float = 100.0,
    ) -> tuple[NDArray[np.uint8], float, int] | None:
        """Find and return the frame closest to a target timestamp.

        Performs a linear scan over the timestamp array. Since capacity
        is bounded (typically ≤ 300), this is efficient enough (~O(N)
        with N ≤ 300 float comparisons).

        Args:
            target_ts: The monotonic timestamp to search for.
            tolerance_ms: Maximum allowed difference in milliseconds.
                          Returns None if no frame is within tolerance.

        Returns:
            Tuple of (frame_copy, actual_timestamp, slot_index), or None
            if no frame is within the tolerance window.
        """
        if self._closed:
            raise RuntimeError("Cannot read from a closed buffer.")

        if self._frame_count.value == 0:
            return None

        tolerance_s = tolerance_ms / 1000.0

        # Snapshot the timestamp array (atomic-ish read of float64 values)
        ts_snapshot = self._timestamps.copy()

        # Find the slot with the closest timestamp
        valid_mask = ts_snapshot > 0.0  # Filter unwritten slots
        if not np.any(valid_mask):
            return None

        diffs = np.abs(ts_snapshot - target_ts)
        # Set invalid slots to infinity so they're never selected
        diffs[~valid_mask] = np.inf

        best_idx = int(np.argmin(diffs))
        best_diff = diffs[best_idx]

        if best_diff > tolerance_s:
            logger.debug(
                f"No frame within {tolerance_ms}ms of target "
                f"(closest was {best_diff * 1000:.1f}ms away at slot {best_idx})"
            )
            return None

        frame_copy = self._frames[best_idx].copy()
        actual_ts = float(ts_snapshot[best_idx])

        return frame_copy, actual_ts, best_idx

    def read_at_index(self, index: int) -> tuple[NDArray[np.uint8], float]:
        """Read a frame at a specific slot index.

        Args:
            index: Slot index in the range [0, capacity).

        Returns:
            Tuple of (frame_copy, timestamp).

        Raises:
            IndexError: If index is out of range.
        """
        if self._closed:
            raise RuntimeError("Cannot read from a closed buffer.")

        if not 0 <= index < self._capacity:
            raise IndexError(
                f"Slot index {index} out of range [0, {self._capacity})"
            )

        frame_copy = self._frames[index].copy()
        ts = float(self._timestamps[index])
        return frame_copy, ts

    @property
    def stats(self) -> BufferStats:
        """Current buffer statistics for monitoring dashboards."""
        total_written = self._frame_count.value
        occupancy = min(1.0, total_written / self._capacity) if self._capacity > 0 else 0.0
        frame_bytes = int(np.prod(self._frame_shape)) * np.dtype(np.uint8).itemsize
        total_bytes = self._capacity * frame_bytes

        return BufferStats(
            capacity=self._capacity,
            frames_written=total_written,
            occupancy=occupancy,
            frame_shape=self._frame_shape,
            memory_bytes=total_bytes,
            buffer_id=self._buffer_id,
        )

    def close(self) -> None:
        """Detach from shared memory segments.

        Call this from every process that has accessed the buffer.
        After calling ``close()``, the buffer instance can no longer
        be used for reads or writes.
        """
        if self._closed:
            return

        self._closed = True
        try:
            self._shm_frames.close()
        except Exception as exc:
            logger.warning(f"Error closing frame SHM: {exc}")
        try:
            self._shm_timestamps.close()
        except Exception as exc:
            logger.warning(f"Error closing timestamp SHM: {exc}")

        logger.info(f"Frame buffer '{self._buffer_id}' closed.")

    def unlink(self) -> None:
        """Destroy the shared memory segments permanently.

        Call this **only** from the process that created the buffer
        (the owner). After unlinking, the shared memory is released
        back to the OS and is no longer accessible by any process.
        """
        try:
            self._shm_frames.unlink()
        except Exception as exc:
            logger.warning(f"Error unlinking frame SHM: {exc}")
        try:
            self._shm_timestamps.unlink()
        except Exception as exc:
            logger.warning(f"Error unlinking timestamp SHM: {exc}")

        logger.info(f"Frame buffer '{self._buffer_id}' unlinked (memory released).")

    def __repr__(self) -> str:
        return (
            f"RollingFrameBuffer("
            f"capacity={self._capacity}, "
            f"frame_shape={self._frame_shape}, "
            f"id='{self._buffer_id}', "
            f"written={self._frame_count.value})"
        )
