"""Rolling frame buffer backed by POSIX shared memory.

This module provides a zero-copy circular buffer for video frames,
designed for high-throughput producer/consumer pipelines where the
video capture thread writes frames continuously and downstream
consumers (YOLO watchdog, VLM worker) read them without blocking
the producer.

Architecture:
    - Three shared memory segments: pixel data (uint8), timestamps (float64),
      and a two-slot int64 meta block holding the write head and frame count.
    - Circular write via the shared write head.
    - Writer-side locking only; readers are lock-free (stale-OK semantics).

Torn reads are detected, not tolerated. A slot's timestamp is zeroed before
its pixels are overwritten and restored after, a sequence-lock in miniature:
a reader that re-reads the timestamp after copying and finds it changed knows
the pixels it holds may belong to two different frames. ``read_latest`` needs
no such check — the head only advances once a write is complete, and a slot
cannot come round again within one copy — but ``read_by_timestamp`` and
``read_at_index`` reach back into slots the writer is about to reuse. With the
default 150 frames at 30 FPS and a 5 s staleness threshold, Stage 2 lands on
the oldest slot routinely.

Example:
    >>> buffer = RollingFrameBuffer(capacity=300, frame_height=720, frame_width=1280)
    >>> buffer.write(frame, timestamp=time.monotonic())
    >>> latest_frame, ts, idx = buffer.read_latest()
    >>> buffer.close()
    >>> buffer.unlink()  # Only in the owning process
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
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

        SHM Block 3 ("eyeq_meta_{id}"):
            np.ndarray of shape (2,), dtype=int64 — [write_head, frame_count]

    The counters live in shared memory rather than ``multiprocessing.Value``
    so that readers in other processes still see them, while the buffer owns
    every resource it allocates. ``multiprocessing`` primitives each carry a
    POSIX semaphore that ``unlink()`` cannot release, which leaked on every
    shutdown. Writers are threads in one process, so a ``threading.Lock``
    is the right level of exclusion here.

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
        total_meta_bytes = 2 * np.dtype(np.int64).itemsize

        logger.info(
            f"Allocating frame buffer '{buffer_id}': "
            f"{capacity} frames × {frame_height}×{frame_width}×{frame_channels} = "
            f"{total_frame_bytes / (1024**2):.1f} MB"
        )

        # Helper to safely create shared memory, cleaning up existing orphaned ones
        def _safe_create_shm(name: str, size: int) -> SharedMemory:
            try:
                return SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                logger.warning(
                    f"Shared memory '{name}' already exists. Unlinking and recreating..."
                )
                existing = SharedMemory(name=name)
                existing.unlink()
                return SharedMemory(name=name, create=True, size=size)

        # Allocate the three segments. If a later one fails, release the ones
        # already created: nothing else holds their names, so they would
        # outlive the process.
        created: list[SharedMemory] = []
        try:
            self._shm_frames = _safe_create_shm(f"eyeq_frames_{buffer_id}", total_frame_bytes)
            created.append(self._shm_frames)
            self._shm_timestamps = _safe_create_shm(f"eyeq_ts_{buffer_id}", total_ts_bytes)
            created.append(self._shm_timestamps)
            self._shm_meta = _safe_create_shm(f"eyeq_meta_{buffer_id}", total_meta_bytes)
        except Exception:
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as exc:
                    logger.warning(f"Error releasing SHM '{shm.name}': {exc}")
            raise

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

        self._meta: NDArray[np.int64] = np.ndarray(
            (2,),
            dtype=np.int64,
            buffer=self._shm_meta.buf,
        )

        # Initialize timestamps to zero (marks slots as unwritten)
        self._timestamps[:] = 0.0
        self._meta[:] = 0

        # Writer-side lock (readers are lock-free)
        self._write_lock = threading.Lock()

        self._closed = False

        logger.info(
            f"Frame buffer '{buffer_id}' allocated: "
            f"{total_frame_bytes / (1024**2):.1f} MB frames + "
            f"{total_ts_bytes / 1024:.1f} KB timestamps + "
            f"{total_meta_bytes} B meta"
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

    @property
    def _write_head(self) -> int:
        """Slot index the next frame will be written to."""
        return int(self._meta[0])

    @property
    def _frame_count(self) -> int:
        """Total frames written since allocation (monotonic, never wraps)."""
        return int(self._meta[1])

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

        needs_resize = frame.shape[0] != expected_h or frame.shape[1] != expected_w

        with self._write_lock:
            idx = int(self._meta[0])

            # Zero the timestamp first so a reader can tell this slot is being
            # rewritten: ts > 0 excludes it from lookups while the pixels are
            # in flux, and a reader that already picked it sees the change
            # when it re-checks. See the module docstring.
            self._timestamps[idx] = 0.0
            if needs_resize:
                # Straight into the slot — no intermediate frame-sized array.
                import cv2

                cv2.resize(frame, (expected_w, expected_h), dst=self._frames[idx])
            else:
                np.copyto(self._frames[idx], frame)
            self._timestamps[idx] = timestamp

            # Advance the write head (circular)
            self._meta[0] = (idx + 1) % self._capacity
            self._meta[1] += 1

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

        if self._frame_count == 0:
            raise RuntimeError("Buffer is empty — no frames have been written yet.")

        # The latest frame is at (write_head - 1) mod capacity
        idx = (self._write_head - 1) % self._capacity
        frame_copy = self._frames[idx].copy()
        ts = float(self._timestamps[idx])

        return frame_copy, ts, idx

    def latest_timestamp(self) -> float | None:
        """Timestamp of the most recent frame, without copying its pixels.

        Lets a consumer that polls faster than capture — the watchdog under a
        slow source — recognise a frame it already processed before paying
        for a full-frame copy.

        Returns:
            The timestamp, or None if nothing has been written yet.
        """
        if self._closed:
            raise RuntimeError("Cannot read from a closed buffer.")
        if self._frame_count == 0:
            return None
        return float(self._timestamps[(self._write_head - 1) % self._capacity])

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

        if self._frame_count == 0:
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

        # The slot was chosen from a snapshot, and it is often the oldest one
        # — the next the writer reuses. If its timestamp moved while the
        # pixels were being copied, they may be half one frame and half the
        # next, or a different frame entirely.
        if float(self._timestamps[best_idx]) != actual_ts:
            logger.debug(f"Slot {best_idx} was overwritten during read; discarding.")
            return None

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

        ts = float(self._timestamps[index])
        frame_copy = self._frames[index].copy()
        if float(self._timestamps[index]) != ts:
            # Overwritten mid-copy. Report the slot as unwritten rather than
            # hand back pixels no timestamp describes.
            return frame_copy, 0.0
        return frame_copy, ts

    @property
    def stats(self) -> BufferStats:
        """Current buffer statistics for monitoring dashboards."""
        total_written = self._frame_count
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

        # Drop the numpy views first — SharedMemory.close() fails while an
        # exported buffer still references the mapping.
        self._frames = None
        self._timestamps = None
        self._meta = None

        try:
            self._shm_frames.close()
        except Exception as exc:
            logger.warning(f"Error closing frame SHM: {exc}")
        try:
            self._shm_timestamps.close()
        except Exception as exc:
            logger.warning(f"Error closing timestamp SHM: {exc}")
        try:
            self._shm_meta.close()
        except Exception as exc:
            logger.warning(f"Error closing meta SHM: {exc}")

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
        try:
            self._shm_meta.unlink()
        except Exception as exc:
            logger.warning(f"Error unlinking meta SHM: {exc}")

        logger.info(f"Frame buffer '{self._buffer_id}' unlinked (memory released).")

    def __repr__(self) -> str:
        return (
            f"RollingFrameBuffer("
            f"capacity={self._capacity}, "
            f"frame_shape={self._frame_shape}, "
            f"id='{self._buffer_id}', "
            f"written={self._frame_count})"
        )
