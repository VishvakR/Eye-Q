"""Stage 1 instance tracking — stable identity for each detected object.

YOLO answers *what* is in the frame, never *who*. Two consecutive frames
containing the same five workers produce ten unrelated boxes, so anything
keyed on the class label alone — the trigger cooldown, the Stage 2 batch
coalescer — treats all five as interchangeable and judges only one.

This module assigns each detection a stable integer id by matching it against
the previous frame's boxes on intersection-over-union. That id is what lets
the rest of the pipeline say "this person has already been judged" rather than
"a person has already been judged."

Why greedy IoU rather than a real tracker:
    - **No dependency.** Ultralytics' ByteTrack calls ``check_requirements()``
      on ``lap`` at *import* time, i.e. it attempts a pip install at runtime.
      On an offline appliance that is a hard failure, not a slow path.
    - **Free.** O(N·M·min(N,M)) with N under 20 is a few thousand float
      operations — invisible beside YOLO's 2-4 ms.
    - **Deterministic**, so it unit-tests without loading any model.

The trade-off is honest: no motion model, so it will swap ids when two objects
occlude each other and re-cross. The cost of a swap is one redundant inference,
not a missed alert, because a swapped id looks like a new subject and gets
judged rather than skipped.

Example:
    >>> tracker = GreedyIoUTracker()
    >>> tracker.update([("person", box_a), ("person", box_b)])
    [1, 2]
    >>> tracker.update([("person", box_a_moved)])   # b left the frame
    [1]
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.events import BoundingBox


@dataclass
class _Track:
    """One tracked object between frames."""

    track_id: int
    label: str
    bbox: BoundingBox
    last_seen: int


class GreedyIoUTracker:
    """Assigns stable per-instance ids by greedy IoU matching.

    Ids are unique across labels, never reused, and monotonically increasing,
    so ``track_id`` alone is a valid key downstream — no ``(label, id)`` pair
    is needed to disambiguate.

    Args:
        iou_threshold: Minimum overlap for a detection to continue an existing
            track. Below this it is treated as a new object.
        max_age_frames: Frames a track survives unmatched before it is retired.
            This is what bounds memory: an object that leaves the frame stops
            costing anything within roughly half a second at 30 FPS.
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_age_frames: int = 60,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_frames = max_age_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame_no = 0

    @property
    def active_count(self) -> int:
        """Number of tracks currently alive."""
        return len(self._tracks)

    def reset(self) -> None:
        """Forget every track. Ids are not reused afterwards."""
        self._tracks.clear()

    def update(self, detections: list[tuple[str, BoundingBox]]) -> list[int]:
        """Assign a track id to each detection in this frame.

        Args:
            detections: ``(label, bbox)`` pairs for one frame.

        Returns:
            Track ids **in the same order as the input**, so the caller can zip
            them straight back onto its own detection list.
        """
        self._frame_no += 1

        if not detections:
            self._evict()
            return []

        assigned: list[int | None] = [None] * len(detections)
        claimed: set[int] = set()

        # Match within a label only. Two different classes at the same pixels
        # (a person and the forklift they are driving) are separate objects,
        # and their boxes can overlap almost perfectly.
        by_label: dict[str, list[int]] = {}
        for idx, (label, _bbox) in enumerate(detections):
            by_label.setdefault(label, []).append(idx)

        for label, indices in by_label.items():
            candidates = [t for t in self._tracks.values() if t.label == label]
            if not candidates:
                continue

            # Score every (detection, track) pair once, then take them
            # best-first. Sorting by index as a tie-break keeps the assignment
            # deterministic when two pairs score identically.
            pairs = []
            for idx in indices:
                bbox = detections[idx][1]
                for track in candidates:
                    iou = bbox.iou(track.bbox)
                    if iou >= self._iou_threshold:
                        pairs.append((-iou, idx, track.track_id))
            pairs.sort()

            for _neg_iou, idx, track_id in pairs:
                if assigned[idx] is not None or track_id in claimed:
                    continue
                assigned[idx] = track_id
                claimed.add(track_id)

        # Anything unmatched is a newly arrived object.
        result: list[int] = []
        for idx, (label, bbox) in enumerate(detections):
            track_id = assigned[idx]
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1

            self._tracks[track_id] = _Track(
                track_id=track_id,
                label=label,
                bbox=bbox,
                last_seen=self._frame_no,
            )
            result.append(track_id)

        self._evict()
        return result

    def _evict(self) -> None:
        """Retire tracks not seen for ``max_age_frames``.

        This is what keeps tracker state bounded — without it a busy scene
        would accumulate a track per object per lifetime of the process.
        """
        cutoff = self._frame_no - self._max_age_frames
        stale = [tid for tid, track in self._tracks.items() if track.last_seen < cutoff]
        for tid in stale:
            del self._tracks[tid]

    def __repr__(self) -> str:
        return (
            f"GreedyIoUTracker(active={self.active_count}, "
            f"iou>={self._iou_threshold}, max_age={self._max_age_frames})"
        )
