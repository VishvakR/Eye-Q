"""Tests for Stage 1 instance tracking.

Track identity is what turns "a person was judged" into "this person was
judged". Every assertion here is about identity surviving the things a real
camera does: subjects drifting, arriving, leaving, and passing each other.
"""

from __future__ import annotations

from app.pipelines.tracker import GreedyIoUTracker
from app.schemas.events import BoundingBox


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class TestTrackAssignment:
    """Ids must be stable for the same object and distinct between objects."""

    def test_first_frame_assigns_distinct_ids(self):
        tracker = GreedyIoUTracker()
        ids = tracker.update(
            [("person", box(0.0, 0.0, 0.2, 0.4)), ("person", box(0.6, 0.0, 0.8, 0.4))]
        )

        assert len(set(ids)) == 2, "Two people must not share one id"

    def test_identical_box_keeps_its_id(self):
        tracker = GreedyIoUTracker()
        first = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])
        second = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])

        assert first == second

    def test_small_drift_keeps_the_id(self):
        """A walking subject must stay the same subject between frames."""
        tracker = GreedyIoUTracker(iou_threshold=0.3)
        first = tracker.update([("person", box(0.10, 0.1, 0.30, 0.5))])
        second = tracker.update([("person", box(0.13, 0.1, 0.33, 0.5))])

        assert first == second

    def test_disjoint_box_mints_a_new_id(self):
        tracker = GreedyIoUTracker()
        first = tracker.update([("person", box(0.0, 0.0, 0.2, 0.4))])
        second = tracker.update([("person", box(0.7, 0.6, 0.9, 0.9))])

        assert first != second

    def test_ids_are_never_shared_across_labels(self):
        """A person and the forklift they drive overlap almost exactly."""
        tracker = GreedyIoUTracker()
        ids = tracker.update(
            [("person", box(0.1, 0.1, 0.3, 0.5)), ("forklift", box(0.1, 0.1, 0.3, 0.5))]
        )

        assert ids[0] != ids[1]

    def test_a_new_arrival_does_not_disturb_existing_ids(self):
        tracker = GreedyIoUTracker()
        [first_id] = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])
        ids = tracker.update(
            [("person", box(0.1, 0.1, 0.3, 0.5)), ("person", box(0.6, 0.1, 0.8, 0.5))]
        )

        assert ids[0] == first_id
        assert ids[1] != first_id

    def test_ids_are_returned_in_input_order(self):
        """The caller zips these back onto its own list — order is the contract."""
        tracker = GreedyIoUTracker()
        left, right = tracker.update(
            [("person", box(0.0, 0.0, 0.2, 0.4)), ("person", box(0.6, 0.0, 0.8, 0.4))]
        )

        # Same two boxes, submitted the other way round.
        swapped = tracker.update(
            [("person", box(0.6, 0.0, 0.8, 0.4)), ("person", box(0.0, 0.0, 0.2, 0.4))]
        )
        assert swapped == [right, left]

    def test_greedy_matching_prefers_the_better_overlap(self):
        """Near-crossing subjects must not trade ids while still distinguishable."""
        tracker = GreedyIoUTracker(iou_threshold=0.1)
        a_id, b_id = tracker.update(
            [("person", box(0.10, 0.1, 0.30, 0.5)), ("person", box(0.40, 0.1, 0.60, 0.5))]
        )

        # Both step toward each other but each still overlaps itself far more.
        ids = tracker.update(
            [("person", box(0.16, 0.1, 0.36, 0.5)), ("person", box(0.34, 0.1, 0.54, 0.5))]
        )
        assert ids == [a_id, b_id]

    def test_one_track_cannot_absorb_two_detections(self):
        """Two boxes over one old track must yield two ids, not one."""
        tracker = GreedyIoUTracker(iou_threshold=0.1)
        tracker.update([("person", box(0.2, 0.1, 0.5, 0.5))])
        ids = tracker.update(
            [("person", box(0.2, 0.1, 0.45, 0.5)), ("person", box(0.25, 0.1, 0.5, 0.5))]
        )

        assert len(set(ids)) == 2


class TestTrackLifecycle:
    """Tracks must expire, and expiring is what bounds memory."""

    def test_track_survives_a_brief_disappearance(self):
        """A subject occluded for a frame or two is still the same subject."""
        tracker = GreedyIoUTracker(max_age_frames=5)
        [first] = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])

        tracker.update([])  # occluded
        tracker.update([])

        [again] = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])
        assert again == first

    def test_track_retires_after_max_age(self):
        tracker = GreedyIoUTracker(max_age_frames=3)
        [first] = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])

        for _ in range(5):
            tracker.update([])

        assert tracker.active_count == 0
        [again] = tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))])
        assert again != first, "A long-gone subject should be treated as new"

    def test_ids_are_never_reused(self):
        tracker = GreedyIoUTracker(max_age_frames=1)
        seen = set()
        for _ in range(10):
            seen.update(tracker.update([("person", box(0.1, 0.1, 0.3, 0.5))]))
            for _ in range(3):
                tracker.update([])

        assert len(seen) == 10

    def test_state_stays_bounded_under_heavy_churn(self):
        """Weeks of uptime must not accumulate tracks — this is the leak test."""
        tracker = GreedyIoUTracker(max_age_frames=5)
        for i in range(500):
            offset = (i % 9) * 0.1
            tracker.update([("person", box(offset, 0.0, offset + 0.05, 0.1))])

        assert tracker.active_count <= 10, (
            f"Tracker retained {tracker.active_count} tracks after 500 frames"
        )

    def test_empty_frames_are_safe(self):
        tracker = GreedyIoUTracker()
        assert tracker.update([]) == []
        assert tracker.active_count == 0
