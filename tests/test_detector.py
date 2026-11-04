"""Tests for the Stage 1 YOLO Watchdog and supporting components.

Test groups:
    - RegionOfInterest: point-in-polygon, bbox containment, factory methods
    - CooldownGate: trigger throttling, reset, timing
    - YOLOWatchdog integration: model loading, frame processing, event publishing
      (uses synthetic frames and mocking where YOLO model is unavailable)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.roi import RegionOfInterest
from app.pipelines.detector import (
    _COOLDOWN_PRUNE_AT,
    CooldownGate,
    YOLOWatchdog,
)
from app.schemas.events import BoundingBox

# ═══════════════════════════════════════════════════════════════════
# Ultralytics Result Stubs
# ═══════════════════════════════════════════════════════════════════


class _StubTensor:
    """Minimal stand-in for the torch tensor Ultralytics hands back."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _StubTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _StubBoxes:
    """Stands in for ``ultralytics.engine.results.Boxes``.

    Stage 1 reads a frame's detections as one ``(N, 6)`` block —
    ``[x1, y1, x2, y2, conf, cls]`` — rather than field by field per box, so
    the stub carries a real array. Mocking the per-box accessors instead would
    let the production code drift back to the synchronization-heavy form
    without a test noticing.
    """

    def __init__(self, rows: list[list[float]]) -> None:
        self.data = _StubTensor(np.array(rows, dtype=np.float32).reshape(-1, 6))

    def __len__(self) -> int:
        return len(self.data.numpy())


def _stub_result(detections):
    """Build a stub ultralytics result from the given detections.

    Each entry is ``(label, conf)`` — which keeps the original box, so callers
    written before tracking existed are unaffected — or ``(label, conf, xyxy)``
    to place boxes apart, which multi-instance tests need: identical pixels are
    correctly one object, not several.
    """
    names, rows = {}, []
    for i, detection in enumerate(detections):
        if len(detection) == 3:
            label, conf, xyxy = detection
        else:
            label, conf = detection
            xyxy = [10.0, 10.0, 50.0, 50.0]

        names[i] = label
        rows.append([*xyxy, conf, float(i)])

    result = MagicMock()
    result.names = names
    result.boxes = _StubBoxes(rows)
    return result

# ═══════════════════════════════════════════════════════════════════
# Bounding Box Geometry Tests
# ═══════════════════════════════════════════════════════════════════


class TestBoundingBoxIoU:
    """IoU underpins both frame-to-frame tracking and re-judge decisions."""

    def test_identical_boxes_fully_overlap(self):
        box = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        left = BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.2)
        right = BoundingBox(x1=0.6, y1=0.6, x2=0.9, y2=0.9)
        assert left.iou(right) == 0.0

    def test_touching_edges_do_not_overlap(self):
        """Shared edge, zero area — adjacency is not identity."""
        left = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
        right = BoundingBox(x1=0.4, y1=0.0, x2=0.8, y2=0.4)
        assert left.iou(right) == 0.0

    def test_half_overlap_is_one_third(self):
        """Two unit-area boxes sharing half their area: 0.5 / 1.5."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
        b = BoundingBox(x1=0.2, y1=0.0, x2=0.6, y2=0.4)
        assert a.iou(b) == pytest.approx(1.0 / 3.0)

    def test_contained_box_scores_the_area_ratio(self):
        outer = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
        inner = BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
        assert outer.iou(inner) == pytest.approx(0.25)

    def test_is_symmetric(self):
        a = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.7)
        b = BoundingBox(x1=0.3, y1=0.1, x2=0.8, y2=0.6)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_degenerate_box_returns_zero_not_an_error(self):
        """A zero-area box must not divide by zero — YOLO can emit one."""
        flat = BoundingBox(x1=0.3, y1=0.3, x2=0.3, y2=0.7)
        real = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.9)
        assert flat.iou(real) == 0.0
        assert flat.iou(flat) == 0.0


class TestBoundingBoxOverlapRatio:
    """Containment, not similarity — the predicate co-occurrence gating needs."""

    def test_a_contained_small_box_scores_one_where_iou_scores_almost_nothing(self):
        """This test is the whole reason the method exists.

        A phone fully inside a person scores ~0.028 on IoU, not because the
        phone is barely inside but because the person is ~36x larger and IoU
        divides by the union. Any IoU threshold loose enough to admit this pair
        also admits two boxes that merely graze at a corner.
        """
        person = BoundingBox(x1=0.20, y1=0.20, x2=0.60, y2=0.90)
        phone = BoundingBox(x1=0.35, y1=0.45, x2=0.41, y2=0.58)

        assert person.iou(phone) < 0.03
        assert person.overlap_ratio(phone) == pytest.approx(1.0)

    def test_identical_boxes_score_one(self):
        box = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
        assert box.overlap_ratio(box) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        left = BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.2)
        right = BoundingBox(x1=0.6, y1=0.6, x2=0.9, y2=0.9)
        assert left.overlap_ratio(right) == 0.0

    def test_touching_edges_score_zero(self):
        """A phone beside someone is not a phone in their hand."""
        left = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
        right = BoundingBox(x1=0.4, y1=0.0, x2=0.8, y2=0.4)
        assert left.overlap_ratio(right) == 0.0

    def test_half_the_smaller_box_inside_scores_one_half(self):
        """Same geometry IoU would call 0.2 — the divisor is the difference."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.4, y2=0.4)
        b = BoundingBox(x1=0.3, y1=0.0, x2=0.5, y2=0.4)
        assert a.overlap_ratio(b) == pytest.approx(0.5)
        assert a.iou(b) == pytest.approx(0.2)

    def test_is_symmetric(self):
        """`min` is symmetric, so a rule cannot depend on argument order."""
        a = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.7)
        b = BoundingBox(x1=0.3, y1=0.1, x2=0.8, y2=0.6)
        assert a.overlap_ratio(b) == pytest.approx(b.overlap_ratio(a))

    def test_degenerate_box_returns_zero_not_an_error(self):
        flat = BoundingBox(x1=0.3, y1=0.3, x2=0.3, y2=0.7)
        real = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.9)
        assert flat.overlap_ratio(real) == 0.0
        assert flat.overlap_ratio(flat) == 0.0


class TestBoundingBoxPixelSize:
    """Both stages' size floors measure through this one method."""

    def test_reports_the_boxs_own_extent(self):
        box = BoundingBox(x1=0.25, y1=0.10, x2=0.75, y2=0.60)
        assert box.pixel_size(640, 480) == (320, 240)

    def test_both_stages_measure_the_same_box(self):
        """The only thing standing between the two floors and a 1px drift."""
        from app.pipelines.vlm_worker import subject_pixel_size

        box = BoundingBox(x1=0.2, y1=0.2, x2=0.6, y2=0.9)
        assert subject_pixel_size((480, 640, 3), box) == box.pixel_size(640, 480)


# ═══════════════════════════════════════════════════════════════════
# Region of Interest Tests
# ═══════════════════════════════════════════════════════════════════


class TestRegionOfInterest:
    """Test ROI geometry operations."""

    def test_full_frame_contains_center(self):
        """Full-frame ROI should contain any point in [0, 1]."""
        roi = RegionOfInterest.full_frame()
        assert roi.contains_point(0.5, 0.5)
        assert roi.contains_point(0.01, 0.01)
        assert roi.contains_point(0.99, 0.99)

    def test_full_frame_excludes_outside(self):
        """Full-frame ROI should exclude points outside [0, 1]."""
        roi = RegionOfInterest.full_frame()
        # Points exactly on boundary may be in or out depending on ray-cast
        # but clearly outside points should be excluded
        assert not roi.contains_point(1.5, 0.5)
        assert not roi.contains_point(-0.1, 0.5)

    def test_rectangular_roi_contains(self):
        """Rectangular ROI should contain interior points."""
        roi = RegionOfInterest.from_rect(0.2, 0.2, 0.8, 0.8)
        assert roi.contains_point(0.5, 0.5)
        assert roi.contains_point(0.3, 0.3)
        assert roi.contains_point(0.7, 0.7)

    def test_rectangular_roi_excludes(self):
        """Rectangular ROI should exclude exterior points."""
        roi = RegionOfInterest.from_rect(0.2, 0.2, 0.8, 0.8)
        assert not roi.contains_point(0.1, 0.1)
        assert not roi.contains_point(0.9, 0.9)
        assert not roi.contains_point(0.1, 0.5)

    def test_triangular_roi(self):
        """Triangular ROI should work with 3 vertices."""
        roi = RegionOfInterest(
            vertices=[(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)],
            name="triangle",
        )
        # Center of the triangle
        assert roi.contains_point(0.5, 0.7)
        # Outside the triangle (top-left)
        assert not roi.contains_point(0.1, 0.1)

    def test_contains_bbox_center(self):
        """contains_bbox_center should check the bbox center point."""
        roi = RegionOfInterest.from_rect(0.2, 0.2, 0.8, 0.8)

        # Bbox centered at (0.5, 0.5) — inside
        inside_bbox = BoundingBox(x1=0.3, y1=0.3, x2=0.7, y2=0.7)
        assert roi.contains_bbox_center(inside_bbox)

        # Bbox centered at (0.05, 0.05) — outside
        outside_bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        assert not roi.contains_bbox_center(outside_bbox)

    def test_intersects_bbox(self):
        """intersects_bbox should check sampling-based overlap."""
        roi = RegionOfInterest.from_rect(0.3, 0.3, 0.7, 0.7)

        # Bbox fully inside ROI
        inside = BoundingBox(x1=0.35, y1=0.35, x2=0.65, y2=0.65)
        assert roi.intersects_bbox(inside, min_overlap=0.5)

        # Bbox fully outside ROI
        outside = BoundingBox(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        assert not roi.intersects_bbox(outside, min_overlap=0.1)

    def test_from_json(self):
        """from_json should create a valid ROI from coordinate pairs."""
        data = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        roi = RegionOfInterest.from_json(data, name="test_json")
        assert roi.name == "test_json"
        assert len(roi.vertices) == 4
        assert roi.contains_point(0.5, 0.5)

    def test_to_json(self):
        """to_json should produce a serializable list."""
        roi = RegionOfInterest.from_rect(0.2, 0.3, 0.8, 0.9)
        data = roi.to_json()
        assert len(data) == 4
        assert data[0] == [0.2, 0.3]
        assert data[2] == [0.8, 0.9]

    def test_json_roundtrip(self):
        """from_json(to_json()) should produce an equivalent ROI."""
        original = RegionOfInterest.from_rect(0.15, 0.25, 0.85, 0.95)
        data = original.to_json()
        restored = RegionOfInterest.from_json(data)

        assert restored.vertices == original.vertices
        # Both should agree on containment
        assert restored.contains_point(0.5, 0.5) == original.contains_point(0.5, 0.5)
        assert restored.contains_point(0.1, 0.1) == original.contains_point(0.1, 0.1)

    def test_invalid_too_few_vertices(self):
        """ROI with fewer than 3 vertices should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            RegionOfInterest(vertices=[(0.0, 0.0), (1.0, 1.0)])

    def test_invalid_out_of_range(self):
        """ROI with coordinates outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="out of normalized range"):
            RegionOfInterest(vertices=[(0.0, 0.0), (1.5, 0.0), (0.5, 1.0)])

    def test_repr(self):
        """__repr__ should be readable."""
        roi = RegionOfInterest.full_frame()
        r = repr(roi)
        assert "RegionOfInterest" in r
        assert "full_frame" in r


# ═══════════════════════════════════════════════════════════════════
# Cooldown Gate Tests
# ═══════════════════════════════════════════════════════════════════


class TestCooldownGate:
    """Test per-label trigger throttling."""

    def test_first_trigger_always_passes(self):
        """First trigger for any label should always pass."""
        gate = CooldownGate(cooldown_seconds=5.0)
        assert gate.should_trigger("person") is True
        assert gate.should_trigger("car") is True

    def test_first_trigger_passes_on_a_freshly_booted_machine(self, monkeypatch):
        """The first trigger must not wait for the machine's uptime.

        ``time.monotonic()`` counts from boot on Linux and macOS, so a gate
        that stood in ``0.0`` for "never triggered" compared a real clock
        against a boot-relative one and suppressed everything until uptime
        exceeded ``cooldown_seconds``. An appliance that starts on power-up
        is in exactly this state, and a long cooldown — the setting an
        operator reaches for when inference is expensive — makes the blind
        window long too.
        """
        monkeypatch.setattr(
            "app.pipelines.detector.time.monotonic", lambda: 3.0
        )
        gate = CooldownGate(cooldown_seconds=999.0)

        assert gate.should_trigger("person#1") is True
        assert gate.should_trigger("person#1") is False

    def test_immediate_retrigger_blocked(self):
        """Immediate re-trigger of the same label should be blocked."""
        gate = CooldownGate(cooldown_seconds=5.0)
        assert gate.should_trigger("person") is True
        assert gate.should_trigger("person") is False

    def test_different_labels_independent(self):
        """Different labels should have independent cooldowns."""
        gate = CooldownGate(cooldown_seconds=5.0)
        assert gate.should_trigger("person") is True
        assert gate.should_trigger("car") is True
        # person still on cooldown, car just triggered
        assert gate.should_trigger("person") is False
        assert gate.should_trigger("car") is False

    def test_cooldown_expires(self):
        """Trigger should pass after cooldown elapses."""
        gate = CooldownGate(cooldown_seconds=0.1)  # 100ms cooldown
        assert gate.should_trigger("person") is True
        assert gate.should_trigger("person") is False

        time.sleep(0.15)  # Wait for cooldown to expire

        assert gate.should_trigger("person") is True

    def test_zero_cooldown_always_triggers(self):
        """Zero cooldown should allow every trigger."""
        gate = CooldownGate(cooldown_seconds=0.0)
        for _ in range(10):
            assert gate.should_trigger("person") is True

    def test_reset_single_label(self):
        """Resetting a single label should re-enable it."""
        gate = CooldownGate(cooldown_seconds=60.0)
        assert gate.should_trigger("person") is True
        assert gate.should_trigger("person") is False

        gate.reset("person")
        assert gate.should_trigger("person") is True

    def test_reset_all_labels(self):
        """Resetting all labels should re-enable everything."""
        gate = CooldownGate(cooldown_seconds=60.0)
        gate.should_trigger("person")
        gate.should_trigger("car")

        gate.reset()

        assert gate.should_trigger("person") is True
        assert gate.should_trigger("car") is True

    def test_cooldown_setter(self):
        """Cooldown duration should be updatable at runtime."""
        gate = CooldownGate(cooldown_seconds=5.0)
        assert gate.cooldown_seconds == 5.0

        gate.cooldown_seconds = 10.0
        assert gate.cooldown_seconds == 10.0

        # Negative values should be clamped to 0
        gate.cooldown_seconds = -1.0
        assert gate.cooldown_seconds == 0.0


# ═══════════════════════════════════════════════════════════════════
# YOLO Watchdog Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestYOLOWatchdogUnit:
    """Unit tests for YOLOWatchdog using mocked YOLO model."""

    def _make_mock_buffer(self) -> MagicMock:
        """Create a mock buffer that returns a synthetic frame."""
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_latest.return_value = (frame, time.monotonic(), 0)
        return buffer

    def _make_mock_publisher(self) -> MagicMock:
        """Create a mock publisher that always succeeds."""
        pub = MagicMock()
        pub.publish.return_value = True
        return pub

    def test_watchdog_initialization(self):
        """Watchdog should initialize without loading the model."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(
            buffer=buffer,
            publisher=publisher,
            device="cpu",
            model_name="yolov8n.pt",
        )

        assert not watchdog.is_running
        assert watchdog._model is None

    def test_watchdog_stats_initial(self):
        """Initial stats should all be zero."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(
            buffer=buffer,
            publisher=publisher,
        )
        stats = watchdog.stats

        assert stats.frames_processed == 0
        assert stats.detections_total == 0
        assert stats.detections_published == 0
        assert stats.is_running is False

    def test_watchdog_update_roi(self):
        """ROI should be updatable at runtime."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(buffer=buffer, publisher=publisher)

        new_roi = RegionOfInterest.from_rect(0.1, 0.1, 0.5, 0.5)
        watchdog.update_roi(new_roi)
        assert watchdog._roi is new_roi

    def test_watchdog_update_labels(self):
        """Target labels should be updatable at runtime."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(buffer=buffer, publisher=publisher)

        watchdog.update_target_labels({"person", "car"})
        assert watchdog._target_labels == {"person", "car"}

        watchdog.update_target_labels(None)
        assert watchdog._target_labels is None

    def test_watchdog_update_cooldown(self):
        """Cooldown should be updatable at runtime."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(
            buffer=buffer, publisher=publisher, cooldown_seconds=5.0
        )
        watchdog.update_cooldown(10.0)
        assert watchdog._cooldown.cooldown_seconds == 10.0

    def test_watchdog_repr(self):
        """__repr__ should produce a readable string."""
        buffer = self._make_mock_buffer()
        publisher = self._make_mock_publisher()

        watchdog = YOLOWatchdog(
            buffer=buffer, publisher=publisher, model_name="yolov8n.pt"
        )
        r = repr(watchdog)
        assert "YOLOWatchdog" in r
        assert "yolov8n" in r
        assert "stopped" in r


class TestYOLOWatchdogLive:
    """Live integration tests that actually load YOLO and run inference.

    These tests are slower (~5-10s) because they download/load the
    model. They can be skipped in CI with ``pytest -m 'not slow'``.
    """

    @pytest.fixture
    def live_buffer(self):
        """Create a real buffer for live testing."""
        from app.core.buffer import RollingFrameBuffer

        buf = RollingFrameBuffer(
            capacity=10,
            frame_height=480,
            frame_width=640,
            frame_channels=3,
            buffer_id=f"yolo_test_{int(time.monotonic() * 1000)}",
        )
        yield buf
        buf.close()
        buf.unlink()

    @pytest.mark.slow
    def test_watchdog_live_start_stop(self, live_buffer):
        """Watchdog should start, process frames, and stop cleanly."""
        from app.core.ipc import EventPublisher

        endpoint = f"tcp://127.0.0.1:{np.random.randint(16000, 26000)}"

        with EventPublisher(endpoint) as pub:
            # Write a synthetic frame to the buffer
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            live_buffer.write(frame, time.monotonic())

            watchdog = YOLOWatchdog(
                buffer=live_buffer,
                publisher=pub,
                device="cpu",  # Use CPU for CI compatibility
                model_name="yolov8n.pt",
                target_fps=5.0,
            )

            watchdog.start()
            assert watchdog.is_running

            # Let it run for a couple of seconds
            time.sleep(3.0)

            watchdog.stop()
            assert not watchdog.is_running

            stats = watchdog.stats
            assert stats.frames_processed > 0
            assert stats.model_name == "yolov8n.pt"

    @pytest.mark.slow
    def test_watchdog_live_detects_with_real_model(self, live_buffer):
        """Watchdog with real YOLO should produce detections on a test image."""
        from app.core.ipc import EventPublisher, EventSubscriber

        endpoint = f"tcp://127.0.0.1:{np.random.randint(16000, 26000)}"

        with EventPublisher(endpoint) as pub:
            with EventSubscriber(endpoint, topic="detection"):
                # Create a colorful image that YOLO might detect something in
                # (random noise — YOLO likely won't detect anything, but the
                #  pipeline should still process frames without errors)
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                live_buffer.write(frame, time.monotonic())

                watchdog = YOLOWatchdog(
                    buffer=live_buffer,
                    publisher=pub,
                    device="cpu",
                    confidence_threshold=0.01,  # Very low to catch anything
                    target_fps=2.0,
                )

                watchdog.start()
                time.sleep(4.0)
                watchdog.stop()

                # Whether or not YOLO detected anything on random noise,
                # the watchdog should have processed frames without crashing
                assert watchdog.stats.frames_processed > 0


class TestDisplayVsTriggerScope:
    """A narrow rule must scope alerts, not blind the live view.

    `sync_rules()` narrows the watchdog's label whitelist to the labels your
    rules care about. That must gate what wakes the VLM — not what the
    dashboard draws, or adding a "person" rule would hide the forklift beside
    them from the operator.
    """

    def _watchdog(self, **kwargs):
        from app.pipelines.detector import YOLOWatchdog

        return YOLOWatchdog(buffer=MagicMock(), publisher=MagicMock(), **kwargs)

    def _fake_result(self, detections):
        return _stub_result(detections)

    def _run_frame(self, watchdog, detections):
        watchdog._model = MagicMock()
        watchdog._model.predict.return_value = [self._fake_result(detections)]
        watchdog._process_frame(np.zeros((100, 100, 3), dtype=np.uint8), 1.0, 0)

    def test_out_of_scope_labels_still_render(self):
        """The whole scene stays visible even when one label is watched."""
        watchdog = self._watchdog(target_labels={"person"})
        self._run_frame(watchdog, [("person", 0.9), ("forklift", 0.8), ("car", 0.7)])

        drawn = {e.label for e, _fired in watchdog.recent_detections(max_age_s=60)}
        assert drawn == {"person", "forklift", "car"}

    def test_only_in_scope_labels_trigger(self):
        watchdog = self._watchdog(target_labels={"person"})
        self._run_frame(watchdog, [("person", 0.9), ("forklift", 0.8)])

        triggered = {
            e.label for e, fired in watchdog.recent_detections(max_age_s=60) if fired
        }
        assert triggered == {"person"}
        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_label == 1

    def test_cooldown_holds_still_render(self):
        """Boxes must persist between triggers, not blink once per cooldown."""
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, [("person", 0.9)])
        self._run_frame(watchdog, [("person", 0.9)])

        recent = watchdog.recent_detections(max_age_s=60)
        assert len(recent) == 2
        assert [fired for _e, fired in recent] == [True, False]
        assert watchdog.stats.detections_published == 1

    def test_low_confidence_is_not_rendered(self):
        """Confidence is the one filter that also suppresses display."""
        watchdog = self._watchdog(confidence_threshold=0.5)
        self._run_frame(watchdog, [("person", 0.2)])

        assert watchdog.recent_detections(max_age_s=60) == []


# ═══════════════════════════════════════════════════════════════════
# Stage 1 Load Shedding
# ═══════════════════════════════════════════════════════════════════

# Shared geometry for the two gate suites, on a 100x100 frame.
_PERSON = [20.0, 20.0, 60.0, 90.0]  # 40 x 70 px
_PHONE_HELD = [35.0, 45.0, 41.0, 58.0]  # 6 x 13 px, fully inside _PERSON
_PHONE_AWAY = [80.0, 10.0, 86.0, 23.0]  # 6 x 13 px, disjoint from _PERSON
_CAR = [5.0, 5.0, 45.0, 45.0]  # 40 x 40 px


class _GateHarness:
    """Drives `_process_frame` directly on a 100x100 frame."""

    def _watchdog(self, **kwargs):
        return YOLOWatchdog(buffer=MagicMock(), publisher=MagicMock(), **kwargs)

    def _run_frame(self, watchdog, detections, timestamp=1.0):
        watchdog._model = MagicMock()
        watchdog._model.predict.return_value = [_stub_result(detections)]
        watchdog._process_frame(
            np.zeros((100, 100, 3), dtype=np.uint8), timestamp, 0
        )

    def _published(self, watchdog):
        """The payload dicts handed to the ZMQ publisher, in order."""
        return [
            call.args[1] for call in watchdog._publisher.publish.call_args_list
        ]


class TestStageOneSizeFloor(_GateHarness):
    """The floor Stage 2 already applied, moved to where it is cheapest.

    A box rejected here never crosses ZMQ, never costs Stage 2 a ring-buffer
    read, and can no longer displace a legible subject from the batch that
    `vlm_max_instances_per_batch` truncates.
    """

    def test_a_box_under_the_floor_never_publishes(self):
        watchdog = self._watchdog(min_subject_px=40)
        self._run_frame(watchdog, [("person", 0.9, [40.0, 40.0, 70.0, 70.0])])

        assert watchdog.stats.detections_published == 0
        assert watchdog.stats.detections_filtered_size == 1

    def test_a_box_under_the_floor_is_still_drawn(self):
        """The display-vs-trigger invariant: gating decides what wakes the VLM."""
        watchdog = self._watchdog(min_subject_px=40)
        self._run_frame(watchdog, [("person", 0.9, [40.0, 40.0, 70.0, 70.0])])

        recent = watchdog.recent_detections(max_age_s=60)
        assert len(recent) == 1
        assert recent[0][1] is False

    def test_the_longer_side_is_what_counts(self):
        """A person is tall and narrow; both-sides would drop legible subjects."""
        watchdog = self._watchdog(min_subject_px=40)
        self._run_frame(watchdog, [("person", 0.9, [47.0, 25.0, 53.0, 75.0])])

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_size == 0

    def test_zero_disables_the_floor(self):
        watchdog = self._watchdog(min_subject_px=0)
        self._run_frame(watchdog, [("person", 0.9, [10.0, 10.0, 14.0, 14.0])])

        assert watchdog.stats.detections_published == 1

    def test_the_floor_does_not_consume_the_cooldown(self):
        """The ordering test. Fails if the gate moves below the cooldown.

        `CooldownGate.should_trigger` records the trigger and returns True in
        the same breath, so a gate below it would spend this subject's cooldown
        on a frame it discarded — and the person would be suppressed when they
        finally walk close enough to answer.
        """
        watchdog = self._watchdog(min_subject_px=40, cooldown_seconds=999.0)
        self._run_frame(watchdog, [("person", 0.9, [40.0, 40.0, 70.0, 70.0])])
        self._run_frame(watchdog, [("person", 0.9, [35.0, 35.0, 85.0, 85.0])])

        recent = watchdog.recent_detections(max_age_s=60)
        # Vacuous unless both frames resolved to the same subject.
        assert recent[0][0].track_id == recent[1][0].track_id

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_cooldown == 0

    def test_a_subject_under_the_floor_still_reaches_the_tracker(self):
        """Fails if anyone "optimises" the floor into Pass 1.

        Filtering before `tracker.update()` would give a subject a fresh
        track_id on crossing the floor, and Stage 2's memo is keyed
        (track_id, rule_id) — so every approach would become a cache miss and
        the filter would *increase* VLM load.
        """
        watchdog = self._watchdog(min_subject_px=40)
        self._run_frame(watchdog, [("person", 0.9, [40.0, 40.0, 70.0, 70.0])])
        self._run_frame(watchdog, [("person", 0.9, [35.0, 35.0, 85.0, 85.0])])

        recent = watchdog.recent_detections(max_age_s=60)
        assert recent[0][0].track_id == recent[1][0].track_id
        assert watchdog.stats.tracks_active == 1


class TestCoOccurrenceGating(_GateHarness):
    """Only wake the VLM about a person who actually has a phone."""

    PAIR = {"person": {"cell phone"}}

    def test_a_person_without_the_partner_does_not_trigger(self):
        watchdog = self._watchdog(pair_requirements=self.PAIR)
        self._run_frame(watchdog, [("person", 0.9, _PERSON)])

        assert watchdog.stats.detections_published == 0
        assert watchdog.stats.detections_filtered_pair == 1

    def test_a_person_with_an_overlapping_partner_triggers(self):
        """And the *person's* box publishes — not the phone's, not a union."""
        watchdog = self._watchdog(pair_requirements=self.PAIR)
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_HELD)],
        )

        people = [p for p in self._published(watchdog) if p["label"] == "person"]
        assert len(people) == 1
        assert people[0]["bbox"] == pytest.approx(
            {"x1": 0.20, "y1": 0.20, "x2": 0.60, "y2": 0.90}
        )
        assert watchdog.stats.detections_filtered_pair == 0

    def test_the_pair_is_found_even_when_the_partner_is_out_of_scope(self):
        """The ordering test. Fails if the pair search moves below the label gate.

        `sync_rules` narrows target_labels to {"person"} for a person-only
        rule, so the phone would already be gone if the search ran after it.
        """
        watchdog = self._watchdog(
            pair_requirements=self.PAIR, target_labels={"person"}
        )
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_HELD)],
        )

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_label == 1

    def test_a_disjoint_partner_does_not_pair(self):
        """A phone on a shelf across the room is not a phone in a hand."""
        watchdog = self._watchdog(pair_requirements=self.PAIR)
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_AWAY)],
        )

        assert watchdog.stats.detections_published == 1  # the phone itself
        assert self._published(watchdog)[0]["label"] == "cell phone"
        assert watchdog.stats.detections_filtered_pair == 1

    def test_iou_would_have_missed_this_pair(self):
        """Guards against a future "simplification" back to `iou`."""
        person = BoundingBox(x1=0.20, y1=0.20, x2=0.60, y2=0.90)
        phone = BoundingBox(x1=0.35, y1=0.45, x2=0.41, y2=0.58)
        assert person.iou(phone) < 0.03

        watchdog = self._watchdog(pair_requirements=self.PAIR)
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_HELD)],
        )
        assert watchdog.stats.detections_filtered_pair == 0

    def test_a_tiny_partner_is_not_dropped_by_the_size_floor(self):
        """The interaction test.

        A phone is *always* under the floor. If the pair search ran over a
        size-filtered list the gate could never fire — and it fails closed, so
        Stage 2 would go quiet while every counter looked like a calm scene.
        """
        watchdog = self._watchdog(pair_requirements=self.PAIR, min_subject_px=40)
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_HELD)],
        )

        published = self._published(watchdog)
        assert len(published) == 1
        assert published[0]["label"] == "person"
        # The phone was too small to publish itself, yet still a valid partner.
        assert watchdog.stats.detections_filtered_size == 1
        assert watchdog.stats.detections_filtered_pair == 0

    def test_other_labels_are_unaffected_by_the_pair_gate(self):
        """"person + cell phone" says nothing about a car."""
        watchdog = self._watchdog(
            pair_requirements=self.PAIR, target_labels={"person", "car"}
        )
        self._run_frame(watchdog, [("person", 0.9, _PERSON), ("car", 0.8, _CAR)])

        published = self._published(watchdog)
        assert [p["label"] for p in published] == ["car"]
        assert watchdog.stats.detections_filtered_pair == 1

    def test_the_pair_gate_does_not_consume_the_cooldown(self):
        """Picking a phone up must not be suppressed by an unspent cooldown."""
        watchdog = self._watchdog(
            pair_requirements=self.PAIR,
            target_labels={"person"},
            cooldown_seconds=999.0,
        )
        self._run_frame(watchdog, [("person", 0.9, _PERSON)])
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, _PHONE_HELD)],
        )

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_cooldown == 0

    def test_a_partial_overlap_below_the_threshold_does_not_pair(self):
        """Default 0.5: at least half the smaller box must be inside."""
        watchdog = self._watchdog(pair_requirements=self.PAIR, pair_overlap=0.5)
        # Two thirds of the phone sits outside the person's right edge.
        self._run_frame(
            watchdog,
            [("person", 0.9, _PERSON), ("cell phone", 0.6, [58.0, 45.0, 64.0, 58.0])],
        )

        assert watchdog.stats.detections_filtered_pair == 1

    def test_no_pair_configured_changes_nothing(self):
        """Regression guard for every appliance not using the gate."""
        watchdog = self._watchdog(pair_requirements=None)
        self._run_frame(watchdog, [("person", 0.9, _PERSON)])

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_pair == 0


class TestMultiInstanceTriggering:
    """A frame with several subjects must wake the VLM about each of them.

    This is the regression suite for the defect where a rule like "alert if a
    person has no helmet" judged exactly one person per frame, chosen by
    whichever box YOLO happened to emit first — so in a group where one worker
    was non-compliant, the violation was usually never looked at.
    """

    # Four people standing apart, in pixels on the 100x100 test frame.
    FOUR_PEOPLE = [
        ("person", 0.9, [5.0, 10.0, 20.0, 60.0]),
        ("person", 0.9, [30.0, 10.0, 45.0, 60.0]),
        ("person", 0.9, [55.0, 10.0, 70.0, 60.0]),
        ("person", 0.9, [78.0, 10.0, 93.0, 60.0]),
    ]

    def _watchdog(self, **kwargs):
        from app.pipelines.detector import YOLOWatchdog

        return YOLOWatchdog(buffer=MagicMock(), publisher=MagicMock(), **kwargs)

    def _fake_result(self, detections):
        return _stub_result(detections)

    def _run_frame(self, watchdog, detections):
        watchdog._model = MagicMock()
        watchdog._model.predict.return_value = [self._fake_result(detections)]
        watchdog._process_frame(np.zeros((100, 100, 3), dtype=np.uint8), 1.0, 0)

    def test_every_person_in_the_frame_triggers(self):
        """The headline fix: four people, four triggers — not one."""
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert watchdog.stats.detections_published == 4
        assert watchdog.stats.detections_filtered_cooldown == 0

    def test_each_trigger_carries_a_distinct_track_id(self):
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        published = watchdog._publisher.publish.call_args_list
        track_ids = [call.args[1]["track_id"] for call in published]
        assert len(set(track_ids)) == 4, f"Ids collided: {track_ids}"

    def test_the_same_people_do_not_retrigger_on_the_next_frame(self):
        """Identity is what stops four subjects becoming a trigger firehose."""
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, self.FOUR_PEOPLE)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert watchdog.stats.detections_published == 4
        assert watchdog.stats.detections_filtered_cooldown == 4

    def test_a_new_arrival_triggers_while_others_are_in_cooldown(self):
        """The case that used to fail silently: someone walks in mid-cooldown."""
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, self.FOUR_PEOPLE)
        assert watchdog.stats.detections_published == 4

        newcomer = ("person", 0.9, [5.0, 70.0, 20.0, 95.0])
        self._run_frame(watchdog, [*self.FOUR_PEOPLE, newcomer])

        assert watchdog.stats.detections_published == 5, (
            "A newly arrived subject was suppressed by another subject's cooldown"
        )

    def test_a_moving_subject_keeps_its_id(self):
        """Drift must not look like a new person and burn a fresh inference."""
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, [("person", 0.9, [10.0, 10.0, 30.0, 60.0])])
        self._run_frame(watchdog, [("person", 0.9, [13.0, 10.0, 33.0, 60.0])])

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_cooldown == 1

    def test_tracking_off_restores_the_original_behaviour(self):
        """The kill switch: one trigger per label, exactly as before."""
        watchdog = self._watchdog(cooldown_seconds=999.0, track_enabled=False)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert watchdog.stats.detections_published == 1
        assert watchdog.stats.detections_filtered_cooldown == 3

        published = watchdog._publisher.publish.call_args_list
        assert published[0].args[1]["track_id"] == -1

    def test_out_of_roi_subjects_are_still_excluded(self):
        """Per-instance triggering must not widen the ROI gate."""
        roi = RegionOfInterest.from_rect(0.0, 0.0, 0.5, 1.0)
        watchdog = self._watchdog(cooldown_seconds=999.0, roi=roi)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert watchdog.stats.detections_published == 2
        assert watchdog.stats.detections_filtered_roi == 2

    def test_every_subject_is_still_drawn_regardless_of_gating(self):
        watchdog = self._watchdog(cooldown_seconds=999.0)
        self._run_frame(watchdog, self.FOUR_PEOPLE)
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert len(watchdog.recent_detections(max_age_s=60)) == 8

    def test_tracks_active_is_reported(self):
        watchdog = self._watchdog()
        self._run_frame(watchdog, self.FOUR_PEOPLE)

        assert watchdog.stats.tracks_active == 4


class TestCooldownPruning:
    """Per-instance keys mean the gate's map grows with track churn."""

    def test_map_stays_bounded_under_churn(self):
        gate = CooldownGate(cooldown_seconds=0.01)
        for i in range(5000):
            gate.should_trigger(f"person#{i}")

        assert gate.tracked_keys < 1000, (
            f"Cooldown map leaked to {gate.tracked_keys} keys"
        )

    def test_realistic_churn_never_resurrects_a_live_key(self):
        """Below the hard cap, pruning is exact — it drops only dead entries."""
        gate = CooldownGate(cooldown_seconds=60.0)
        gate.should_trigger("person#live")

        # Well past the prune threshold, well short of the hard cap. This is
        # the range any real scene occupies.
        for i in range(2000):
            gate.should_trigger(f"person#{i}")

        assert gate.should_trigger("person#live") is False

    def test_hard_cap_degrades_by_forgetting_the_nearest_to_expiry(self):
        """Past the hard cap something must give; document which.

        Retaining every key would be an unbounded leak, so the cap drops the
        oldest entries — the ones with the least cooldown left to run, whose
        loss causes the least premature re-triggering. Reaching this at all
        means over 4096 distinct instances inside one cooldown window, which is
        two orders of magnitude beyond what the VLM can judge, so the practical
        consequence is a few redundant inferences, not missed alerts.
        """
        gate = CooldownGate(cooldown_seconds=1e6)
        gate.should_trigger("person#oldest")
        for i in range(5000):
            gate.should_trigger(f"person#{i}")

        assert gate.tracked_keys <= 4096
        # The oldest was sacrificed; the most recent are all still suppressing.
        assert gate.should_trigger("person#4999") is False

    def test_bounded_even_when_nothing_ages_out(self):
        """A very long cooldown means age-based pruning alone cannot bound it."""
        gate = CooldownGate(cooldown_seconds=1e6)
        for i in range(20000):
            gate.should_trigger(f"person#{i}")

        # Both halves matter. The upper bound is the leak guard; the lower one
        # is what stops this passing against a gate that suppressed every key
        # and therefore stored none.
        assert 4096 >= gate.tracked_keys > _COOLDOWN_PRUNE_AT // 2

    def test_is_known_reports_membership(self):
        gate = CooldownGate(cooldown_seconds=60.0)
        assert gate.is_known("person#1") is False
        gate.should_trigger("person#1")
        assert gate.is_known("person#1") is True


class TestWatchdogLoopPacing:
    """The loop must hit its target rate and never copy a frame it has seen."""

    class _Buffer:
        """Hands out a new frame on every read, or the same one forever."""

        def __init__(self, fresh: bool) -> None:
            self._fresh = fresh
            self._ts = 1.0
            self.reads = 0

        def latest_timestamp(self) -> float:
            if self._fresh:
                self._ts += 1.0
            return self._ts

        def read_latest(self):
            self.reads += 1
            return np.zeros((8, 8, 3), dtype=np.uint8), self._ts, 0

    def _run(self, watchdog, seconds: float, work_s: float = 0.0) -> list[float]:
        import threading

        started: list[float] = []

        def process(_frame, _ts, _idx):
            started.append(time.monotonic())
            if work_s:
                time.sleep(work_s)

        watchdog._load_model = lambda: True
        watchdog._process_frame = process
        threading.Timer(seconds, watchdog._shutdown_event.set).start()
        watchdog._watchdog_loop()
        return started

    def test_inference_time_does_not_stretch_the_interval(self):
        """20 FPS with 30 ms of work per frame is still 20 FPS.

        Pacing from the end of the previous cycle made each one last
        interval + inference — 80 ms here, 12.5 FPS.
        """
        watchdog = YOLOWatchdog(
            buffer=self._Buffer(fresh=True), publisher=MagicMock(), target_fps=20.0
        )
        started = self._run(watchdog, seconds=1.0, work_s=0.03)

        assert len(started) >= 17, f"ran at {len(started)} FPS against a target of 20"

    def test_a_repeated_frame_is_not_copied(self):
        buffer = self._Buffer(fresh=False)
        watchdog = YOLOWatchdog(buffer=buffer, publisher=MagicMock(), target_fps=50.0)

        started = self._run(watchdog, seconds=0.3)

        assert len(started) == 1
        assert buffer.reads == 1, "a frame already processed was copied again"
        assert watchdog.stats.frames_repeated > 0


class TestRuntimeReconfigurationMidFrame:
    """sync_rules runs on another thread; one frame sees one configuration."""

    def test_label_scope_swapped_mid_frame_does_not_split_the_frame(self):
        watchdog = YOLOWatchdog(
            buffer=MagicMock(), publisher=MagicMock(), target_labels={"person"}
        )
        watchdog._model = MagicMock()
        watchdog._model.predict.return_value = [
            _stub_result(
                [
                    ("person", 0.9, [0.0, 0.0, 40.0, 80.0]),
                    ("person", 0.9, [60.0, 0.0, 99.0, 80.0]),
                ]
            )
        ]

        roi = MagicMock()

        def swap_scope(_bbox):
            watchdog.update_target_labels(set())  # a rule was disabled
            return True

        roi.contains_bbox_center.side_effect = swap_scope
        watchdog._roi = roi

        watchdog._process_frame(np.zeros((100, 100, 3), dtype=np.uint8), 1.0, 0)

        assert watchdog.stats.detections_published == 2
        assert watchdog.stats.detections_filtered_label == 0

    def test_overlay_history_covers_a_busy_second(self):
        """Twenty boxes a frame for a second at 30 FPS all stay drawable."""
        watchdog = YOLOWatchdog(buffer=MagicMock(), publisher=MagicMock(), target_fps=30.0)
        watchdog._model = MagicMock()
        boxes = [("person", 0.9, [i * 4.0, 0.0, i * 4.0 + 3.0, 3.0]) for i in range(20)]
        watchdog._model.predict.return_value = [_stub_result(boxes)]

        for _ in range(30):
            watchdog._process_frame(np.zeros((100, 100, 3), dtype=np.uint8), 1.0, 0)

        assert len(watchdog.recent_detections(max_age_s=60)) == 600
