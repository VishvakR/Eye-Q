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
from app.pipelines.detector import CooldownGate, YOLOWatchdog
from app.schemas.events import BoundingBox

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
        """Build a stub ultralytics result with the given (label, conf) pairs."""
        names, boxes = {}, []
        for i, (label, conf) in enumerate(detections):
            names[i] = label
            box = MagicMock()
            box.cls = [MagicMock(item=lambda i=i: i)]
            box.conf = [MagicMock(item=lambda c=conf: c)]
            box.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 50.0, 50.0])]
            boxes.append(box)

        result = MagicMock()
        result.names = names
        result.boxes = boxes
        return result

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
