"""Tests for Stage 2 VLM Worker and Binary Verdict Parser.

Test groups:
    - BinaryVerdictParser: first-word detection, keyword scoring, negation,
      edge cases, confidence-scored variant
    - crop_detection: padding, clamping, edge cases
    - VLMWorker unit tests: initialization, rule management, staleness,
      statistics (using mocked VLM model)
    - MoondreamVLM: initialization, repr (without loading the actual model)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.pipelines.vlm_judge import BinaryVerdictParser
from app.pipelines.vlm_worker import (
    MoondreamVLM,
    VLMWorker,
    crop_detection,
)
from app.schemas.events import BoundingBox


# ═══════════════════════════════════════════════════════════════════
# Binary Verdict Parser Tests
# ═══════════════════════════════════════════════════════════════════


class TestBinaryVerdictParser:
    """Test the multi-strategy verdict parsing."""

    # ── Strategy 1: First word ───────────────────────────────────

    def test_yes_first_word(self):
        """Responses starting with 'Yes' should return True."""
        assert BinaryVerdictParser.parse("Yes, there is a pallet blocking the exit.") is True

    def test_no_first_word(self):
        """Responses starting with 'No' should return False."""
        assert BinaryVerdictParser.parse("No, the fire exit appears to be clear.") is False

    def test_yeah_first_word(self):
        """Informal 'Yeah' should be recognized."""
        assert BinaryVerdictParser.parse("Yeah, I can see a person without a hard hat.") is True

    def test_nope_first_word(self):
        """Informal 'Nope' should be recognized."""
        assert BinaryVerdictParser.parse("Nope, everything looks compliant.") is False

    def test_certainly_first_word(self):
        """'Certainly' as first word should indicate yes."""
        assert BinaryVerdictParser.parse("Certainly, there is an unauthorized person.") is True

    # ── Strategy 2: Keyword scoring ──────────────────────────────

    def test_affirmative_keywords(self):
        """Responses with more affirmative keywords should return True."""
        assert BinaryVerdictParser.parse(
            "The area is obstructed and blocked by a detected violation."
        ) is True

    def test_negative_keywords(self):
        """Responses with more negative keywords should return False."""
        assert BinaryVerdictParser.parse(
            "The exit is clear, unblocked, and accessible. Nothing was detected."
        ) is False

    def test_keyword_tie_defaults_false(self):
        """When keyword counts are tied, should default to False."""
        # Mix of positive and negative keywords
        result = BinaryVerdictParser.parse(
            "It might be something but also nothing specific."
        )
        assert result is False

    # ── Edge cases ───────────────────────────────────────────────

    def test_empty_response(self):
        """Empty response should return False."""
        assert BinaryVerdictParser.parse("") is False

    def test_whitespace_only(self):
        """Whitespace-only response should return False."""
        assert BinaryVerdictParser.parse("   \n  \t  ") is False

    def test_case_insensitive(self):
        """Parsing should be case-insensitive."""
        assert BinaryVerdictParser.parse("YES! DEFINITELY!") is True
        assert BinaryVerdictParser.parse("NO, absolutely not.") is False

    def test_punctuated_first_word(self):
        """First word with punctuation should be handled."""
        assert BinaryVerdictParser.parse("Yes. There is a clear violation.") is True
        assert BinaryVerdictParser.parse("No! The area is clear.") is False

    # ── Realistic VLM responses ──────────────────────────────────

    def test_realistic_positive_pallet(self):
        """Realistic Moondream2 response about pallet blocking exit."""
        response = (
            "Yes, there is a large wooden pallet directly in front of "
            "the emergency exit door, partially blocking access."
        )
        assert BinaryVerdictParser.parse(response) is True

    def test_realistic_negative_clear(self):
        """Realistic Moondream2 response about clear exit."""
        response = (
            "No, the fire exit is clear and unobstructed. The hallway "
            "leading to the door is empty and accessible."
        )
        assert BinaryVerdictParser.parse(response) is False

    def test_realistic_hardhat_violation(self):
        """Realistic response about hard hat violation."""
        response = "Yes, the person in the image is not wearing a hard hat."
        assert BinaryVerdictParser.parse(response) is True

    def test_realistic_hardhat_compliant(self):
        """Realistic response about hard hat compliance."""
        response = "No, the person is wearing a yellow hard hat correctly."
        assert BinaryVerdictParser.parse(response) is False

    # ── Confidence-scored variant ────────────────────────────────

    def test_parse_with_confidence_yes(self):
        """parse_with_confidence should return high confidence for clear yes."""
        verdict, conf, strategy = BinaryVerdictParser.parse_with_confidence(
            "Yes, there is a violation."
        )
        assert verdict is True
        assert conf >= 0.9
        assert strategy == "first_word_yes"

    def test_parse_with_confidence_no(self):
        """parse_with_confidence should return high confidence for clear no."""
        verdict, conf, strategy = BinaryVerdictParser.parse_with_confidence(
            "No, everything is fine."
        )
        assert verdict is False
        assert conf >= 0.9
        assert strategy == "first_word_no"

    def test_parse_with_confidence_ambiguous(self):
        """parse_with_confidence should return low confidence for ambiguous."""
        verdict, conf, strategy = BinaryVerdictParser.parse_with_confidence(
            "It is difficult to determine from this angle."
        )
        assert conf < 0.5  # Low confidence
        assert strategy == "default_conservative"

    def test_parse_with_confidence_empty(self):
        """parse_with_confidence on empty input should return zero confidence."""
        verdict, conf, strategy = BinaryVerdictParser.parse_with_confidence("")
        assert verdict is False
        assert conf == 0.0
        assert strategy == "empty_response"


# ═══════════════════════════════════════════════════════════════════
# Frame Cropping Tests
# ═══════════════════════════════════════════════════════════════════


class TestCropDetection:
    """Test bounding box cropping with padding."""

    def test_basic_crop(self):
        """Basic crop should return a sub-region of the frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)

        cropped = crop_detection(frame, bbox, padding_ratio=0.0)

        # Without padding, crop should be ~50% of frame dimensions
        assert cropped.shape[0] == 240  # 480 * 0.5
        assert cropped.shape[1] == 320  # 640 * 0.5
        assert cropped.shape[2] == 3

    def test_crop_with_padding(self):
        """Crop with padding should be larger than without."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)

        no_pad = crop_detection(frame, bbox, padding_ratio=0.0)
        with_pad = crop_detection(frame, bbox, padding_ratio=0.15)

        assert with_pad.shape[0] > no_pad.shape[0]
        assert with_pad.shape[1] > no_pad.shape[1]

    def test_crop_clamped_to_frame(self):
        """Padding near frame edges should be clamped."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Box at top-left corner
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.3, y2=0.3)

        cropped = crop_detection(frame, bbox, padding_ratio=0.5)

        # Should not exceed frame bounds
        assert cropped.shape[0] <= 480
        assert cropped.shape[1] <= 640

    def test_crop_large_bbox_returns_full_frame(self):
        """Bbox covering >80% of frame should return the full frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.95)

        cropped = crop_detection(frame, bbox)

        # Should return full frame copy
        assert cropped.shape == frame.shape

    def test_crop_returns_copy(self):
        """Cropped result should be a copy, not a view."""
        frame = np.full((480, 640, 3), 42, dtype=np.uint8)
        bbox = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)

        cropped = crop_detection(frame, bbox)
        cropped[:] = 0

        # Original frame should be unmodified
        assert frame[240, 320, 0] == 42


# ═══════════════════════════════════════════════════════════════════
# MoondreamVLM Tests (without loading the model)
# ═══════════════════════════════════════════════════════════════════


class TestMoondreamVLM:
    """Test VLM wrapper initialization and state management."""

    def test_initialization(self):
        """VLM should initialize without loading the model."""
        vlm = MoondreamVLM(device="cpu")
        assert not vlm.is_loaded
        assert vlm.device == "cpu"

    def test_query_before_load_raises(self):
        """Querying before loading should raise RuntimeError."""
        vlm = MoondreamVLM(device="cpu")
        from PIL import Image
        dummy_image = Image.new("RGB", (64, 64))

        with pytest.raises(RuntimeError, match="not loaded"):
            vlm.query(dummy_image, "Is there a person?")

    def test_repr(self):
        """__repr__ should be informative."""
        vlm = MoondreamVLM(device="mps", model_id="vikhyatk/moondream2")
        r = repr(vlm)
        assert "MoondreamVLM" in r
        assert "moondream2" in r
        assert "unloaded" in r


# ═══════════════════════════════════════════════════════════════════
# VLM Worker Unit Tests (mocked model)
# ═══════════════════════════════════════════════════════════════════


class TestVLMWorkerUnit:
    """Unit tests for VLMWorker using mocked components."""

    def _make_mock_buffer(self) -> MagicMock:
        """Create a mock buffer that returns a synthetic frame."""
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())
        return buffer

    def test_worker_initialization(self):
        """Worker should initialize without starting."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")

        assert not worker.is_running
        assert worker._triggers_received == 0

    def test_worker_stats_initial(self):
        """Initial stats should all be zero."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")
        stats = worker.stats

        assert stats.is_running is False
        assert stats.triggers_received == 0
        assert stats.triggers_processed == 0
        assert stats.verdicts_positive == 0
        assert stats.verdicts_negative == 0

    def test_add_rule(self):
        """Adding a rule should register it for the specified label."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")

        worker.add_rule(
            label="person",
            prompt="Is the person wearing a hard hat?",
            rule_id=1,
            rule_name="Hard hat check",
        )

        assert "person" in worker._rules
        assert len(worker._rules["person"]) == 1
        assert worker._rules["person"][0]["prompt"] == "Is the person wearing a hard hat?"

    def test_add_multiple_rules_same_label(self):
        """Multiple rules for the same label should all be registered."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")

        worker.add_rule("person", "Is the person wearing a hard hat?", rule_id=1)
        worker.add_rule("person", "Is the person in a restricted area?", rule_id=2)

        assert len(worker._rules["person"]) == 2

    def test_clear_rules(self):
        """Clearing rules should remove all registered rules."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")

        worker.add_rule("person", "Test prompt", rule_id=1)
        worker.add_rule("car", "Another prompt", rule_id=2)
        worker.clear_rules()

        assert len(worker._rules) == 0

    def test_staleness_detection(self):
        """Stale triggers should be discarded."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(
            buffer=buffer,
            device="cpu",
            staleness_threshold_s=2.0,
        )

        # Create a trigger that's 10 seconds old
        stale_payload = {
            "timestamp": time.monotonic() - 10.0,
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
            "frame_index": 0,
        }

        worker._process_trigger(stale_payload)

        assert worker._triggers_stale == 1
        assert worker._triggers_processed == 0

    def test_worker_repr(self):
        """__repr__ should be informative."""
        buffer = self._make_mock_buffer()
        worker = VLMWorker(buffer=buffer, device="cpu")
        r = repr(worker)
        assert "VLMWorker" in r
        assert "stopped" in r


class TestVLMWorkerLive:
    """Live integration tests that actually load the VLM model.

    These tests download ~3.7GB on first run and take 30-60s.
    Skip with: ``pytest -m 'not slow'``
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
            buffer_id=f"vlm_test_{int(time.monotonic() * 1000)}",
        )
        yield buf
        buf.close()
        buf.unlink()

    @pytest.mark.slow
    def test_vlm_load_and_query(self):
        """VLM should load and answer a basic question."""
        from PIL import Image

        vlm = MoondreamVLM(device="cpu")
        vlm.load()
        assert vlm.is_loaded

        # Create a simple test image (red square)
        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        answer = vlm.query(img, "What color is this image?")

        assert isinstance(answer, str)
        assert len(answer) > 0

        vlm.unload()
        assert not vlm.is_loaded
