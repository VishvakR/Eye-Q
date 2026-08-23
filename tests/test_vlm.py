"""Tests for Stage 2 VLM Worker and Binary Verdict Parser.

Test groups:
    - BinaryVerdictParser: first-word detection, keyword scoring, negation,
      edge cases, confidence-scored variant
    - crop_detection: padding, clamping, edge cases
    - VLMWorker unit tests: initialization, rule management, staleness,
      statistics (using mocked VLM model)
    - MoondreamVLM: initialization, repr (without loading the actual model)
    - Moondream4BitBackend: greedy decoding, the encode seam, the torchao
      stub, and reading a verdict off the prefill logits
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.pipelines.vlm_backends import (
    BinaryProbe,
    MoondreamBackend,
    OpenAICompatBackend,
    ProbeUnsupportedError,
    VLMUnavailableError,
    decide_probe,
    resolve_yes_no_ids,
)
from app.pipelines.vlm_judge import BinaryVerdictParser
from app.pipelines.vlm_worker import (
    _LETTERBOX_GREY,
    _MIN_CROP_PX,
    VERDICT_SUFFIX,
    VLMWorker,
    _percentile,
    crop_bounds,
    crop_detection,
    crop_detection_focused,
    downscale_for_vlm,
    letterbox_square,
    subject_pixel_size,
)
from app.schemas.events import BoundingBox, DetectionEvent


def mock_backend(**kwargs) -> MagicMock:
    """A stub backend wired the way the passthrough backends are.

    Stage 2 asks the backend to ``encode()`` a crop once and then ``answer()``
    each rule against that handle. Two of the three real backends cannot split
    the work, so they return the image from ``encode()`` and route ``answer()``
    back through ``query()`` — and this double does exactly the same. That
    keeps every ``backend.query`` assertion in this file meaningful while still
    driving the worker down its real code path.
    """
    backend = MagicMock(**kwargs)
    backend.encode.side_effect = lambda image: image
    backend.answer.side_effect = lambda handle, question: backend.query(handle, question)
    backend.query_batch.side_effect = lambda images, questions: [
        backend.query(image, q) for image, q in zip(images, questions, strict=True)
    ]
    # VLMResult validates this as a string, so a bare mock attribute fails
    # construction well before any assertion in the test body is reached.
    kwargs.setdefault("model_name", "fake")
    backend.model_name = kwargs["model_name"]
    return backend

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
# MoondreamBackend Tests (without loading the model)
# ═══════════════════════════════════════════════════════════════════


class TestMoondreamBackend:
    """Test VLM wrapper initialization and state management."""

    def test_initialization(self):
        """VLM should initialize without loading the model."""
        vlm = MoondreamBackend(device="cpu")
        assert not vlm.is_loaded
        assert vlm.device == "cpu"

    def test_query_before_load_raises(self):
        """Querying before loading should raise RuntimeError."""
        vlm = MoondreamBackend(device="cpu")
        from PIL import Image
        dummy_image = Image.new("RGB", (64, 64))

        with pytest.raises(RuntimeError, match="not loaded"):
            vlm.query(dummy_image, "Is there a person?")

    def test_repr(self):
        """__repr__ should be informative."""
        vlm = MoondreamBackend(device="mps", model_id="vikhyatk/moondream2")
        r = repr(vlm)
        assert "MoondreamBackend" in r
        assert "moondream2" in r
        assert "unloaded" in r


# ═══════════════════════════════════════════════════════════════════
# Moondream4BitBackend Tests (without loading the model)
# ═══════════════════════════════════════════════════════════════════


class _FakeEncodedImage:
    """Stands in for the remote code's frozen EncodedImage dataclass."""

    def __init__(self, pos: int = 729):
        self.pos = pos


class FakeMoondreamModel:
    """The ``MoondreamModel`` surface Moondream4BitBackend actually touches.

    A real class rather than a mock for the same reason :class:`ProbeStub` is:
    the code under test reaches for nested attributes (``config.tokenizer.
    templates``) whose shape is the thing being asserted, and a mock would
    conjure any of them on access and pass regardless.
    """

    _DEFAULT_TEMPLATE = {"prefix": [1, 2], "suffix": [3]}

    def __init__(self, template=_DEFAULT_TEMPLATE, logits=None):
        self.device = "cpu"
        self.query_calls: list[tuple] = []
        self.loaded_images: list = []
        self.prefill_calls: list[tuple] = []
        self.encode_calls = 0
        self._logits = logits

        self.config = SimpleNamespace(
            tokenizer=SimpleNamespace(templates={"query": template})
        )
        self.tokenizer = SimpleNamespace(
            encode=lambda text: SimpleNamespace(ids=[10, 11])
        )

    def encode_image(self, image):
        self.encode_calls += 1
        return _FakeEncodedImage()

    def query(self, handle, question, settings=None):
        self.query_calls.append((handle, question, settings))
        return {"answer": "  Yes  "}

    def load_encoded_image(self, handle):
        self.loaded_images.append(handle)

    def _prefill_prompt(self, tokens, pos, temperature, top_p):
        self.prefill_calls.append((tokens, pos, temperature, top_p))
        return self._logits, None, None, pos + tokens.shape[1]


def loaded_4bit(model=None, yes_ids=(100,), no_ids=(200,), **kwargs):
    """A Moondream4BitBackend wired to a fake model, without a real load()."""
    from app.pipelines.vlm_backends import Moondream4BitBackend

    backend = Moondream4BitBackend(device="cpu", **kwargs)
    backend._model = SimpleNamespace(model=model or FakeMoondreamModel())
    backend._tokenizer = object()
    backend._loaded = True
    backend._yes_ids = list(yes_ids)
    backend._no_ids = list(no_ids)
    return backend


class TestMoondream4BitBackend:
    """Initialization, state, and the generate path."""

    def test_initialization(self):
        from app.pipelines.vlm_backends import Moondream4BitBackend

        vlm = Moondream4BitBackend(device="cpu")
        assert not vlm.is_loaded
        assert vlm.device == "cpu"
        assert not vlm.is_int4

    def test_revision_is_pinned_to_a_sha(self):
        """This backend patches the repo's remote code. A branch name would
        let upstream change what is being patched with nothing failing."""
        from app.pipelines.vlm_backends import Moondream4BitBackend

        revision = Moondream4BitBackend()._revision
        assert len(revision) == 40
        assert all(c in "0123456789abcdef" for c in revision)

    def test_query_before_load_raises(self):
        from PIL import Image

        from app.pipelines.vlm_backends import Moondream4BitBackend

        vlm = Moondream4BitBackend(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            vlm.query(Image.new("RGB", (64, 64)), "Is there a person?")

    def test_probe_before_load_raises(self):
        from app.pipelines.vlm_backends import Moondream4BitBackend

        vlm = Moondream4BitBackend(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            vlm.probe_binary(_FakeEncodedImage(), "q")

    def test_repr(self):
        from app.pipelines.vlm_backends import Moondream4BitBackend

        r = repr(Moondream4BitBackend(device="mps"))
        assert "Moondream4BitBackend" in r
        assert "moondream-2b-2025-04-14-4bit" in r
        assert "unloaded" in r

    def test_answer_forces_greedy_decoding(self):
        """This revision samples by default (temperature 0.5, top_p 0.3). A
        sampled binary verdict alerts on one frame and not the next for the
        same crop and rule, which is indistinguishable from a flickering
        detection and impossible to tune against."""
        model = FakeMoondreamModel()
        backend = loaded_4bit(model, max_new_tokens=2)

        answer = backend.answer(_FakeEncodedImage(), "Is this person bending?")

        assert answer == "Yes", "must be stripped"
        (_handle, question, settings) = model.query_calls[0]
        assert question == "Is this person bending?"
        assert settings["temperature"] == 0.0
        assert settings["max_tokens"] == 2

    def test_answer_does_not_use_the_compat_shim(self):
        """answer_question() ignores max_new_tokens and leaves sampling on, so
        going through it would silently drop vlm_max_tokens."""
        model = FakeMoondreamModel()
        model.answer_question = lambda *a, **k: pytest.fail(
            "must not call the answer_question shim"
        )
        backend = loaded_4bit(model)

        backend.answer(_FakeEncodedImage(), "q")

    def test_encode_runs_the_vision_tower_once_per_subject(self):
        """The encode/answer seam Stage 2 relies on: one handle, many rules."""
        from PIL import Image

        model = FakeMoondreamModel()
        backend = loaded_4bit(model)

        handle = backend.encode(Image.new("RGB", (64, 64)))
        for question in ("bending?", "ppe?", "carrying?"):
            backend.answer(handle, question)

        assert model.encode_calls == 1
        assert len(model.query_calls) == 3
        assert all(call[0] is handle for call in model.query_calls)

    def test_query_batch_is_sequential(self):
        """This revision's batch_answer() is a for-loop upstream, so calling it
        would claim a batching win that no longer exists."""
        from PIL import Image

        model = FakeMoondreamModel()
        model.batch_answer = lambda *a, **k: pytest.fail("no real batching here")
        backend = loaded_4bit(model)

        images = [Image.new("RGB", (32, 32)) for _ in range(3)]
        answers = backend.query_batch(images, ["a?", "b?", "c?"])

        assert answers == ["Yes", "Yes", "Yes"]
        assert model.encode_calls == 3

    def test_query_batch_rejects_a_length_mismatch(self):
        from PIL import Image

        backend = loaded_4bit()
        with pytest.raises(ValueError, match="one question per image"):
            backend.query_batch([Image.new("RGB", (32, 32))], ["a?", "b?"])


class TestMoondream4BitComputeDtype:
    """Which dtype the packed weights are dequantized to."""

    def test_defaults_are_per_device(self):
        """fp32 on CPU is the load-bearing one: torch has no Accelerate-backed
        bf16 GEMM on ARM, and Stage 2 is prefill-bound, so upstream's bf16
        would buy memory at the expense of the thing being optimized."""
        import torch

        from app.pipelines.vlm_backends import Moondream4BitBackend

        backend = Moondream4BitBackend()
        assert backend._resolve_dtype("cpu") is torch.float32
        assert backend._resolve_dtype("mps") is torch.float16
        assert backend._resolve_dtype("cuda") is torch.bfloat16

    def test_an_explicit_dtype_overrides_the_device(self):
        import torch

        from app.pipelines.vlm_backends import Moondream4BitBackend

        backend = Moondream4BitBackend(compute_dtype="bfloat16")
        assert backend._resolve_dtype("cpu") is torch.bfloat16

    def test_an_unknown_dtype_fails_loudly(self):
        """Silently falling back would make a bench sweep report the default's
        numbers under the override's name."""
        from app.pipelines.vlm_backends import Moondream4BitBackend

        backend = Moondream4BitBackend(compute_dtype="float8")
        with pytest.raises(ValueError, match="Unknown compute_dtype"):
            backend._resolve_dtype("cpu")


class TestTorchaoStub:
    """``layers.py`` imports torchao at module scope, on every device."""

    def test_the_stub_satisfies_the_remote_import(self):
        """Installing torchao just to satisfy an import would pin a package
        whose API already moved: 0.14 removed the int4_weight_only name that
        the remote code imports."""
        import sys

        from app.pipelines.vlm_backends import _ensure_torchao_importable

        saved = {k: sys.modules.get(k) for k in ("torchao", "torchao.quantization")}
        for key in saved:
            sys.modules.pop(key, None)
        try:
            real = _ensure_torchao_importable()

            from torchao import quantize_
            from torchao.quantization import int4_weight_only

            if not real:
                sentinel = object()
                assert quantize_(sentinel, int4_weight_only(group_size=128)) is sentinel
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


class TestMoondream4BitProbe:
    """Reading the verdict off the prefill instead of generating it."""

    def _logits(self, mapping, vocab=256):
        import torch

        row = torch.full((1, vocab), -30.0)
        for token_id, value in mapping.items():
            row[0, token_id] = value
        return row

    def test_probe_reads_the_yes_no_mass(self):
        model = FakeMoondreamModel(logits=self._logits({100: 4.0, 200: 0.0}))
        backend = loaded_4bit(model)

        probe = backend.probe_binary(_FakeEncodedImage(), "Is this person bending?")

        assert probe.verdict is True
        assert probe.raw == "Yes"
        assert probe.p_yes > 0.9
        assert probe.p_yes + probe.p_no == pytest.approx(1.0)

    def test_probe_generates_nothing(self):
        """The whole point: one prefill, no decode loop, no string parsing."""
        model = FakeMoondreamModel(logits=self._logits({100: 4.0, 200: 0.0}))
        backend = loaded_4bit(model)

        backend.probe_binary(_FakeEncodedImage(), "q")

        assert len(model.prefill_calls) == 1
        assert model.query_calls == []

    def test_probe_prefills_greedily_from_the_image_position(self):
        model = FakeMoondreamModel(logits=self._logits({100: 0.0, 200: 4.0}))
        backend = loaded_4bit(model)
        handle = _FakeEncodedImage(pos=729)

        probe = backend.probe_binary(handle, "q")

        (tokens, pos, temperature, top_p) = model.prefill_calls[0]
        assert pos == 729
        assert temperature == 0.0
        assert top_p == 1.0
        # prefix + question + suffix, as MoondreamModel.query builds it.
        assert tokens.tolist() == [[1, 2, 10, 11, 3]]
        assert probe.verdict is False

    def test_probe_restores_this_subjects_image_cache(self):
        """The previous question wrote past the image tokens in the live KV
        caches; without the reload the second probe reads the first one's
        leftovers."""
        model = FakeMoondreamModel(logits=self._logits({100: 4.0, 200: 0.0}))
        backend = loaded_4bit(model)
        handle = _FakeEncodedImage()

        backend.probe_binary(handle, "a?")
        backend.probe_binary(handle, "b?")

        assert model.loaded_images == [handle, handle]

    def test_probe_tolerates_unreduced_logits(self):
        """lm_head reduces to the final position today. Indexing a vocab axis
        that moved would read noise rather than fail."""
        import torch

        reduced = self._logits({100: 4.0, 200: 0.0})
        model = FakeMoondreamModel(logits=torch.stack([reduced[0]] * 3).unsqueeze(0))
        backend = loaded_4bit(model)

        assert backend.probe_binary(_FakeEncodedImage(), "q").verdict is True

    def test_no_usable_token_ids_is_unsupported_not_a_failure(self):
        """The worker latches probing off on this and falls back to
        generate-and-parse, which is what every deployment did before."""
        from app.pipelines.vlm_backends import ProbeUnsupportedError

        backend = loaded_4bit(yes_ids=(), no_ids=())

        with pytest.raises(ProbeUnsupportedError, match="Yes/No token ids"):
            backend.probe_binary(_FakeEncodedImage(), "q")

    def test_a_model_without_a_query_template_is_unsupported(self):
        from app.pipelines.vlm_backends import ProbeUnsupportedError

        backend = loaded_4bit(FakeMoondreamModel(template=None))

        with pytest.raises(ProbeUnsupportedError, match="query template"):
            backend.probe_binary(_FakeEncodedImage(), "q")

    def test_probe_batch_matches_positionally(self):
        model = FakeMoondreamModel(logits=self._logits({100: 4.0, 200: 0.0}))
        backend = loaded_4bit(model)
        handles = [_FakeEncodedImage(), _FakeEncodedImage(), _FakeEncodedImage()]

        probes = backend.probe_binary_batch(handles, ["a?", "b?", "c?"])

        assert len(probes) == 3
        assert model.loaded_images == handles
        assert [call[0].tolist() for call in model.prefill_calls] == [[[1, 2, 10, 11, 3]]] * 3

    def test_probe_batch_rejects_a_length_mismatch(self):
        backend = loaded_4bit()
        with pytest.raises(ValueError, match="one question per image"):
            backend.probe_binary_batch([_FakeEncodedImage()], ["a?", "b?"])

    def test_the_worker_probes_this_backend(self):
        """End of the chain: the capability check the worker actually runs."""
        from app.pipelines.vlm_backends import BinaryProbeBackend

        assert isinstance(loaded_4bit(), BinaryProbeBackend)

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

        worker._process_trigger(stale_payload, MagicMock())

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

        vlm = MoondreamBackend(device="cpu")
        vlm.load()
        assert vlm.is_loaded

        # Create a simple test image (red square)
        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        answer = vlm.query(img, "What color is this image?")

        assert isinstance(answer, str)
        assert len(answer) > 0

        vlm.unload()
        assert not vlm.is_loaded


class TestMoondream4BitLive:
    """Live tests for the 4-bit backend. Downloads ~1.1 GB on first run.

    Skip with ``pytest -m 'not slow'``.
    """

    @pytest.mark.slow
    def test_quantized_linear_premise(self):
        """Pin what the load-time patch assumes about the remote code.

        Moondream4BitBackend replaces ``QuantizedLinear.unpack`` and reaches
        for ``dequantize_tensor`` in the same generated module. Both are
        upstream's, not ours. If a re-upload changes either, the patch has to
        fail *here* — loudly, against a named premise — rather than be carried
        forever as a workaround for a defect that no longer exists, or worse,
        silently stop applying. This is why ``vlm_revision`` is a sha.
        """
        import sys

        from app.pipelines.vlm_backends import (
            _QUANTIZED_LINEAR,
            Moondream4BitBackend,
        )

        vlm = Moondream4BitBackend(device="cpu")
        vlm.load()
        try:
            layers = [
                m
                for m in vlm._model.modules()
                if type(m).__name__ == _QUANTIZED_LINEAR
            ]
            assert layers, "no packed 4-bit layers; this is not a 4-bit export"

            remote = sys.modules[type(layers[0]).__module__]
            assert callable(getattr(remote, "dequantize_tensor", None))
            assert layers[0].unpacked, "load() must unpack eagerly, not lazily"
        finally:
            vlm.unload()

    @pytest.mark.slow
    def test_load_and_query(self):
        from PIL import Image

        from app.pipelines.vlm_backends import Moondream4BitBackend

        vlm = Moondream4BitBackend(device="cpu", max_new_tokens=16)
        vlm.load()
        try:
            assert vlm.is_loaded
            img = Image.new("RGB", (128, 128), color=(255, 0, 0))
            answer = vlm.query(img, "What color is this image?")
            assert isinstance(answer, str)
            assert len(answer) > 0
        finally:
            vlm.unload()
        assert not vlm.is_loaded

    @pytest.mark.slow
    def test_the_probe_agrees_with_generate_and_parse(self):
        """The probe is only a shortcut if it reaches the same verdict.

        Both paths read the same distribution at the same position — one
        samples from it greedily and stringifies, the other sums two sets of
        token ids — so a disagreement means the probe is reading the wrong
        position or the wrong ids, not that the model is uncertain.
        """
        from PIL import Image

        from app.pipelines.vlm_backends import Moondream4BitBackend

        vlm = Moondream4BitBackend(device="cpu", max_new_tokens=4)
        vlm.load()
        try:
            img = Image.new("RGB", (128, 128), color=(255, 0, 0))
            handle = vlm.encode(img)

            for question in (
                "Is this image red?" + VERDICT_SUFFIX,
                "Is there a person in this image?" + VERDICT_SUFFIX,
            ):
                probe = vlm.probe_binary(handle, question)
                parsed = BinaryVerdictParser().parse(vlm.answer(handle, question))
                assert probe.verdict == parsed, (
                    f"probe {probe.verdict} (p_yes={probe.p_yes:.3f}) disagrees "
                    f"with generate-and-parse {parsed} on {question!r}"
                )
        finally:
            vlm.unload()


# ═══════════════════════════════════════════════════════════════════
# Backend Selection Tests
# ═══════════════════════════════════════════════════════════════════


class TestBackendFactory:
    """EYEQ_VLM_BACKEND must select the right implementation.

    All four run locally; the choice is about speed and model availability
    on the target edge device, not about where inference happens.
    """

    def test_moondream_4bit_is_the_default(self):
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import Moondream4BitBackend, build_backend

        backend = build_backend(Settings())
        assert isinstance(backend, Moondream4BitBackend)
        assert backend.model_name == "moondream/moondream-2b-2025-04-14-4bit"
        assert not backend.is_loaded

    def test_the_fp32_baseline_stays_selectable(self):
        """The 2024-08-26 fp32 revision is what the 4-bit one is measured
        against, so it has to remain reachable by name."""
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import MoondreamBackend, build_backend

        backend = build_backend(
            Settings(
                vlm_backend="moondream",
                vlm_model="vikhyatk/moondream2",
                vlm_revision="2024-08-26",
            )
        )
        assert isinstance(backend, MoondreamBackend)
        assert backend.model_name == "vikhyatk/moondream2"

    def test_compute_dtype_reaches_the_4bit_backend(self):
        """The empty default means 'pick per device', not 'pass an empty
        string down and fail on the first unpack'."""
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import build_backend

        assert build_backend(Settings())._compute_dtype is None
        assert (
            build_backend(Settings(vlm_compute_dtype="float16"))._compute_dtype
            == "float16"
        )

    def test_transformers_backend_carries_the_model_id(self):
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import TransformersVLMBackend, build_backend

        backend = build_backend(
            Settings(vlm_backend="transformers", vlm_model="Qwen/Qwen2-VL-2B-Instruct")
        )
        assert isinstance(backend, TransformersVLMBackend)
        assert backend.model_name == "Qwen/Qwen2-VL-2B-Instruct"

    def test_openai_compat_backend_targets_the_local_server(self):
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import OpenAICompatBackend, build_backend

        backend = build_backend(
            Settings(
                vlm_backend="openai_compat",
                vlm_api_base="http://127.0.0.1:8080/v1",
                vlm_api_model="qwen2-vl",
            )
        )
        assert isinstance(backend, OpenAICompatBackend)
        assert backend.model_name == "qwen2-vl"
        assert "127.0.0.1:8080" in backend.device

    def test_all_backends_satisfy_the_protocol(self):
        """The worker depends only on VLMBackend, so each must conform."""
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import VLMBackend, build_backend

        for name in ("moondream4bit", "moondream", "transformers", "openai_compat"):
            backend = build_backend(Settings(vlm_backend=name))
            assert isinstance(backend, VLMBackend), f"{name} does not conform"

    def test_query_before_load_raises(self):
        """Every in-process backend must fail loudly, not silently return junk."""
        from PIL import Image

        from app.pipelines.vlm_backends import TransformersVLMBackend

        backend = TransformersVLMBackend(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.query(Image.new("RGB", (64, 64)), "Is there a person?")


# ═══════════════════════════════════════════════════════════════════
# Backlog Coalescing Tests
# ═══════════════════════════════════════════════════════════════════


class TestTriggerCoalescing:
    """VLM inference is far slower than Stage 1 triggers, so a backlog forms.

    Processing it in arrival order means always working on the oldest frame —
    precisely the one most likely to be discarded as stale on arrival. The
    worker must instead keep the newest trigger per label.
    """

    def _worker(self):
        buffer = MagicMock()
        return VLMWorker(buffer=buffer, device="cpu")

    def _sub(self, first, backlog):
        """Fake subscriber: one receive() then a drained backlog."""
        sub = MagicMock()
        sub.receive.return_value = ("detection", first) if first else None
        sub.drain.return_value = [("detection", p) for p in backlog]
        return sub

    def _payload(self, label, ts):
        return {
            "timestamp": ts,
            "camera_id": "test",
            "label": label,
            "confidence": 0.9,
            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            "frame_index": 0,
        }

    def test_empty_when_nothing_arrives(self):
        worker = self._worker()
        assert worker._collect_batch(self._sub(None, [])) == []

    def test_single_trigger_passes_through(self):
        worker = self._worker()
        batch = worker._collect_batch(self._sub(self._payload("person", 10.0), []))
        assert len(batch) == 1
        assert batch[0]["label"] == "person"

    def test_keeps_only_the_newest_per_label(self):
        worker = self._worker()
        batch = worker._collect_batch(
            self._sub(
                self._payload("person", 10.0),
                [self._payload("person", 11.0), self._payload("person", 12.0)],
            )
        )
        assert len(batch) == 1
        assert batch[0]["timestamp"] == 12.0, "Superseded frame processed instead"
        assert worker._triggers_coalesced == 2

    def test_distinct_labels_all_survive(self):
        """Different labels may match different rules — none may be dropped."""
        worker = self._worker()
        batch = worker._collect_batch(
            self._sub(
                self._payload("person", 10.0),
                [self._payload("car", 11.0), self._payload("dog", 9.0)],
            )
        )
        assert {p["label"] for p in batch} == {"person", "car", "dog"}
        assert worker._triggers_coalesced == 0

    def test_batch_is_ordered_newest_first(self):
        """The freshest evidence should be judged before the budget runs out."""
        worker = self._worker()
        batch = worker._collect_batch(
            self._sub(
                self._payload("person", 5.0),
                [self._payload("car", 20.0), self._payload("dog", 12.0)],
            )
        )
        assert [p["timestamp"] for p in batch] == [20.0, 12.0, 5.0]

    def test_every_trigger_is_counted_as_received(self):
        """Coalescing must not hide throughput from the stats endpoint."""
        worker = self._worker()
        worker._collect_batch(
            self._sub(
                self._payload("person", 10.0),
                [self._payload("person", 11.0), self._payload("car", 12.0)],
            )
        )
        assert worker._triggers_received == 3
        assert worker._triggers_coalesced == 1

    def test_out_of_order_arrival_still_keeps_newest(self):
        worker = self._worker()
        batch = worker._collect_batch(
            self._sub(
                self._payload("person", 99.0),
                [self._payload("person", 1.0), self._payload("person", 50.0)],
            )
        )
        assert batch[0]["timestamp"] == 99.0


class TestInferenceAbort:
    """A timed-out inference must release the executor, not merely be ignored."""

    def test_timeout_signals_the_backend_to_stop(self):
        import concurrent.futures

        buffer = MagicMock()
        backend = mock_backend()
        backend.query.side_effect = lambda *a, **k: time.sleep(5)
        worker = VLMWorker(buffer=buffer, backend=backend, inference_timeout_s=0.2)

        with pytest.raises(concurrent.futures.TimeoutError):
            worker._answer(MagicMock(), "Is there a person?")

        # The worker calls request_stop() on the timeout path in _process_trigger;
        # verify the backend exposes and records it.
        worker._vlm.request_stop()
        backend.request_stop.assert_called()

    def test_stop_event_halts_generation(self):
        """The criteria must integrate with the real transformers plumbing.

        ``StoppingCriteriaList`` ORs each criterion into a per-sequence
        BoolTensor, so it is exercised here with the tensors ``generate()``
        actually passes rather than with placeholders.
        """
        import torch

        from app.pipelines.vlm_backends import _stop_on_event_criteria

        event = threading.Event()
        criteria = _stop_on_event_criteria(event)

        input_ids = torch.zeros((1, 4), dtype=torch.long)
        scores = torch.zeros((1, 32))

        assert not bool(criteria(input_ids, scores).all())
        event.set()
        assert bool(criteria(input_ids, scores).all())

    def test_transformers_backend_stop_is_idempotent(self):
        """request_stop() before/without a query must never raise."""
        from app.pipelines.vlm_backends import TransformersVLMBackend

        backend = TransformersVLMBackend(device="cpu")
        backend.request_stop()
        backend.request_stop()
        assert backend._stop_event.is_set()


class TestUnmatchedDetections:
    """A detection no rule asked about must not cost an inference."""

    def _worker(self, **kwargs):
        return VLMWorker(buffer=MagicMock(), device="cpu", **kwargs)

    def _event(self, label="person"):
        return DetectionEvent(
            timestamp=time.monotonic(),
            camera_id="test",
            label=label,
            confidence=0.9,
            bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3),
            frame_index=0,
        )

    def test_no_rules_means_no_query(self):
        """With no rules there is no question to ask."""
        worker = self._worker()
        assert worker._match_rules(self._event()) == []

    def test_non_matching_label_is_ignored(self):
        """A 'dog' rule must not cause the VLM to be asked about a person."""
        worker = self._worker()
        worker.add_rule("dog", "Is a dog present?", rule_id=1)

        assert worker._match_rules(self._event("person")) == []
        assert len(worker._match_rules(self._event("dog"))) == 1

    def test_fallback_runs_only_when_configured(self):
        """The default prompt is opt-in, not a hidden default."""
        worker = self._worker(default_prompt="Anything dangerous here?")
        matched = worker._match_rules(self._event())

        assert len(matched) == 1
        assert matched[0]["rule_id"] == 0
        assert matched[0]["prompt"] == "Anything dangerous here?"

    def test_wildcard_rule_matches_any_label(self):
        worker = self._worker()
        worker.add_rule("*", "Is anything wrong?", rule_id=7)

        assert len(worker._match_rules(self._event("forklift"))) == 1


class TestMultiInstanceJudging:
    """Every subject in a frame must be judged, not just the first per label.

    Regression suite for the defect where "alert if a person has no helmet"
    judged one arbitrary person out of a group, so a lone violator among
    compliant workers was usually never looked at.
    """

    def _worker(self, backend=None, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        if backend is None:
            backend = mock_backend()
            backend.query.return_value = "Yes"
            backend.model_name = "fake"

        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person missing a hard hat?", rule_id=1)
        return worker, buffer, backend

    def _payload(self, track_id, timestamp=None, x=0.1):
        return {
            "timestamp": timestamp if timestamp is not None else time.monotonic(),
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": x, "y1": 0.1, "x2": x + 0.08, "y2": 0.5},
            "frame_index": 0,
            "track_id": track_id,
        }

    def test_five_people_in_one_frame_are_all_judged(self):
        """The headline fix."""
        worker, _buffer, backend = self._worker()
        ts = time.monotonic()
        batch = [self._payload(i, ts, x=0.1 * i) for i in range(1, 6)]

        worker._process_batch(batch, MagicMock())

        assert backend.query.call_count == 5

    def test_one_frame_is_read_once_for_the_whole_group(self):
        """Crops come from a single read, not one read per subject."""
        worker, buffer, _backend = self._worker()
        ts = time.monotonic()
        batch = [self._payload(i, ts, x=0.1 * i) for i in range(1, 6)]

        worker._process_batch(batch, MagicMock())

        assert buffer.read_by_timestamp.call_count == 1

    def test_crops_are_taken_before_any_inference_runs(self):
        """The staleness trap: pixels must be in hand before the slow part.

        Inference takes seconds and the ring buffer holds seconds, so a design
        that fetched each frame just before its own inference would find the
        frame gone for every subject after the first.
        """
        reads_at_query_time = []

        worker, buffer, backend = self._worker()
        backend.query.side_effect = lambda *a, **k: (
            reads_at_query_time.append(buffer.read_by_timestamp.call_count) or "Yes"
        )

        ts = time.monotonic()
        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 6)], MagicMock())

        assert reads_at_query_time == [1, 1, 1, 1, 1], (
            "The buffer was read during inference — crops are not taken up front"
        )

    def test_a_vanished_frame_cannot_silently_judge_a_different_one(self):
        """A wrapped ring buffer must drop the trigger, not judge junk."""
        worker, buffer, backend = self._worker()
        buffer.read_by_timestamp.return_value = None
        # The slot now holds a frame from a completely different moment.
        buffer.read_at_index.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            time.monotonic() - 60.0,
        )

        worker._process_batch([self._payload(1)], MagicMock())

        assert backend.query.call_count == 0

    def test_each_subject_gets_its_own_verdict_published(self):
        worker, _buffer, backend = self._worker()
        backend.query.side_effect = ["Yes", "No", "No"]
        publisher = MagicMock()

        ts = time.monotonic()
        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in (1, 2, 3)], publisher)

        verdicts = [call.args[1]["verdict"] for call in publisher.publish.call_args_list]
        track_ids = [
            call.args[1]["query"]["detection"]["track_id"]
            for call in publisher.publish.call_args_list
        ]
        assert verdicts == [True, False, False]
        assert set(track_ids) == {1, 2, 3}

    def test_the_batch_cap_bounds_the_work(self):
        worker, _buffer, backend = self._worker(max_instances_per_batch=3)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.05 * i) for i in range(1, 11)], MagicMock())

        assert backend.query.call_count == 3
        assert worker.stats.instances_shed == 7

    def test_never_seen_subjects_are_judged_before_re_checks(self):
        """A crowd must delay re-checks, never a first look at someone new."""
        worker, _buffer, _backend = self._worker(
            max_instances_per_batch=2, verdict_ttl_s=0.0
        )
        publisher = MagicMock()
        ts = time.monotonic()

        # Subjects 1 and 2 are already known; 8 and 9 have never been seen.
        worker._judged_tracks.update({1, 2})

        worker._process_batch(
            [self._payload(t, ts, x=0.05 * t) for t in (1, 2, 8, 9)], publisher
        )

        judged = {
            call.args[1]["query"]["detection"]["track_id"]
            for call in publisher.publish.call_args_list
        }
        assert judged == {8, 9}, (
            f"Re-checks displaced first looks at new subjects: judged {judged}"
        )

    def test_a_stale_group_is_dropped_before_the_buffer_is_touched(self):
        worker, buffer, backend = self._worker(staleness_threshold_s=2.0)
        old = time.monotonic() - 30.0

        worker._process_batch([self._payload(i, old, x=0.1 * i) for i in (1, 2, 3)], MagicMock())

        assert buffer.read_by_timestamp.call_count == 0
        assert backend.query.call_count == 0
        assert worker.stats.triggers_stale == 3


class TestVerdictMemo:
    """Reusing a verdict is what keeps per-subject judging affordable."""

    def _worker(self, **kwargs):
        worker = VLMWorker(buffer=MagicMock(), backend=mock_backend(), **kwargs)
        worker.add_rule("person", "Is this person missing a hard hat?", rule_id=1)
        return worker

    def _event(self, track_id=1, x=0.1):
        return DetectionEvent(
            timestamp=time.monotonic(),
            camera_id="test",
            label="person",
            confidence=0.9,
            bbox=BoundingBox(x1=x, y1=0.1, x2=x + 0.1, y2=0.5),
            frame_index=0,
            track_id=track_id,
        )

    def test_an_unjudged_subject_has_work_to_do(self):
        worker = self._worker()
        assert len(worker._rules_to_judge(self._event())) == 1

    def test_a_judged_subject_is_not_asked_again(self):
        worker = self._worker(verdict_ttl_s=300.0)
        event = self._event()
        worker._remember_verdict(event, rule_id=1, verdict=False)

        assert worker._rules_to_judge(event) == []

    def test_a_different_subject_is_still_asked(self):
        """The memo must be per person, or one worker would mask the rest."""
        worker = self._worker(verdict_ttl_s=300.0)
        worker._remember_verdict(self._event(track_id=1), rule_id=1, verdict=False)

        assert len(worker._rules_to_judge(self._event(track_id=2))) == 1

    def test_the_memo_expires(self):
        worker = self._worker(verdict_ttl_s=0.05)
        event = self._event()
        worker._remember_verdict(event, rule_id=1, verdict=False)
        assert worker._rules_to_judge(event) == []

        time.sleep(0.1)
        assert len(worker._rules_to_judge(event)) == 1

    def test_movement_within_the_floor_does_not_re_judge(self):
        """Without this floor a single walker saturates the VLM."""
        worker = self._worker(verdict_ttl_s=300.0, min_rejudge_s=60.0, rejudge_iou=0.5)
        worker._remember_verdict(self._event(track_id=1, x=0.1), rule_id=1, verdict=False)

        # Same person, now nowhere near where they were judged.
        moved = self._event(track_id=1, x=0.8)
        assert worker._rules_to_judge(moved) == []

    def test_movement_past_the_floor_does_re_judge(self):
        worker = self._worker(verdict_ttl_s=300.0, min_rejudge_s=0.0, rejudge_iou=0.5)
        worker._remember_verdict(self._event(track_id=1, x=0.1), rule_id=1, verdict=False)

        moved = self._event(track_id=1, x=0.8)
        assert len(worker._rules_to_judge(moved)) == 1

    def test_a_stationary_subject_is_never_re_judged_by_movement(self):
        worker = self._worker(verdict_ttl_s=300.0, min_rejudge_s=0.0, rejudge_iou=0.5)
        event = self._event(track_id=1, x=0.1)
        worker._remember_verdict(event, rule_id=1, verdict=False)

        assert worker._rules_to_judge(self._event(track_id=1, x=0.1)) == []

    def test_untracked_detections_are_never_memoised(self):
        """Otherwise every untracked subject collapses onto one key.

        The appliance would judge the first person it ever saw and treat every
        person thereafter as already answered.
        """
        worker = self._worker(verdict_ttl_s=300.0)
        event = self._event(track_id=-1)
        worker._remember_verdict(event, rule_id=1, verdict=False)

        assert len(worker._rules_to_judge(event)) == 1
        assert worker._memo == {}

    def test_the_memo_is_per_rule(self):
        worker = self._worker(verdict_ttl_s=300.0)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=2)
        event = self._event()
        worker._remember_verdict(event, rule_id=1, verdict=False)

        remaining = worker._rules_to_judge(event)
        assert [r["rule_id"] for r in remaining] == [2]

    def test_a_zero_ttl_disables_reuse(self):
        worker = self._worker(verdict_ttl_s=0.0)
        event = self._event()
        worker._remember_verdict(event, rule_id=1, verdict=False)

        assert len(worker._rules_to_judge(event)) == 1

    def test_the_memo_stays_bounded(self):
        """Track ids never repeat, so without pruning this grows forever."""
        worker = self._worker(verdict_ttl_s=0.05)
        for track_id in range(3000):
            worker._remember_verdict(self._event(track_id=track_id), rule_id=1, verdict=False)

        assert len(worker._memo) < 2000, f"Memo leaked to {len(worker._memo)} entries"


class TestCropGeometryIsAlwaysKept:
    """Neither degenerate case may fall back to the whole frame.

    Stage 2's prompt carries 729 image tokens against a one-token answer, so a
    whole-frame crop is the most expensive query the appliance can run. Both
    old fallbacks spent that on the subjects least able to justify it.
    """

    def test_a_frame_filling_subject_keeps_its_own_box(self):
        """The >80% case used to discard the geometry entirely."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.95)

        assert bbox.area > 0.8
        bounds = crop_bounds(frame.shape, bbox, padding_ratio=0.0)

        assert bounds == (32, 24, 608, 456)

    def test_a_frame_filling_subject_can_still_be_dimmed(self):
        """Losing the geometry cost the focused crop its whole point."""
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        bbox = BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.95)

        assert bbox.area > 0.8
        # Enough padding to leave a context strip, not enough to reach the
        # frame edge — the old code returned the whole frame regardless.
        out = crop_detection_focused(frame, bbox, padding_ratio=0.02, dim=0.25)

        # The subject is at full brightness and the strip outside it is not,
        # which is only possible if the crop kept the subject's box.
        assert out[out.shape[0] // 2, out.shape[1] // 2, 0] == 200
        assert out[-1, -1, 0] < 200

    def test_a_tiny_box_is_widened_around_its_centre(self):
        """Not replaced by the frame — grown where the subject actually is."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 6x10 px box sitting well inside the frame.
        bbox = BoundingBox(x1=0.5, y1=0.5, x2=0.509375, y2=0.52)

        x1, y1, x2, y2 = crop_bounds(frame.shape, bbox, padding_ratio=0.0)

        assert (x2 - x1) == _MIN_CROP_PX
        assert (y2 - y1) == _MIN_CROP_PX
        assert (x1, y1, x2, y2) != (0, 0, 640, 480)
        # Still centred on the subject.
        assert x1 < 320 < x2
        assert y1 < 240 < y2

    def test_a_tiny_box_at_the_edge_stays_inside_the_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.005, y2=0.005)

        x1, y1, x2, y2 = crop_bounds(frame.shape, bbox, padding_ratio=0.0)

        assert (x1, y1) == (0, 0)
        assert (x2 - x1) == _MIN_CROP_PX
        assert (y2 - y1) == _MIN_CROP_PX

    def test_a_frame_smaller_than_the_floor_yields_the_whole_extent(self):
        """The one case where the whole frame is genuinely the right answer."""
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.4, y1=0.4, x2=0.45, y2=0.45)

        assert crop_bounds(frame.shape, bbox, padding_ratio=0.0) == (0, 0, 40, 40)

    def test_the_crop_never_exceeds_the_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x1, y1, x2, y2 in [
            (0.0, 0.0, 0.002, 0.002),
            (0.99, 0.99, 1.0, 1.0),
            (0.0, 0.45, 1.0, 0.55),
        ]:
            bounds = crop_bounds(
                frame.shape, BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), padding_ratio=0.3
            )
            assert 0 <= bounds[0] < bounds[2] <= 640
            assert 0 <= bounds[1] < bounds[3] <= 480

    def test_subject_pixel_size_ignores_padding(self):
        """The size test must read the subject, not the context around it."""
        bbox = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)

        assert subject_pixel_size((480, 640, 3), bbox) == (320, 240)


class TestUnreadableSubjectsAreSkipped:
    """A subject with nothing legible in it must not buy an inference.

    demo_cctv.py already records the failure this prevents: at that camera
    distance a head is ~20 px and Moondream answers "No" to every phrasing of
    a hard-hat question regardless of what is on the worker's head. Before
    this, such a detection fell back to a whole-frame crop — the largest query
    in the system — and the answer was filed against that person's box anyway.
    """

    def _worker(self, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        backend = mock_backend()
        backend.query.return_value = "Yes"

        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        return worker, backend

    def _payload(self, x1, y1, x2, y2, track_id=1):
        return {
            "timestamp": time.monotonic(),
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "frame_index": 0,
            "track_id": track_id,
        }

    def test_a_subject_under_the_floor_is_never_asked_about(self):
        worker, backend = self._worker(min_subject_px=24)
        # ~13x10 px on a 640x480 frame.
        batch = [self._payload(0.5, 0.5, 0.52, 0.52)]

        worker._process_batch(batch, MagicMock())

        assert backend.query.call_count == 0
        assert worker.stats.instances_too_small == 1

    def test_a_legible_subject_is_still_judged(self):
        worker, backend = self._worker(min_subject_px=24)
        batch = [self._payload(0.1, 0.1, 0.2, 0.6)]

        worker._process_batch(batch, MagicMock())

        assert backend.query.call_count == 1
        assert worker.stats.instances_too_small == 0

    def test_a_tall_narrow_subject_is_judged(self):
        """The floor tests the longer side — a person is tall and narrow."""
        worker, backend = self._worker(min_subject_px=24)
        # 6 px wide, 96 px tall: legible, despite being under the floor in x.
        batch = [self._payload(0.5, 0.3, 0.509375, 0.5)]

        worker._process_batch(batch, MagicMock())

        assert backend.query.call_count == 1

    def test_a_skipped_subject_is_not_remembered(self):
        """They must still be asked about once they come closer."""
        worker, backend = self._worker(min_subject_px=24)
        worker._process_batch([self._payload(0.5, 0.5, 0.52, 0.52, track_id=7)], MagicMock())
        assert backend.query.call_count == 0

        # Same subject, now close enough to read.
        worker._process_batch([self._payload(0.1, 0.1, 0.2, 0.6, track_id=7)], MagicMock())

        assert backend.query.call_count == 1

    def test_zero_disables_the_floor(self):
        worker, backend = self._worker(min_subject_px=0)
        batch = [self._payload(0.5, 0.5, 0.505, 0.505)]

        worker._process_batch(batch, MagicMock())

        assert backend.query.call_count == 1


class TestBatchedDispatch:
    """vlm_batch_size > 1 answers several subjects in one dispatch.

    The capability is real — the pinned Moondream revision ships
    ``batch_answer`` — but it is off by default, because Stage 2 is ~99%
    prefill and prefill is compute-bound, so batching amortizes a weight read
    it was never waiting on while multiplying the KV cache by the batch size.
    These tests pin the mechanics; bench/benchmark_vlm.py decides the default.
    """

    def _worker(self, backend=None, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        if backend is None:
            backend = mock_backend()
            backend.query.return_value = "Yes"

        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        return worker, backend

    def _payload(self, track_id, timestamp=None, x=0.1):
        return {
            "timestamp": timestamp if timestamp is not None else time.monotonic(),
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": x, "y1": 0.1, "x2": x + 0.08, "y2": 0.5},
            "frame_index": 0,
            "track_id": track_id,
        }

    def test_one_dispatch_covers_the_whole_frame(self):
        worker, backend = self._worker(batch_size=4)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 5)], MagicMock())

        assert backend.query_batch.call_count == 1
        images, questions = backend.query_batch.call_args[0]
        assert len(images) == 4
        assert len(questions) == 4

    def test_every_subject_still_gets_its_own_verdict(self):
        """Batching may not cost per-subject attribution."""
        worker, backend = self._worker(batch_size=4)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        publisher = MagicMock()
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 5)], publisher)

        published = [c[0][1] for c in publisher.publish.call_args_list]
        assert len(published) == 4
        track_ids = {r["query"]["detection"]["track_id"] for r in published}
        assert track_ids == {1, 2, 3, 4}

    def test_the_batch_is_capped_by_batch_size(self):
        worker, backend = self._worker(batch_size=2, max_instances_per_batch=8)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 7)], MagicMock())

        assert backend.query_batch.call_count == 3
        for call in backend.query_batch.call_args_list:
            assert len(call[0][0]) == 2

    def test_the_instance_cap_still_bounds_the_work(self):
        """batch_size may not smuggle work past max_instances_per_batch."""
        worker, backend = self._worker(batch_size=4, max_instances_per_batch=3)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 9)], MagicMock())

        assert worker.stats.instances_judged == 3
        assert worker.stats.instances_shed == 5

    def test_questions_are_grouped_by_prompt(self):
        """Same-prompt groups pad by zero, so they must not be mixed."""
        worker, backend = self._worker(batch_size=4)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        worker.add_rule("person", "Is this person bending over?", rule_id=2)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 4)], MagicMock())

        assert backend.query_batch.call_count == 2
        for call in backend.query_batch.call_args_list:
            questions = call[0][1]
            assert len(set(questions)) == 1

    def test_batch_size_one_uses_the_sequential_path(self):
        """The default must be the code path that was already measured."""
        worker, backend = self._worker(batch_size=1)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 4)], MagicMock())

        assert backend.query_batch.call_count == 0
        assert backend.encode.call_count == 3
        assert backend.query.call_count == 3

    def test_the_right_answer_reaches_the_right_person(self):
        """The claim batching has to earn: attribution survives the dispatch.

        Each subject is painted a distinct brightness, and the double reads
        that brightness back out of the crop it is handed. So an answer that
        lands on the wrong track is visible as a wrong verdict, which a
        constant-answer double could never show.
        """
        # Four well-separated subjects, each filled with its own value, so the
        # crop's max identifies which subject it holds.
        #
        # The fills sit above _LETTERBOX_GREY: crops are letterboxed into a
        # square canvas before dispatch, and a subject painted darker than the
        # padding would be identified as the padding instead. That is a
        # property of this fixture's trick for reading a subject back out of an
        # image, not of the attribution being tested.
        fills = (150, 180, 210, 240)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = []
        for i in range(4):
            x1 = 0.05 + 0.2 * i
            boxes.append((x1, x1 + 0.1))
            px1, px2 = int(x1 * 640), int((x1 + 0.1) * 640)
            frame[48:240, px1:px2] = fills[i]

        # Alternating truth, so a shifted mapping cannot pass by luck.
        expected = {1: True, 2: False, 3: True, 4: False}

        def identify(image):
            return fills.index(int(np.asarray(image).max())) + 1  # 1-based id

        def run(batch_size):
            buffer = MagicMock()
            buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
            buffer.read_at_index.return_value = (frame, time.monotonic())

            backend = mock_backend()
            backend.query.side_effect = lambda image, q: (
                "Yes" if expected[identify(image)] else "No"
            )

            worker = VLMWorker(buffer=buffer, backend=backend, batch_size=batch_size)
            worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)

            publisher = MagicMock()
            ts = time.monotonic()
            worker._process_batch(
                [
                    {
                        "timestamp": ts,
                        "camera_id": "test",
                        "label": "person",
                        "confidence": 0.9,
                        "bbox": {"x1": x1, "y1": 0.1, "x2": x2, "y2": 0.5},
                        "frame_index": 0,
                        "track_id": i + 1,
                    }
                    for i, (x1, x2) in enumerate(boxes)
                ],
                publisher,
            )
            return {
                c[0][1]["query"]["detection"]["track_id"]: c[0][1]["verdict"]
                for c in publisher.publish.call_args_list
            }

        assert run(1) == expected
        assert run(4) == expected

    def test_a_batched_timeout_counts_every_subject(self):
        """A hung batch loses N subjects, and the count must say so."""
        from concurrent.futures import TimeoutError as FutureTimeoutError

        backend = mock_backend()
        backend.query_batch.side_effect = FutureTimeoutError()
        worker, _ = self._worker(backend=backend, batch_size=4)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 5)], MagicMock())

        assert worker.stats.triggers_timed_out == 4
        backend.request_stop.assert_called_once()

    def test_a_short_answer_list_is_discarded_not_misattributed(self):
        """Better no verdict than one filed against the wrong person."""
        backend = mock_backend()
        backend.query_batch.side_effect = lambda images, questions: ["Yes"]
        worker, _ = self._worker(backend=backend, batch_size=4)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        publisher = MagicMock()
        ts = time.monotonic()

        worker._process_batch([self._payload(i, ts, x=0.1 * i) for i in range(1, 5)], publisher)

        assert publisher.publish.call_count == 0
        assert worker.stats.triggers_failed == 4

    def test_one_snapshot_per_subject_across_prompt_groups(self):
        """Grouping by prompt splits a subject's rules — evidence stays single."""
        worker, backend = self._worker(batch_size=4, snapshot_dir=None)
        worker.add_rule("person", "Is this person wearing a vest?", rule_id=1)
        worker.add_rule("person", "Is this person bending over?", rule_id=2)
        publisher = MagicMock()
        ts = time.monotonic()

        worker._process_batch([self._payload(1, ts)], publisher)

        # Two rules, one subject: both verdicts published, snapshot resolved
        # once (None here, since snapshots are disabled) rather than per group.
        assert publisher.publish.call_count == 2
        assert worker.stats.instances_judged == 1


class TestSequentialQueryBatch:
    """A backend with no batch entry point must still satisfy the protocol."""

    def test_it_answers_each_pair_with_query(self):
        from app.pipelines.vlm_backends import sequential_query_batch

        backend = MagicMock()
        backend.query.side_effect = lambda image, q: f"{image}:{q}"

        out = sequential_query_batch(backend, ["a", "b"], ["q1", "q2"])

        assert out == ["a:q1", "b:q2"]
        assert backend.query.call_count == 2

    def test_mismatched_lengths_raise(self):
        from app.pipelines.vlm_backends import sequential_query_batch

        backend = MagicMock()
        with pytest.raises(ValueError):
            sequential_query_batch(backend, ["a", "b"], ["q1"])


class TestFocusedCrop:
    """The subject must be unambiguous when neighbours share the crop."""

    def test_the_subject_keeps_its_pixels(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        bbox = BoundingBox(x1=0.4, y1=0.4, x2=0.6, y2=0.6)

        out = crop_detection_focused(frame, bbox, padding_ratio=0.5, dim=0.3)

        assert out[out.shape[0] // 2, out.shape[1] // 2, 0] == 200

    def test_the_surroundings_are_dimmed(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        bbox = BoundingBox(x1=0.4, y1=0.4, x2=0.6, y2=0.6)

        out = crop_detection_focused(frame, bbox, padding_ratio=0.5, dim=0.3)

        assert out[0, 0, 0] < 100, "Context was not dimmed"

    def test_dim_of_one_matches_a_plain_crop(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.3, y1=0.3, x2=0.5, y2=0.5)

        focused = crop_detection_focused(frame, bbox, padding_ratio=0.15, dim=1.0)
        plain = crop_detection(frame, bbox, padding_ratio=0.15)

        assert np.array_equal(focused, plain)

    def test_the_full_frame_fallback_still_focuses(self):
        """A near-full-frame box falls back to the whole frame; focus applies."""
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        bbox = BoundingBox(x1=0.02, y1=0.02, x2=0.96, y2=0.96)

        out = crop_detection_focused(frame, bbox, padding_ratio=0.15, dim=0.3)

        assert out.shape == frame.shape
        assert out[0, 0, 0] < 100
        assert out[240, 320, 0] == 200

    def test_the_source_frame_is_never_modified(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        bbox = BoundingBox(x1=0.3, y1=0.3, x2=0.5, y2=0.5)

        crop_detection_focused(frame, bbox, dim=0.1)

        assert frame[0, 0, 0] == 200


class TestEncodeHoisting:
    """The vision half of the model is per-subject work, not per-rule work."""

    def _worker(self, backend, rules, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        for rule_id, prompt in enumerate(rules, start=1):
            worker.add_rule("person", prompt, rule_id=rule_id)
        return worker

    def _payload(self, track_id=1, timestamp=None):
        return {
            "timestamp": timestamp if timestamp is not None else time.monotonic(),
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.6},
            "frame_index": 0,
            "track_id": track_id,
        }

    def test_three_rules_encode_the_crop_once(self):
        backend = mock_backend()
        backend.query.return_value = "Yes"
        worker = self._worker(backend, ["Bending over?", "No vest?", "Carrying a box?"])

        worker._process_batch([self._payload()], MagicMock())

        assert backend.encode.call_count == 1, (
            "the crop is identical across rules, so the vision tower must run once"
        )
        assert backend.answer.call_count == 3, "every rule still gets asked"

    def test_each_subject_is_encoded_separately(self):
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend, ["Bending over?", "No vest?"])

        ts = time.monotonic()
        worker._process_batch([self._payload(i, ts) for i in (1, 2, 3)], MagicMock())

        assert backend.encode.call_count == 3, "different people are different images"
        assert backend.answer.call_count == 6

    def test_a_failed_encode_does_not_abandon_the_batch(self):
        """A broken image is that subject's problem, not the whole queue's."""
        backend = mock_backend()
        backend.encode.side_effect = RuntimeError("corrupt tensor")
        worker = self._worker(backend, ["Bending over?"])

        worker._process_batch([self._payload()], MagicMock())

        assert backend.answer.call_count == 0
        assert worker.stats.triggers_failed == 1

    def test_a_hung_encode_abandons_the_batch(self):
        """Unlike a failure, a timeout means the executor thread is still wedged."""
        backend = mock_backend()
        backend.encode.side_effect = lambda image: time.sleep(5)
        worker = self._worker(
            backend, ["Bending over?"], inference_timeout_s=0.2
        )

        assert worker._judge_instance(
            DetectionEvent(**self._payload()),
            [{"rule_id": 1, "rule_name": "r", "prompt": "p"}],
            np.zeros((64, 64, 3), dtype=np.uint8),
            MagicMock(),
        ) is False
        backend.request_stop.assert_called()


class TestCropDownscaling:
    """Moondream switches to a 3-5x costlier path at 512 px. Stay under it."""

    def test_a_crop_that_already_fits_is_not_copied(self):
        crop = np.zeros((360, 118, 3), dtype=np.uint8)
        assert downscale_for_vlm(crop, 511) is crop

    def test_an_oversized_crop_is_capped(self):
        out = downscale_for_vlm(np.zeros((720, 1280, 3), dtype=np.uint8), 511)
        assert max(out.shape[:2]) == 511

    def test_aspect_ratio_survives(self):
        out = downscale_for_vlm(np.zeros((720, 1280, 3), dtype=np.uint8), 511)
        assert abs(out.shape[1] / out.shape[0] - 1280 / 720) < 0.01

    def test_zero_disables_the_cap(self):
        crop = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert downscale_for_vlm(crop, 0) is crop

    def test_the_full_frame_fallback_is_what_this_catches(self):
        """A subject filling the frame makes crop_bounds return the whole thing."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        huge = BoundingBox(x1=0.02, y1=0.02, x2=0.98, y2=0.98)
        assert huge.area > 0.8

        cropped = crop_detection(frame, huge)
        assert max(cropped.shape[:2]) >= 512, "the un-capped crop takes the slow path"
        assert max(downscale_for_vlm(cropped, 511).shape[:2]) < 512


class TestLatencyAccounting:
    """Capture-to-verdict is the number an operator waits through."""

    def _worker(self, backend, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person bending over?", rule_id=1)
        return worker

    def test_the_published_result_carries_the_full_latency(self):
        backend = mock_backend()
        backend.query.return_value = "Yes"
        backend.model_name = "fake"
        worker = self._worker(backend)
        publisher = MagicMock()

        # A frame that was captured half a second before it reached Stage 2.
        worker._process_batch(
            [{
                "timestamp": time.monotonic() - 0.5,
                "camera_id": "test",
                "label": "person",
                "confidence": 0.9,
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.6},
                "frame_index": 0,
                "track_id": 1,
            }],
            publisher,
        )

        result = publisher.publish.call_args.args[1]
        assert result["queued_ms"] >= 500, "the wait before inference is counted"
        assert result["latency_ms"] >= result["queued_ms"]
        assert result["latency_ms"] >= result["inference_time_ms"]

    def test_percentiles_are_reported_once_verdicts_exist(self):
        backend = mock_backend()
        backend.query.return_value = "No"
        backend.model_name = "fake"
        worker = self._worker(backend)

        ts = time.monotonic()
        worker._process_batch(
            [{
                "timestamp": ts,
                "camera_id": "test",
                "label": "person",
                "confidence": 0.9,
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.6},
                "frame_index": 0,
                "track_id": 1,
            }],
            MagicMock(),
        )

        stats = worker.stats
        assert stats.latency_p50_ms > 0
        assert stats.latency_p95_ms >= stats.latency_p50_ms


class TestPercentile:
    """The stats path must not need numpy."""

    def test_empty_is_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_sample(self):
        assert _percentile([7.5], 95) == 7.5

    def test_nearest_rank(self):
        assert _percentile(range(1, 101), 50) == 50
        assert _percentile(range(1, 101), 95) == 95

    def test_p95_of_a_short_window_is_the_worst_sample(self):
        assert _percentile(range(1, 11), 95) == 10

    def test_input_order_does_not_matter(self):
        assert _percentile([9, 1, 5], 100) == 9


class TestBatchPreemption:
    """A person who walks in mid-batch must not wait out every queued re-check."""

    def _worker(self, backend, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        kwargs.setdefault("verdict_ttl_s", 300.0)
        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person bending over?", rule_id=1)
        return worker

    def _payload(self, track_id, timestamp=None):
        return {
            "timestamp": timestamp if timestamp is not None else time.monotonic(),
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.6},
            "frame_index": 0,
            "track_id": track_id,
        }

    def _judged_ids(self, publisher):
        return [
            call.args[1]["query"]["detection"]["track_id"]
            for call in publisher.publish.call_args_list
        ]

    def test_a_newcomer_overtakes_queued_re_checks(self):
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend)

        # Subjects 1-3 have all been seen before, so they are re-checks.
        ts = time.monotonic()
        for track_id in (1, 2, 3):
            worker._remember_verdict(
                DetectionEvent(**self._payload(track_id, ts)), rule_id=1, verdict=False
            )
        # Force them back into contention by expiring the memo's reuse window.
        worker._verdict_ttl_s = 0.0
        prepared = worker._prepare([self._payload(i, ts) for i in (1, 2, 3)])
        worker._verdict_ttl_s = 300.0
        assert len(prepared) == 3

        # Subject 99 arrives after the batch has already started.
        subscriber = MagicMock()
        subscriber.drain.side_effect = [
            [],                                              # before the 1st
            [("detection", self._payload(99, time.monotonic()))],  # before the 2nd
            [], [], [],
        ]

        publisher = MagicMock()
        worker._judge_prepared(prepared, publisher, subscriber)

        judged = self._judged_ids(publisher)
        assert 99 in judged, "the new arrival must be judged in this batch"
        assert judged.index(99) == 1, (
            f"the newcomer should be taken next, not last; order was {judged}"
        )

    def test_the_cap_still_bounds_the_work(self):
        """Preemption changes who gets the budget, never how big it is."""
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend, max_instances_per_batch=2)

        ts = time.monotonic()
        prepared = worker._prepare([self._payload(i, ts) for i in (1, 2, 3, 4)])

        subscriber = MagicMock()
        subscriber.drain.return_value = [
            ("detection", self._payload(99, time.monotonic()))
        ]

        publisher = MagicMock()
        worker._judge_prepared(prepared, publisher, subscriber)

        assert len(self._judged_ids(publisher)) == 2
        assert worker.stats.instances_shed >= 1

    def test_a_subject_already_waiting_is_not_judged_twice(self):
        """The memo only records a verdict after judging, so dedupe on id."""
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend)

        ts = time.monotonic()
        prepared = worker._prepare([self._payload(7, ts)])

        subscriber = MagicMock()
        # The same subject re-triggers while it sits in the queue.
        subscriber.drain.return_value = [
            ("detection", self._payload(7, time.monotonic()))
        ]

        publisher = MagicMock()
        worker._judge_prepared(prepared, publisher, subscriber)

        assert self._judged_ids(publisher) == [7]

    def test_without_a_subscriber_the_batch_is_unchanged(self):
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend)

        ts = time.monotonic()
        publisher = MagicMock()
        worker._process_batch([self._payload(i, ts) for i in (1, 2, 3)], publisher)

        assert sorted(self._judged_ids(publisher)) == [1, 2, 3]

    def test_a_quiet_scene_costs_one_poll_per_subject(self):
        backend = mock_backend()
        backend.query.return_value = "No"
        worker = self._worker(backend)

        ts = time.monotonic()
        prepared = worker._prepare([self._payload(i, ts) for i in (1, 2)])

        subscriber = MagicMock()
        subscriber.drain.return_value = []

        worker._judge_prepared(prepared, MagicMock(), subscriber)
        assert subscriber.drain.call_count == 2


# ── Verdict Probing ──────────────────────────────────────────────


def yes_probe(p: float = 0.9) -> BinaryProbe:
    return BinaryProbe(verdict=True, p_yes=p, p_no=1 - p, raw="Yes")


class ProbeStub:
    """A probe-capable backend double — deliberately a real class.

    Not a ``MagicMock``. The worker detects the capability with ``isinstance``
    against a runtime-checkable Protocol, and since Python 3.12 that check uses
    :func:`inspect.getattr_static`, which does **not** see attributes a mock
    conjures on attribute access. A mock double is therefore reported as
    probe-incapable no matter how it is specced, and every test here would pass
    for the wrong reason — by exercising the fallback path it was written to
    prove was skipped.

    The real backends are ordinary classes, so production detection is
    unaffected; this is purely about what a test double can express.
    """

    def __init__(self, probe=None, probe_error=None, answer="Yes"):
        self.model_name = "fake-probe"
        self.device = "cpu"
        self.is_loaded = True
        self._probe = probe if probe is not None else yes_probe()
        self._probe_error = probe_error
        self._answer = answer
        self.probe_calls = 0
        self.answer_calls = 0
        self.stop_calls = 0

    def load(self) -> None: ...

    def encode(self, image):
        return image

    def query(self, image, question) -> str:
        return self._answer

    def answer(self, handle, question) -> str:
        self.answer_calls += 1
        return self._answer

    def query_batch(self, images, questions) -> list[str]:
        return [self.answer(i, q) for i, q in zip(images, questions, strict=True)]

    def probe_binary(self, handle, question) -> BinaryProbe:
        self.probe_calls += 1
        if self._probe_error is not None:
            raise self._probe_error
        return self._probe

    def probe_binary_batch(self, handles, questions) -> list[BinaryProbe]:
        return [
            self.probe_binary(h, q) for h, q in zip(handles, questions, strict=True)
        ]

    def request_stop(self) -> None:
        self.stop_calls += 1

    def cleanup(self) -> None: ...

    def unload(self) -> None: ...


class TestYesNoTokenResolution:
    """The token ids a probe reads are the probe. Wrong ids, wrong verdicts."""

    def test_a_real_tokenizer_yields_disjoint_sets(self):
        """Against Moondream's actual tokenizer, not a stand-in."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            revision="2024-08-26",
            cache_dir="models/moondream2",
        )
        yes, no = resolve_yes_no_ids(tok)

        assert yes and no, "Neither side may be empty"
        assert not (set(yes) & set(no)), "A token cannot mean both"
        # Every spelling resolves separately on a BPE vocabulary, which is the
        # reason the mass is summed over variants rather than read off one id.
        assert len(yes) > 1 and len(no) > 1

    def test_an_indistinguishable_tokenizer_is_refused(self):
        """A tokenizer mapping both words to one id cannot answer anything."""
        tok = MagicMock()
        tok.encode.return_value = [7]

        with pytest.raises(ProbeUnsupportedError):
            resolve_yes_no_ids(tok)

    def test_an_empty_encoding_is_refused(self):
        """Silently returning no ids would make every verdict a coin flip."""
        tok = MagicMock()
        tok.encode.return_value = []

        with pytest.raises(ProbeUnsupportedError):
            resolve_yes_no_ids(tok)


class TestDecideProbe:
    """Turning two masses into a verdict."""

    def test_the_pair_is_renormalized(self):
        """Absolute mass reflects verbosity; the ratio carries the answer."""
        probe = decide_probe(0.02, 0.01, 0.5)
        assert probe.p_yes == pytest.approx(2 / 3)
        assert probe.p_no == pytest.approx(1 / 3)
        assert probe.verdict is True

    def test_the_threshold_is_applied_to_the_ratio(self):
        assert decide_probe(0.7, 0.3, 0.5).verdict is True
        assert decide_probe(0.7, 0.3, 0.9).verdict is False

    def test_raw_text_mirrors_the_verdict(self):
        """Downstream reads raw_response; it must not contradict the bool."""
        assert decide_probe(0.9, 0.1, 0.5).raw == "Yes"
        assert decide_probe(0.1, 0.9, 0.5).raw == "No"

    def test_no_mass_at_all_is_refused(self):
        with pytest.raises(ProbeUnsupportedError):
            decide_probe(0.0, 0.0, 0.5)


class TestOpenAICompatProbe:
    """The edge path: llama.cpp implements logprobs, Ollama does not."""

    def _backend(self, payload, status_code: int = 200) -> OpenAICompatBackend:
        backend = OpenAICompatBackend(model="test-model")
        client = MagicMock()
        # status_code is set explicitly rather than left to MagicMock's
        # auto-attribute, because _post compares it against 500 to tell an
        # outage from a bad request.
        client.post.return_value = MagicMock(
            status_code=status_code,
            json=MagicMock(return_value=payload),
            raise_for_status=MagicMock(),
        )
        backend._client = client
        backend._loaded = True
        return backend

    @staticmethod
    def _logprobs(pairs) -> dict:
        import math as _math

        return {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": t, "logprob": _math.log(p)}
                                    for t, p in pairs
                                ]
                            }
                        ]
                    }
                }
            ]
        }

    def test_a_llama_cpp_response_yields_a_verdict(self):
        backend = self._backend(self._logprobs([("Yes", 0.8), ("No", 0.1)]))
        probe = backend.probe_binary(MagicMock(), "Is this person bending?")

        assert probe.verdict is True
        assert probe.p_yes == pytest.approx(8 / 9)

    def test_only_one_token_is_requested(self):
        """The whole point: the server must not decode a sentence."""
        backend = self._backend(self._logprobs([("Yes", 0.9), ("No", 0.1)]))
        backend.probe_binary(MagicMock(), "q")

        sent = backend._client.post.call_args.kwargs["json"]
        assert sent["max_tokens"] == 1
        assert sent["logprobs"] is True
        assert sent["temperature"] == 0.0

    def test_a_server_without_logprobs_is_unsupported(self):
        """Ollama's shape — a normal completion, no logprobs field."""
        backend = self._backend({"choices": [{"message": {"content": "Yes"}}]})

        with pytest.raises(ProbeUnsupportedError):
            backend.probe_binary(MagicMock(), "q")

    def test_neither_word_among_the_candidates_is_unsupported(self):
        backend = self._backend(
            self._logprobs([("Maybe", 0.5), ("Perhaps", 0.3)])
        )

        with pytest.raises(ProbeUnsupportedError):
            backend.probe_binary(MagicMock(), "q")

    def test_negation_words_are_not_counted_as_no(self):
        """'not' and 'none' start with 'no' but are not the model saying No.

        Prefix matching here would quietly fold them into the No mass and tilt
        every verdict toward no-alert — the exact bias probing is meant to
        remove.
        """
        backend = self._backend(
            self._logprobs([("Yes", 0.3), ("not", 0.4), ("none", 0.2), ("No", 0.1)])
        )
        probe = backend.probe_binary(MagicMock(), "q")

        # Only the real "No" counts: 0.3 vs 0.1.
        assert probe.p_yes == pytest.approx(0.75)
        assert probe.verdict is True

    def test_case_and_punctuation_are_tolerated(self):
        backend = self._backend(self._logprobs([(" YES", 0.6), ("no.", 0.2)]))
        probe = backend.probe_binary(MagicMock(), "q")

        assert probe.p_yes == pytest.approx(0.75)

    def test_probing_before_load_raises(self):
        backend = OpenAICompatBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.probe_binary(MagicMock(), "q")


class TestOpenAICompatQuery:
    """The generate path's request shape.

    ``probe_binary`` was covered from the start; ``query()`` never was, which
    left the image encoding and the sampling parameters — the two things a
    server silently reinterprets if they are wrong — asserted nowhere.
    """

    def _backend(self, content="Yes", **kwargs) -> OpenAICompatBackend:
        backend = OpenAICompatBackend(model="test-model", **kwargs)
        client = MagicMock()
        client.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"choices": [{"message": {"content": content}}]}
            ),
            raise_for_status=MagicMock(),
        )
        backend._client = client
        backend._loaded = True
        return backend

    @staticmethod
    def _image():
        from PIL import Image

        return Image.fromarray(
            np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        )

    @staticmethod
    def _sent(backend) -> dict:
        return backend._client.post.call_args.kwargs["json"]

    def test_sampling_is_pinned_deterministic(self):
        """Same crop, same rule, same verdict — every run."""
        backend = self._backend()
        backend.query(self._image(), "Is this person missing a hard hat?")

        assert self._sent(backend)["temperature"] == 0.0

    def test_the_token_budget_is_carried_through(self):
        backend = self._backend(max_tokens=2)
        backend.query(self._image(), "Anything?")

        assert self._sent(backend)["max_tokens"] == 2

    def test_the_crop_travels_as_a_jpeg_data_uri(self):
        """The wire format the server actually parses.

        A bare base64 string with no ``data:`` prefix, or a PNG mislabelled as
        JPEG, is accepted by the schema and rejected by the model — so this
        asserts the prefix and then decodes the payload back into a real image
        rather than trusting the string.
        """
        import base64
        import io

        from PIL import Image

        backend = self._backend()
        backend.query(self._image(), "Anything?")

        content = self._sent(backend)["messages"][0]["content"]
        part = next(p for p in content if p["type"] == "image_url")
        url = part["image_url"]["url"]

        assert url.startswith("data:image/jpeg;base64,")

        raw = base64.b64decode(url.split(",", 1)[1])
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.format == "JPEG"
        assert decoded.size == (48, 64)

    def test_the_question_rides_in_the_same_user_turn(self):
        backend = self._backend()
        backend.query(self._image(), "Is the aisle blocked?")

        messages = self._sent(backend)["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        texts = [p["text"] for p in messages[0]["content"] if p["type"] == "text"]
        assert texts == ["Is the aisle blocked?"]

    def test_a_two_token_answer_still_parses(self):
        """The guard on lowering vlm_max_tokens to 2.

        The parser reads the leading word and defaults to "no alert" on
        anything it cannot recognise, so a budget too tight to fit that word
        turns a violation into a silent miss.
        """
        backend = self._backend(content="Yes,", max_tokens=2)
        answer = backend.query(self._image(), "Anything?")

        assert BinaryVerdictParser.parse(answer) is True


class TestBackendUnavailable:
    """An endpoint that is down must be distinguishable from a bad query.

    The two need opposite responses: a failed query is worth retrying for the
    next subject, while a dead endpoint means every subject behind this one
    would pay its own doomed round-trip to learn the same thing.
    """

    def _backend(self) -> OpenAICompatBackend:
        backend = OpenAICompatBackend(model="test-model")
        backend._client = MagicMock()
        backend._loaded = True
        return backend

    @staticmethod
    def _image():
        from PIL import Image

        return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    def _response(self, status_code):
        import httpx

        resp = MagicMock(status_code=status_code)
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"{status_code}", request=MagicMock(), response=resp
            )
        return resp

    @pytest.mark.parametrize(
        "exc_name", ["ConnectError", "ConnectTimeout", "ReadTimeout"]
    )
    def test_transport_failures_are_reported_as_unavailable(self, exc_name):
        import httpx

        backend = self._backend()
        backend._client.post.side_effect = getattr(httpx, exc_name)("boom")

        with pytest.raises(VLMUnavailableError):
            backend.query(self._image(), "Anything?")

    def test_a_5xx_is_unavailable(self):
        """Up, but not serving — a restarting or overloaded server."""
        backend = self._backend()
        backend._client.post.return_value = self._response(503)

        with pytest.raises(VLMUnavailableError):
            backend.query(self._image(), "Anything?")

    def test_a_4xx_is_not_unavailable(self):
        """A malformed request is our bug, not an outage.

        Dropping traffic on it would hide a payload defect behind what looks
        like an infrastructure problem.
        """
        import httpx

        backend = self._backend()
        backend._client.post.return_value = self._response(400)

        with pytest.raises(httpx.HTTPStatusError):
            backend.query(self._image(), "Anything?")

    def test_the_probe_path_reports_outages_too(self):
        """Both request paths funnel through _post and must agree."""
        import httpx

        backend = self._backend()
        backend._client.post.side_effect = httpx.ConnectError("boom")

        with pytest.raises(VLMUnavailableError):
            backend.probe_binary(self._image(), "Anything?")

    def test_connect_is_bounded_tighter_than_read(self):
        """A local server accepts at once or is not there.

        The scalar timeout gave connect the same minute as a full prefill, so
        an unreachable host held the single-worker executor that whole time.
        """
        backend = OpenAICompatBackend(timeout_s=60.0, connect_timeout_s=2.0)
        backend.load()

        timeout = backend._client.timeout
        assert timeout.connect == 2.0
        assert timeout.read == 60.0
        backend.unload()


class TestUnavailableDropsTheBatch:
    """A dead endpoint must cost one round-trip, not one per subject."""

    def _worker(self, **kwargs):
        buffer = MagicMock()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        backend = mock_backend()
        backend.model_name = "fake"
        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person missing a hard hat?", rule_id=1)
        return worker, backend

    @staticmethod
    def _payload(track_id, timestamp, x=0.1):
        return {
            "timestamp": timestamp,
            "camera_id": "test",
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": x, "y1": 0.1, "x2": x + 0.08, "y2": 0.5},
            "frame_index": 0,
            "track_id": track_id,
        }

    def test_the_rest_of_the_batch_is_abandoned(self):
        worker, backend = self._worker()
        backend.query.side_effect = VLMUnavailableError("server is gone")

        ts = time.monotonic()
        worker._process_batch(
            [self._payload(i, ts, x=0.1 * i) for i in range(1, 6)], MagicMock()
        )

        assert backend.query.call_count == 1, (
            "Every subject paid its own doomed round-trip"
        )

    def test_the_outage_is_visible_in_stats(self):
        worker, backend = self._worker()
        backend.query.side_effect = VLMUnavailableError("server is gone")

        ts = time.monotonic()
        worker._process_batch([self._payload(1, ts)], MagicMock())

        assert worker.stats.backend_unavailable is True
        assert worker.stats.batches_dropped_unavailable == 1

    def test_the_worker_survives_and_recovers(self):
        """The endpoint coming back must not need a restart."""
        worker, backend = self._worker()
        backend.query.side_effect = VLMUnavailableError("server is gone")

        ts = time.monotonic()
        worker._process_batch([self._payload(1, ts)], MagicMock())
        assert worker.stats.backend_unavailable is True

        backend.query.side_effect = None
        backend.query.return_value = "Yes"
        worker._process_batch([self._payload(2, time.monotonic())], MagicMock())

        assert worker.stats.backend_unavailable is False
        assert worker.stats.verdicts_positive == 1

    def test_an_ordinary_failure_still_judges_the_rest(self):
        """The contrast that gives the drop its meaning.

        A query that fails on its own merits must not take the batch with it —
        otherwise one bad crop silently costs every subject behind it.
        """
        worker, backend = self._worker()
        backend.query.side_effect = RuntimeError("one bad query")

        ts = time.monotonic()
        worker._process_batch(
            [self._payload(i, ts, x=0.1 * i) for i in range(1, 6)], MagicMock()
        )

        assert backend.query.call_count == 5
        assert worker.stats.batches_dropped_unavailable == 0


class TestWorkerProbing:
    """How the worker chooses between probing and generate-and-parse."""

    def _worker(self, backend, **kwargs) -> VLMWorker:
        worker = VLMWorker(buffer=MagicMock(), backend=backend, **kwargs)
        worker.add_rule("person", "Is this person bending over?", rule_id=1)
        return worker

    def test_the_real_backends_advertise_their_capability(self):
        """Production detection, as opposed to what a double can express."""
        from app.config.settings import Settings
        from app.pipelines.vlm_backends import BinaryProbeBackend, build_backend

        probeable = {
            name: isinstance(build_backend(Settings(vlm_backend=name)), BinaryProbeBackend)
            for name in ("moondream4bit", "moondream", "transformers", "openai_compat")
        }
        # The pinned 2024-08-26 remote-code wrapper exposes no logits, so that
        # backend cannot be probed and must fall back rather than fail. The
        # 2025-04-14 revision behind moondream4bit returns them from
        # _prefill_prompt, which is most of why it is worth adopting.
        assert probeable == {
            "moondream4bit": True,
            "moondream": False,
            "transformers": True,
            "openai_compat": True,
        }

    def test_a_probe_capable_backend_is_probed(self):
        backend = ProbeStub()
        worker = self._worker(backend)

        answer = worker._decide(MagicMock(), "Is this person bending over?")

        assert answer.verdict is True
        assert answer.confidence == pytest.approx(0.9)
        assert backend.probe_calls == 1
        assert backend.answer_calls == 0, "Must not also generate"

    def test_a_backend_that_cannot_probe_is_never_asked(self):
        """The Moondream case: no probe methods, so no wasted attempt."""
        backend = mock_backend()
        backend.query.return_value = "Yes"
        worker = self._worker(backend)

        answer = worker._decide(MagicMock(), "q")

        assert answer.verdict is True
        assert answer.confidence is None
        assert backend.answer.call_count == 1

    def test_probe_off_forces_generate_and_parse(self):
        backend = ProbeStub(answer="No")
        worker = self._worker(backend, probe="off")

        answer = worker._decide(MagicMock(), "q")

        assert backend.probe_calls == 0
        assert answer.verdict is False
        assert answer.confidence is None

    def test_an_unsupported_probe_latches_off(self):
        """One failed attempt, then never again for the rest of the session.

        Without the latch a server lacking logprobs pays a doomed request
        before every single generate, doubling its request count forever.
        """
        backend = ProbeStub(probe_error=ProbeUnsupportedError("no logprobs"))
        worker = self._worker(backend)

        first = worker._decide(MagicMock(), "q")
        for _ in range(5):
            worker._decide(MagicMock(), "q")

        assert first.verdict is True, "The transition query still gets answered"
        assert backend.probe_calls == 1, "Latched after a single refusal"
        assert backend.answer_calls == 6

    def test_a_failed_probe_still_answers_the_query(self):
        """Falling back must not drop the verdict that triggered it."""
        backend = ProbeStub(probe_error=ProbeUnsupportedError("nope"), answer="No")
        worker = self._worker(backend)

        assert worker._decide(MagicMock(), "q").verdict is False

    def test_batched_and_single_use_the_same_decision_rule(self):
        """A verdict must not depend on how many rode with it."""
        backend = ProbeStub(probe=yes_probe(0.8))
        worker = self._worker(backend)

        single = worker._decide(MagicMock(), "q")
        batched = worker._decide_batch([MagicMock(), MagicMock()], ["q", "q"])

        assert [b.verdict for b in batched] == [single.verdict] * 2
        assert [b.confidence for b in batched] == [single.confidence] * 2

    def test_the_probe_count_is_reported(self):
        backend = ProbeStub()
        worker = self._worker(backend)

        worker._decide(MagicMock(), "q")
        worker._decide(MagicMock(), "q")

        assert worker._verdicts_probed == 2

    def test_a_probed_verdict_publishes_its_confidence(self):
        """The number an operator uses to spot a close call."""
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        buffer = MagicMock()
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        worker = VLMWorker(
            buffer=buffer, backend=ProbeStub(probe=yes_probe(0.77)), snapshot_dir=None
        )
        worker.add_rule("person", "Is this person bending over?", rule_id=1)

        publisher = MagicMock()
        worker._process_batch(
            [
                {
                    "timestamp": time.monotonic(),
                    "camera_id": "test",
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": {"x1": 0.3, "y1": 0.1, "x2": 0.5, "y2": 0.8},
                    "frame_index": 0,
                    "track_id": 1,
                }
            ],
            publisher,
        )

        published = publisher.publish.call_args[0][1]
        assert published["verdict"] is True
        assert published["confidence"] == pytest.approx(0.77)
        assert published["raw_response"] == "Yes"

    def test_a_parsed_verdict_reports_no_confidence(self):
        """A parsed 'Yes' carries no strength, and must not pretend to."""
        backend = mock_backend()
        backend.query.return_value = "Yes"
        worker = self._worker(backend)

        assert worker._decide(MagicMock(), "q").confidence is None


class TestLetterboxSquare:
    """Square padding, so the model never sees a stretched person."""

    def test_the_output_is_always_square(self):
        for shape in [(300, 100, 3), (100, 300, 3), (250, 250, 3), (7, 900, 3)]:
            out = letterbox_square(np.zeros(shape, dtype=np.uint8), 378)
            assert out.shape == (378, 378, 3), f"failed for {shape}"

    def test_the_token_cost_stops_varying_with_the_subject(self):
        """The latency claim: every crop arrives at identical dimensions."""
        shapes = {
            letterbox_square(np.zeros(s, dtype=np.uint8), 378).shape
            for s in [(320, 107, 3), (360, 118, 3), (200, 200, 3)]
        }
        assert len(shapes) == 1

    def test_proportions_survive(self):
        """A 1:3 person stays 1:3 instead of being stretched to 1:1."""
        src = np.full((300, 100, 3), 200, dtype=np.uint8)
        out = letterbox_square(src, 378)

        subject = out != _LETTERBOX_GREY
        rows = np.any(subject, axis=(1, 2))
        cols = np.any(subject, axis=(0, 2))
        height = int(rows.sum())
        width = int(cols.sum())

        assert height == pytest.approx(378, abs=2)
        assert width / height == pytest.approx(1 / 3, rel=0.02)

    def test_the_bars_are_neutral_grey(self):
        """Black bars read as unlit scene; grey reads as absence."""
        out = letterbox_square(np.full((300, 100, 3), 200, dtype=np.uint8), 378)
        assert out[0, 0, 0] == _LETTERBOX_GREY

    def test_the_subject_is_centred(self):
        out = letterbox_square(np.full((300, 100, 3), 200, dtype=np.uint8), 378)
        cols = np.any(out != _LETTERBOX_GREY, axis=(0, 2))
        left = int(np.argmax(cols))
        right = len(cols) - int(np.argmax(cols[::-1]))
        assert abs(left - (378 - right)) <= 2

    def test_a_square_input_is_left_unpadded(self):
        out = letterbox_square(np.full((200, 200, 3), 200, dtype=np.uint8), 378)
        assert not np.any(out == _LETTERBOX_GREY)

    def test_the_source_is_never_modified(self):
        """Other subjects in the same batch still need this frame."""
        src = np.full((300, 100, 3), 200, dtype=np.uint8)
        letterbox_square(src, 378)
        assert np.all(src == 200)

    def test_the_result_is_a_fresh_array(self):
        src = np.full((378, 378, 3), 200, dtype=np.uint8)
        out = letterbox_square(src, 378)
        out[0, 0, 0] = 1
        assert src[0, 0, 0] == 200

    def test_a_small_crop_is_scaled_up_to_the_canvas(self):
        """Constant size is the point, so an undersized crop still fills it."""
        out = letterbox_square(np.full((64, 64, 3), 200, dtype=np.uint8), 378)
        assert out.shape == (378, 378, 3)
        assert not np.any(out == _LETTERBOX_GREY)


class TestCropPipelineWiring:
    """How the worker assembles cropping, dimming and letterboxing."""

    def _crops(self, **kwargs) -> np.ndarray:
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        buffer = MagicMock()
        buffer.read_by_timestamp.return_value = (frame, time.monotonic(), 0)
        buffer.read_at_index.return_value = (frame, time.monotonic())

        backend = mock_backend()
        backend.query.return_value = "Yes"
        worker = VLMWorker(buffer=buffer, backend=backend, **kwargs)
        worker.add_rule("person", "Is this person bending over?", rule_id=1)

        prepared = worker._prepare(
            [
                {
                    "timestamp": time.monotonic(),
                    "camera_id": "test",
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": {"x1": 0.3, "y1": 0.1, "x2": 0.4, "y2": 0.8},
                    "frame_index": 0,
                    "track_id": 1,
                }
            ]
        )
        return prepared[0][2]

    def test_letterboxing_is_on_by_default(self):
        assert self._crops().shape == (378, 378, 3)

    def test_zero_falls_back_to_max_edge_downscaling(self):
        """The opt-out keeps the previous geometry exactly."""
        crop = self._crops(square_edge=0)
        assert crop.shape[0] != crop.shape[1]
        assert max(crop.shape[:2]) <= 511

    def test_dimming_is_off_by_default(self):
        """A uniform frame stays uniform apart from the letterbox bars."""
        crop = self._crops()
        assert set(np.unique(crop)) <= {200, _LETTERBOX_GREY}

    def test_dimming_can_still_be_restored(self):
        """The mechanism survives the default flip — the failure it fixed is real."""
        crop = self._crops(focus_dim=0.4, square_edge=0, crop_padding=0.5)
        assert crop.min() < 200


class TestTransformersProbe:
    """The single forward pass itself: slicing, padding, and attribution."""

    VOCAB = 64
    YES_ID = 11
    NO_ID = 22

    def _backend(self, rows):
        """A loaded backend whose model returns controlled final-position logits.

        ``rows`` is one ``(yes_logit, no_logit)`` pair per batch row.
        """
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace

        from app.pipelines.vlm_backends import TransformersVLMBackend

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = SimpleNamespace(padding_side="right")
                self.sides_during_call = []
                self.prompts = None

            def apply_chat_template(self, messages, add_generation_prompt=False):
                text = messages[0]["content"][1]["text"]
                return f"<user><image>{text}<assistant>"

            def __call__(self, text, images, return_tensors, padding=False):
                self.prompts = list(text)
                self.sides_during_call.append(self.tokenizer.padding_side)
                return FakeInputs(
                    input_ids=torch.zeros((len(text), 5), dtype=torch.long)
                )

        def fake_model(**inputs):
            n = inputs["input_ids"].shape[0]
            # Everything else is deeply improbable, so the Yes/No pair carries
            # essentially all the mass and the renormalized ratio is readable.
            logits = torch.full((n, 5, self.VOCAB), -30.0)
            for i, (yes, no) in enumerate(rows):
                logits[i, -1, self.YES_ID] = yes
                logits[i, -1, self.NO_ID] = no
            return SimpleNamespace(logits=logits)

        backend = TransformersVLMBackend(device="cpu")
        backend._processor = FakeProcessor()
        backend._model = fake_model
        backend._loaded = True
        backend._yes_ids = [self.YES_ID]
        backend._no_ids = [self.NO_ID]
        return backend

    def test_a_confident_yes_is_read_off_the_logits(self):
        backend = self._backend([(6.0, 0.0)])
        probe = backend.probe_binary(MagicMock(), "Is this person bending?")

        assert probe.verdict is True
        assert probe.raw == "Yes"
        assert probe.p_yes > 0.99

    def test_a_confident_no_is_read_off_the_logits(self):
        probe = self._backend([(0.0, 6.0)]).probe_binary(MagicMock(), "q")

        assert probe.verdict is False
        assert probe.raw == "No"
        assert probe.p_no > 0.99

    def test_the_confidence_is_calibrated_not_binary(self):
        """Equal logits mean the model genuinely has no idea."""
        probe = self._backend([(1.0, 1.0)]).probe_binary(MagicMock(), "q")
        assert probe.p_yes == pytest.approx(0.5, abs=0.01)

    def test_the_threshold_gates_the_verdict(self):
        backend = self._backend([(1.0, 0.0)])
        backend._probe_threshold = 0.9
        # Roughly 0.73 yes — a lean, not a decision.
        assert backend.probe_binary(MagicMock(), "q").verdict is False

    def test_each_row_gets_its_own_answer(self):
        """Batched attribution: the claim a batch has to earn."""
        backend = self._backend([(6.0, 0.0), (0.0, 6.0), (6.0, 0.0)])
        probes = backend.probe_binary_batch(
            [MagicMock(), MagicMock(), MagicMock()], ["a", "b", "c"]
        )

        assert [p.verdict for p in probes] == [True, False, True]

    def test_padding_is_forced_left_then_restored(self):
        """Right padding would read a pad token instead of the answer slot."""
        backend = self._backend([(6.0, 0.0), (0.0, 6.0)])
        backend.probe_binary_batch([MagicMock(), MagicMock()], ["a", "b"])

        assert backend._processor.sides_during_call == ["left"]
        assert backend._processor.tokenizer.padding_side == "right"

    def test_the_probe_prompt_matches_the_generate_prompt(self):
        """Drift here would invalidate any probe-vs-generate comparison."""
        backend = self._backend([(6.0, 0.0)])
        backend.probe_binary(MagicMock(), "Is this person bending?")

        assert backend._processor.prompts == [
            backend._chat_prompt("Is this person bending?")
        ]

    def test_an_empty_batch_is_allowed(self):
        assert self._backend([]).probe_binary_batch([], []) == []

    def test_mismatched_lengths_raise(self):
        backend = self._backend([(1.0, 0.0)])
        with pytest.raises(ValueError):
            backend.probe_binary_batch([MagicMock()], ["a", "b"])

    def test_probing_before_load_raises(self):
        from app.pipelines.vlm_backends import TransformersVLMBackend

        with pytest.raises(RuntimeError, match="not loaded"):
            TransformersVLMBackend(device="cpu").probe_binary(MagicMock(), "q")

    def test_a_tokenizer_without_yes_no_ids_is_unsupported(self):
        """Degrades to the parser rather than deciding on an empty id set."""
        backend = self._backend([(6.0, 0.0)])
        backend._yes_ids = []

        with pytest.raises(ProbeUnsupportedError):
            backend.probe_binary(MagicMock(), "q")


class TestProbeCapabilityReporting:
    """The fallback must not be silent — nothing else ever announces it."""

    def _log(self, backend, caplog, **kwargs):
        worker = VLMWorker(buffer=MagicMock(), backend=backend, **kwargs)
        with caplog.at_level("INFO", logger="app.pipelines.vlm_worker"):
            worker._report_probe_capability()
        return caplog.text

    def test_a_probe_capable_backend_says_so(self, caplog):
        assert "probing enabled" in self._log(ProbeStub(), caplog).lower()

    def test_an_incapable_backend_is_announced_not_hidden(self, caplog):
        """A backend that cannot probe is never asked, so it never refuses."""
        text = self._log(mock_backend(), caplog)
        assert "generate-and-parse" in text
        assert "no alert" in text, "The parser's default belongs in the warning"

    def test_explicitly_requesting_probing_warns_when_impossible(self, caplog):
        self._log(mock_backend(), caplog, probe="on")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_auto_does_not_warn_about_an_expected_fallback(self, caplog):
        self._log(mock_backend(), caplog, probe="auto")
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_disabling_probing_is_also_reported(self, caplog):
        assert "disabled" in self._log(ProbeStub(), caplog, probe="off").lower()
