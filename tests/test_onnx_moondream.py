"""Tests for the ONNX Moondream2 driver.

The heavy graphs are not loaded here — these cover the glue that the
published export makes necessary: the tokenizer repair, SigLIP
preprocessing, and the per-stage trace arithmetic. Tests that need the
actual 1.5 GB of weights are marked ``slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipelines.onnx_moondream import (
    OnnxMoondreamBackend,
    QueryTrace,
    load_tokenizer,
    preprocess_image,
)

MODEL_DIR = Path("models/moondream2-q4-onnx")
_HAS_MODEL = (MODEL_DIR / "tokenizer.json").exists()


# ═══════════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════════


class TestPreprocessImage:
    """SigLIP preprocessing must match preprocessor_config.json."""

    def test_output_shape_is_square_378(self):
        """Any input aspect ratio is squashed to 378x378 NCHW."""
        # Stage 2 crops are tall and narrow (a standing person), so the
        # non-square case is the normal one, not an edge case.
        img = Image.new("RGB", (107, 323), (128, 128, 128))
        out = preprocess_image(img)
        assert out.shape == (1, 3, 378, 378)
        assert out.dtype == np.float32

    def test_normalization_maps_mid_gray_to_zero(self):
        """With mean=std=0.5, a 127.5-valued pixel lands at ~0."""
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        out = preprocess_image(img)
        assert abs(float(out.mean())) < 0.01

    def test_normalization_range(self):
        """Pure black and white map to the ends of [-1, 1]."""
        black = preprocess_image(Image.new("RGB", (32, 32), (0, 0, 0)))
        white = preprocess_image(Image.new("RGB", (32, 32), (255, 255, 255)))
        assert float(black.max()) == pytest.approx(-1.0, abs=1e-5)
        assert float(white.min()) == pytest.approx(1.0, abs=1e-5)

    def test_grayscale_input_is_converted(self):
        """A single-channel image still yields three channels."""
        out = preprocess_image(Image.new("L", (64, 64), 200))
        assert out.shape[1] == 3


# ═══════════════════════════════════════════════════════════════════
# Tokenizer repair
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_MODEL, reason="model export not downloaded")
class TestLoadTokenizer:
    """The published tokenizer.json is unloadable in Python until repaired."""

    def test_published_file_is_rejected_unrepaired(self):
        """Guards the premise: without the repair, loading raises.

        If a future re-upload fixes the file upstream, this test fails and
        the workaround can be deleted rather than carried forever.
        """
        from tokenizers import Tokenizer

        with pytest.raises(Exception):
            Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    def test_repaired_tokenizer_loads_and_encodes(self):
        """After repair, encoding produces ids led by BOS."""
        tok = load_tokenizer(MODEL_DIR)
        ids = tok.encode("\n\nQuestion: Is this person bending over?\n\nAnswer:").ids
        assert ids[0] == 50256
        assert len(ids) > 5

    def test_round_trips_text(self):
        """Decode(encode(x)) preserves the answer text."""
        tok = load_tokenizer(MODEL_DIR)
        text = " Yes, the person is bending over."
        assert tok.decode(tok.encode(text).ids) == text

    def test_negative_image_token_is_dropped(self):
        """The -200 <image> token must not survive into the vocabulary."""
        raw = json.loads((MODEL_DIR / "tokenizer.json").read_text())
        assert any(t["id"] < 0 for t in raw["added_tokens"]), "premise changed"

        tok = load_tokenizer(MODEL_DIR)
        assert tok.token_to_id("<image>") is None


# ═══════════════════════════════════════════════════════════════════
# Trace arithmetic
# ═══════════════════════════════════════════════════════════════════


class TestQueryTrace:
    """The per-stage breakdown is what makes a slow result actionable."""

    def test_total_sums_every_stage(self):
        trace = QueryTrace(vision_ms=100.0, embed_ms=5.0, prefill_ms=800.0, decode_ms=95.0)
        assert trace.total_ms == pytest.approx(1000.0)

    def test_decode_rate(self):
        """8 tokens in 1000 ms is 8 tok/s."""
        trace = QueryTrace(decode_ms=1000.0, n_generated=8)
        assert trace.decode_tokens_per_s == pytest.approx(8.0)

    def test_decode_rate_is_zero_when_nothing_generated(self):
        """An immediate EOS must not divide by zero."""
        assert QueryTrace(decode_ms=0.0, n_generated=0).decode_tokens_per_s == 0.0


# ═══════════════════════════════════════════════════════════════════
# Backend protocol conformance
# ═══════════════════════════════════════════════════════════════════


class TestBackendProtocol:
    """The adapter must be drop-in for VLMWorker without loading weights."""

    def test_satisfies_vlm_backend_protocol(self):
        from app.pipelines.vlm_backends import VLMBackend

        assert isinstance(OnnxMoondreamBackend(), VLMBackend)

    def test_not_loaded_before_load(self):
        assert OnnxMoondreamBackend().is_loaded is False

    def test_query_before_load_raises(self):
        backend = OnnxMoondreamBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.query(Image.new("RGB", (64, 64)), "Is this person bending over?")

    def test_model_name_records_the_variant(self):
        """Results must be attributable to the exact graph that produced them."""
        assert OnnxMoondreamBackend(decoder_variant="q4").model_name == (
            "moondream2-q4-onnx"
        )
        assert OnnxMoondreamBackend(decoder_variant="q4f16").model_name == (
            "moondream2-q4f16-onnx"
        )

    def test_missing_weights_raise_file_not_found(self):
        backend = OnnxMoondreamBackend(model_dir="models/does-not-exist")
        with pytest.raises(FileNotFoundError):
            backend.load()

    def test_repr_shows_variant_and_status(self):
        assert "q4" in repr(OnnxMoondreamBackend())
        assert "unloaded" in repr(OnnxMoondreamBackend())
