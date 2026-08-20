"""ONNX Runtime driver for the quantized Moondream2 export.

`Nagafi/moondream2-q4-onnx` re-publishes Xenova's ONNX conversion of
Moondream2 with 4-bit weights. It is packaged for **transformers.js**, so
nothing in Python knows how to drive it: `optimum` has no `moondream1`
architecture, and the three graphs must be wired together by hand.

The export splits the model into three sessions:

    vision_encoder      pixel_values [1,3,378,378] -> image_features [1,729,2048]
    embed_tokens        input_ids    [1,L]         -> inputs_embeds  [1,L,2048]
    decoder_model_merged
                        inputs_embeds + attention_mask + 24 layers of KV cache
                        -> logits [1,L,51200] + updated cache

`decoder_model_merged` handles both prefill and decode: prefill passes a
zero-length cache, decode passes back what the previous step returned.

Two defects in the published repo have to be worked around before any of
this runs; see :func:`load_tokenizer`.

Example:
    >>> rt = OnnxMoondreamRuntime("models/moondream2-q4-onnx")
    >>> rt.load()
    >>> rt.query(pil_image, "Is this person bending over?")
    'Yes'
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Moondream's prompt format. The image occupies its own span of embeddings
# between the BOS token and the question.
_PROMPT_TEMPLATE = "\n\nQuestion: {question}\n\nAnswer:"

_BOS_ID = 50256
_EOS_ID = 50256

# SigLIP preprocessing, from the repo's preprocessor_config.json.
_IMAGE_SIZE = 378
_IMAGE_MEAN = 0.5
_IMAGE_STD = 0.5


def load_tokenizer(model_dir: Path):
    """Load the repo's tokenizer, repairing two transformers.js-only defects.

    Neither defect is reachable from JavaScript, so both ship unnoticed:

    1. ``<image>`` is an added token with id ``-200``. The Rust ``tokenizers``
       library types token ids as ``u32`` and refuses the file outright
       ("invalid value: integer `-200`, expected u32"). The token is dropped
       here — the image span is spliced in by embedding index instead, so a
       marker token in the vocabulary is never needed.

    2. The ``TemplateProcessing`` post-processor references the special token
       ``<|endoftext|>``, but its ``special_tokens`` map is keyed ``<s>``.
       The lookup misses and the library panics ("no entry found for key").
       The key is rewritten to match what the template asks for.

    Args:
        model_dir: Directory holding ``tokenizer.json``.

    Returns:
        A ``tokenizers.Tokenizer`` that prepends BOS and round-trips text.
    """
    from tokenizers import Tokenizer

    raw = json.loads((model_dir / "tokenizer.json").read_text())

    raw["added_tokens"] = [tok for tok in raw["added_tokens"] if tok["id"] >= 0]

    post = raw.get("post_processor") or {}
    specials = post.get("special_tokens") or {}
    if specials and "<|endoftext|>" not in specials:
        # Re-key by the id the template actually references.
        entry = next(iter(specials.values()))
        post["special_tokens"] = {"<|endoftext|>": entry}

    return Tokenizer.from_str(json.dumps(raw))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize a PIL image into the vision encoder's input.

    Args:
        image: Any PIL image.

    Returns:
        Float32 NCHW array of shape ``(1, 3, 378, 378)``, normalized to
        roughly [-1, 1] with SigLIP's mean/std of 0.5.
    """
    from PIL import Image as PILImage

    resized = image.convert("RGB").resize(
        (_IMAGE_SIZE, _IMAGE_SIZE), PILImage.BICUBIC
    )
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - _IMAGE_MEAN) / _IMAGE_STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


@dataclass
class QueryTrace:
    """Per-stage timings for one inference, in milliseconds.

    Recorded because the aggregate latency hides which stage dominates —
    and on this export the answer decides whether the model is deployable.
    """

    vision_ms: float = 0.0
    embed_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    n_prompt_tokens: int = 0
    n_generated: int = 0
    stopped_early: bool = False
    per_token_ms: list[float] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        """End-to-end inference time."""
        return self.vision_ms + self.embed_ms + self.prefill_ms + self.decode_ms

    @property
    def decode_tokens_per_s(self) -> float:
        """Sustained generation rate, excluding prefill."""
        if self.decode_ms <= 0 or self.n_generated == 0:
            return 0.0
        return self.n_generated / (self.decode_ms / 1000.0)


class OnnxMoondreamRuntime:
    """Drives the three-session ONNX export end to end.

    Args:
        model_dir: Directory containing ``onnx/`` plus tokenizer files.
        decoder_variant: ``"q4"`` or ``"q4f16"`` — which decoder graph to use.
        intra_op_threads: ORT intra-op thread count. ``0`` lets ORT choose
            (one thread per physical core).
        max_new_tokens: Generation cap.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/moondream2-q4-onnx",
        decoder_variant: str = "q4",
        intra_op_threads: int = 0,
        max_new_tokens: int = 16,
    ) -> None:
        self._dir = Path(model_dir)
        self._variant = decoder_variant
        self._threads = intra_op_threads
        self._max_new_tokens = max_new_tokens

        self._vision = None
        self._embed = None
        self._decoder = None
        self._tokenizer = None
        self._loaded = False
        self._stop_event = threading.Event()

        # Discovered from the decoder graph at load time.
        self._n_layers = 0
        self._n_heads = 0
        self._head_dim = 0
        self._kv_dtype = np.float32
        self._decoder_inputs: list[str] = []
        self._wants_position_ids = False

        self.last_trace: QueryTrace | None = None

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Whether all three sessions are live."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Identifier recorded on results."""
        return f"moondream2-{self._variant}-onnx"

    def _session_options(self):
        """Build ORT session options with the configured thread count."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._threads > 0:
            opts.intra_op_num_threads = self._threads
        return opts

    # ── Load ─────────────────────────────────────────────────────

    def load(self) -> dict[str, float]:
        """Create the three ORT sessions and inspect the decoder's cache shape.

        Returns:
            Per-session load wall times in seconds, for the benchmark report.

        Raises:
            FileNotFoundError: If an expected ONNX file is missing.
        """
        if self._loaded:
            return {}

        import onnxruntime as ort

        onnx_dir = self._dir / "onnx"
        paths = {
            "vision": onnx_dir / "vision_encoder_q4.onnx",
            "embed": onnx_dir / "embed_tokens_q4.onnx",
            "decoder": onnx_dir / f"decoder_model_merged_{self._variant}.onnx",
        }
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} graph: {path}")

        opts = self._session_options()
        timings: dict[str, float] = {}

        for name, path in paths.items():
            t0 = time.perf_counter()
            session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            timings[name] = time.perf_counter() - t0
            setattr(self, f"_{name}", session)
            logger.info(f"Loaded {name} graph in {timings[name]:.2f}s")

        self._tokenizer = load_tokenizer(self._dir)
        self._discover_cache_shape()
        self._loaded = True
        return timings

    def _discover_cache_shape(self) -> None:
        """Read layer count, head count, and head dim off the decoder graph.

        Reading these rather than hardcoding them means a differently-shaped
        export still runs, and a shape mismatch surfaces here instead of as
        an opaque ORT error mid-inference.
        """
        self._decoder_inputs = [i.name for i in self._decoder.get_inputs()]
        self._wants_position_ids = "position_ids" in self._decoder_inputs

        key_inputs = [n for n in self._decoder_inputs if n.endswith(".key")]
        self._n_layers = len(key_inputs)
        if self._n_layers == 0:
            raise RuntimeError(
                f"No past_key_values inputs found on the decoder graph; "
                f"inputs are {self._decoder_inputs}"
            )

        spec = next(i for i in self._decoder.get_inputs() if i.name == key_inputs[0])
        # Shape is [batch, num_heads, past_len, head_dim]; the symbolic dims
        # come back as strings, the concrete ones as ints.
        self._n_heads = spec.shape[1] if isinstance(spec.shape[1], int) else 32
        self._head_dim = spec.shape[3] if isinstance(spec.shape[3], int) else 64
        self._kv_dtype = np.float16 if "float16" in spec.type else np.float32

        logger.info(
            f"Decoder cache: {self._n_layers} layers x {self._n_heads} heads x "
            f"{self._head_dim} dim ({np.dtype(self._kv_dtype).name}), "
            f"position_ids={'yes' if self._wants_position_ids else 'no'}"
        )

    # ── Inference ────────────────────────────────────────────────

    def _empty_cache(self) -> dict[str, np.ndarray]:
        """Build a zero-length KV cache for the prefill pass."""
        empty = np.zeros((1, self._n_heads, 0, self._head_dim), dtype=self._kv_dtype)
        cache = {}
        for i in range(self._n_layers):
            cache[f"past_key_values.{i}.key"] = empty
            cache[f"past_key_values.{i}.value"] = empty
        return cache

    def _build_prompt_embeds(
        self, image: Image.Image, question: str, trace: QueryTrace
    ) -> np.ndarray:
        """Encode the image and splice it between BOS and the question.

        The published tokenizer cannot represent the image token, so the
        image span is inserted positionally: ``[BOS] <image tokens> <text>``,
        which is the order the export was traced with.
        """
        pixel_values = preprocess_image(image)

        t0 = time.perf_counter()
        image_features = self._vision.run(None, {"pixel_values": pixel_values})[0]
        trace.vision_ms = (time.perf_counter() - t0) * 1000.0

        text_ids = self._tokenizer.encode(
            _PROMPT_TEMPLATE.format(question=question)
        ).ids
        # The post-processor prepends BOS; keep it only on the leading span.
        if text_ids and text_ids[0] == _BOS_ID:
            text_ids = text_ids[1:]

        t0 = time.perf_counter()
        bos_embed = self._embed_ids([_BOS_ID])
        text_embed = self._embed_ids(text_ids)
        trace.embed_ms = (time.perf_counter() - t0) * 1000.0

        embeds = np.concatenate(
            [bos_embed, image_features.astype(np.float32), text_embed], axis=1
        )
        trace.n_prompt_tokens = embeds.shape[1]
        return embeds

    def _embed_ids(self, ids: list[int]) -> np.ndarray:
        """Look up embeddings for a list of token ids."""
        arr = np.asarray([ids], dtype=np.int64)
        return self._embed.run(None, {"input_ids": arr})[0].astype(np.float32)

    def request_stop(self) -> None:
        """Ask an in-flight generation to stop at the next token boundary."""
        self._stop_event.set()

    def query(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image with greedy decoding.

        Args:
            image: PIL image (already cropped to the subject by the caller).
            question: Plain-English question.

        Returns:
            The generated answer text, stripped.

        Raises:
            RuntimeError: If the runtime was never loaded.
        """
        if not self._loaded:
            raise RuntimeError("ONNX runtime is not loaded. Call load() first.")

        self._stop_event.clear()
        trace = QueryTrace()

        embeds = self._build_prompt_embeds(image, question, trace)
        cache = self._empty_cache()
        attention_mask = np.ones((1, embeds.shape[1]), dtype=np.int64)

        generated: list[int] = []
        past_len = 0

        for step in range(self._max_new_tokens):
            feeds = {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": attention_mask,
                **cache,
            }
            if self._wants_position_ids:
                seq = embeds.shape[1]
                feeds["position_ids"] = np.arange(
                    past_len, past_len + seq, dtype=np.int64
                )[None, :]

            t0 = time.perf_counter()
            outputs = self._decoder.run(None, feeds)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if step == 0:
                trace.prefill_ms = elapsed_ms
            else:
                trace.decode_ms += elapsed_ms
                trace.per_token_ms.append(elapsed_ms)

            logits = outputs[0]
            next_id = int(np.argmax(logits[0, -1, :]))

            names = [o.name for o in self._decoder.get_outputs()]
            cache = {
                name.replace("present", "past_key_values"): outputs[idx]
                for idx, name in enumerate(names)
                if name.startswith("present")
            }

            past_len += embeds.shape[1]
            if next_id == _EOS_ID:
                break

            generated.append(next_id)
            if self._stop_event.is_set():
                trace.stopped_early = True
                break

            embeds = self._embed_ids([next_id])
            attention_mask = np.ones((1, past_len + 1), dtype=np.int64)

        trace.n_generated = len(generated)
        self.last_trace = trace

        return self._tokenizer.decode(generated).strip()

    def unload(self) -> None:
        """Release all three sessions."""
        self._vision = None
        self._embed = None
        self._decoder = None
        self._tokenizer = None
        self._loaded = False

        import gc

        gc.collect()
        logger.info("ONNX Moondream runtime unloaded.")


class OnnxMoondreamBackend:
    """VLMBackend adapter so Stage 2 can run the ONNX export unchanged.

    Satisfies the same protocol as :class:`~app.pipelines.vlm_backends.MoondreamBackend`,
    so :class:`~app.pipelines.vlm_worker.VLMWorker` needs no changes.

    Args:
        model_dir: Directory holding the export.
        decoder_variant: ``"q4"`` or ``"q4f16"``.
        max_new_tokens: Generation cap. Verdicts are short.
        intra_op_threads: ORT intra-op threads; 0 lets ORT decide.
    """

    def __init__(
        self,
        model_dir: str = "models/moondream2-q4-onnx",
        decoder_variant: str = "q4",
        max_new_tokens: int = 16,
        intra_op_threads: int = 0,
    ) -> None:
        self._runtime = OnnxMoondreamRuntime(
            model_dir=model_dir,
            decoder_variant=decoder_variant,
            intra_op_threads=intra_op_threads,
            max_new_tokens=max_new_tokens,
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the sessions are live."""
        return self._runtime.is_loaded

    @property
    def device(self) -> str:
        """ORT execution provider in use."""
        return "onnxruntime:cpu"

    @property
    def model_name(self) -> str:
        """Identifier recorded on each result."""
        return self._runtime.model_name

    def load(self) -> None:
        """Create the ORT sessions."""
        self._runtime.load()

    def query(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image."""
        return self._runtime.query(image, question)

    def request_stop(self) -> None:
        """Abandon generation at the next token boundary."""
        self._runtime.request_stop()

    def cleanup(self) -> None:
        """No per-inference cleanup — ORT manages its own arenas."""

    def unload(self) -> None:
        """Release the sessions."""
        self._runtime.unload()

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "unloaded"
        return f"OnnxMoondreamBackend(variant='{self._runtime._variant}', {status})"
