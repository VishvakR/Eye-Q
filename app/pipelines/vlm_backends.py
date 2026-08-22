"""Pluggable Vision-Language Model backends for Stage 2.

Eye-Q targets edge hardware with **no cloud dependency**, so both backends
here run entirely on the local device:

    - :class:`MoondreamBackend` — loads Moondream2 in-process via Transformers.
      Zero external services; the simplest thing that works, and the default.
    - :class:`OpenAICompatBackend` — POSTs to a *local* OpenAI-compatible
      server (Ollama, llama.cpp, vLLM) on the same box. On edge CPUs a
      quantized server is typically several times faster than HF Transformers,
      and it keeps multi-GB model weights out of this process.

Both satisfy the :class:`VLMBackend` protocol, so :class:`~app.pipelines.vlm_worker.VLMWorker`
does not care which is in use. Select with ``EYEQ_VLM_BACKEND``.

Example:
    >>> backend = build_backend(get_settings())
    >>> backend.load()
    >>> backend.query(pil_image, "Is a pallet blocking the fire exit?")
    'Yes, a wooden pallet is directly in front of the exit door.'
"""

from __future__ import annotations

import base64
import io
import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image

    from app.config.settings import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class VLMBackend(Protocol):
    """Interface every Stage 2 VLM implementation must provide."""

    @property
    def is_loaded(self) -> bool:
        """Whether the backend is ready to answer queries."""
        ...

    @property
    def device(self) -> str:
        """Human-readable device/target description, for the status endpoint."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the active model, recorded on each result."""
        ...

    def load(self) -> None:
        """Prepare the backend (load weights, or verify the server is up)."""
        ...

    def query(self, image: Image.Image, question: str) -> str:
        """Ask a question about an image and return the raw text answer."""
        ...

    def encode(self, image: Image.Image) -> object:
        """Do whatever per-image work can be shared across several questions.

        Stage 2 asks every rule matching a subject about the *same* crop, so
        without this the vision half of the model is re-run once per rule. The
        returned value is an opaque handle, meaningful only to the backend that
        produced it, and is passed straight back to :meth:`answer`.

        Backends that cannot separate the two halves return the image unchanged
        and do the whole job in :meth:`answer`; the worker's code path is the
        same either way, it simply saves nothing.
        """
        ...

    def answer(self, handle: object, question: str) -> str:
        """Answer one question about an image already passed to :meth:`encode`."""
        ...

    def query_batch(
        self, images: list[Image.Image], questions: list[str]
    ) -> list[str]:
        """Answer one question per image, in as few forward passes as possible.

        Positional: ``questions[i]`` is asked about ``images[i]`` and the
        returned answer at index ``i`` belongs to that pair. Backends that
        cannot batch delegate to :func:`sequential_query_batch` and satisfy the
        contract at the same cost as calling :meth:`query` in a loop.

        Batching is not free and is off by default — see ``vlm_batch_size``.
        Stage 2's prompt carries 729 image tokens against a single-token
        answer, which makes it prefill-bound and compute-bound, and batching
        amortizes a *weight read* that prefill was never waiting on. It also
        multiplies the KV cache by the batch size.
        """
        ...

    def request_stop(self) -> None:
        """Ask an in-flight ``query()`` to abandon generation early.

        Called by the worker when an inference blows its timeout. Python
        cannot kill a thread, so without cooperation a timed-out inference
        keeps burning the executor and stalls the *next* trigger too. Backends
        that cannot interrupt themselves implement this as a no-op.
        """
        ...

    def cleanup(self) -> None:
        """Best-effort resource release after a single inference."""
        ...

    def unload(self) -> None:
        """Release the model entirely."""
        ...


def sequential_query_batch(
    backend: VLMBackend, images: list[Image.Image], questions: list[str]
) -> list[str]:
    """Answer each pair with its own :meth:`~VLMBackend.query` call.

    The honest implementation of :meth:`~VLMBackend.query_batch` for a backend
    with no batch entry point. Kept as a free function so those backends are a
    two-line delegation rather than a copied loop, and so the worker never has
    to ask whether real batching is available.

    Args:
        backend: The backend to drive.
        images: One image per question.
        questions: One question per image.

    Returns:
        Answers, positionally matched to the inputs.
    """
    return [backend.query(image, q) for image, q in zip(images, questions, strict=True)]


# ── Binary Verdict Probing (optional capability) ─────────────────
#
# Stage 2 only ever needs one bit back. Generating it as *text* means running
# the language model autoregressively — a prefill over ~729 image tokens to
# produce "Yes", then another full forward pass per token after that, all so
# the answer can be turned back into a bool by a string parser that defaults
# to False on anything it fails to recognise.
#
# Probing reads the same bit off the distribution the prefill already
# computed: compare the probability mass on the Yes tokens against the No
# tokens at the first answer position. One forward pass, no decode, no
# parsing, and a calibrated confidence falls out for free.
#
# This is a *capability*, not part of :class:`VLMBackend` — Moondream's
# remote-code wrapper exposes no logits, and an OpenAI-compatible server only
# exposes them if it implements ``logprobs``. The worker checks for it and
# falls back to generate-and-parse, so no deployment loses the ability to
# answer, only the ability to answer cheaply.

# Casing and leading-space variants of the two words. Which one a model emits
# first depends on its chat template and its tokenizer, so the mass is summed
# over all of them rather than betting on one spelling.
_YES_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
_NO_VARIANTS = ("No", " No", "no", " no", "NO", " NO")

# How many candidates to ask an HTTP server for. The winning token is almost
# always Yes or No outright; this is headroom for a model that leads with
# punctuation or a stray article.
_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class BinaryProbe:
    """A yes/no decision read directly from the model's own probabilities.

    Attributes:
        verdict: True when the Yes mass wins against the configured threshold.
        p_yes: Yes probability, renormalized over the yes/no pair alone.
        p_no: ``1.0 - p_yes``.
        raw: ``"Yes"`` or ``"No"``, so callers expecting the old text answer
            keep working unchanged.
    """

    verdict: bool
    p_yes: float
    p_no: float
    raw: str


class ProbeUnsupportedError(RuntimeError):
    """Raised when a backend cannot report yes/no probabilities.

    Distinct from a failed inference: this says *this backend will never be
    able to do this*, so the caller should stop asking and fall back for the
    rest of the session rather than retrying per query.
    """


class VLMUnavailableError(RuntimeError):
    """Raised when the backend's endpoint itself is unusable.

    Distinct from a query that failed on its own merits: this says *nothing
    will get through right now*, so every query queued behind this one would
    pay a full doomed round-trip to learn the same thing. The worker abandons
    the rest of the batch on it rather than grinding through the remainder.

    Raised for transport failures (refused connection, connect or read
    timeout) and for 5xx replies, which mean the server is up but not
    serving. A 4xx is left alone — that is a malformed request, which is our
    bug to fix and not a reason to drop traffic.
    """


@runtime_checkable
class BinaryProbeBackend(Protocol):
    """Optional capability: decide a yes/no question without generating text."""

    def probe_binary(self, handle: object, question: str) -> BinaryProbe:
        """Probe one question about a handle from :meth:`VLMBackend.encode`."""
        ...

    def probe_binary_batch(
        self, handles: list[object], questions: list[str]
    ) -> list[BinaryProbe]:
        """Probe one question per handle, positionally matched."""
        ...


def resolve_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids for the Yes and No spellings, as two disjoint lists.

    Only the *first* token of each variant is taken. A multi-token spelling
    still commits to the answer at its first token, which is the only position
    a single forward pass can see, and it is discriminative there.

    Raises:
        ProbeUnsupportedError: If either side resolves to nothing, or the two are
            indistinguishable. Both mean a probe on this tokenizer would be
            reading noise, and silently returning an empty id set would make
            every verdict a coin flip.
    """
    def first_ids(variants: tuple[str, ...]) -> set[int]:
        ids = set()
        for text in variants:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
        return ids

    yes = first_ids(_YES_VARIANTS)
    no = first_ids(_NO_VARIANTS)

    # A token that spells both cannot separate them, so it informs neither.
    ambiguous = yes & no
    yes -= ambiguous
    no -= ambiguous

    if not yes or not no:
        raise ProbeUnsupportedError(
            "Tokenizer yields no distinguishable Yes/No first tokens "
            f"(yes={sorted(yes)}, no={sorted(no)}, dropped={sorted(ambiguous)})."
        )

    return sorted(yes), sorted(no)


def decide_probe(p_yes: float, p_no: float, threshold: float) -> BinaryProbe:
    """Turn raw Yes/No mass into a verdict, renormalized over the pair.

    Renormalizing matters: the absolute mass on Yes depends on how much the
    model wanted to say anything else at all ("The", "I", "Sure"), which is a
    property of its verbosity, not of its answer. The ratio between the two is
    the part that carries the verdict, and it is what the threshold applies to.

    Raises:
        ProbeUnsupportedError: If neither token drew any mass, so there is no
            answer in the distribution to read.
    """
    total = p_yes + p_no
    if total <= 0.0:
        raise ProbeUnsupportedError(
            "Neither 'Yes' nor 'No' drew any probability mass at the answer "
            "position; this prompt cannot be probed."
        )

    norm_yes = p_yes / total
    verdict = norm_yes >= threshold
    return BinaryProbe(
        verdict=verdict,
        p_yes=norm_yes,
        p_no=1.0 - norm_yes,
        raw="Yes" if verdict else "No",
    )


# ── In-Process Moondream2 ────────────────────────────────────────


class MoondreamBackend:
    """Wrapper around vikhyatk/moondream2 for visual question answering.

    Moondream2 is a tiny (1.86B params) vision-language model built for edge
    deployment. It runs fully in-process — no network, no external service.

    Note on the API surface: revision ``2024-08-26`` exposes
    ``encode_image()`` + ``answer_question()``. Newer revisions expose a
    single ``.query()`` method instead. This class targets the pinned
    revision, which is why ``transformers`` is capped below 4.50 in
    ``pyproject.toml``.

    Args:
        device: PyTorch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
        model_id: Hugging Face model identifier.
        revision: Model revision for reproducibility.
        cache_dir: Local directory for downloaded weights.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        cache_dir: str = "models/moondream2",
        max_new_tokens: int = 24,
    ) -> None:
        self._device = device
        self._model_id = model_id
        self._revision = revision
        self._cache_dir = cache_dir
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident in memory."""
        return self._loaded

    @property
    def device(self) -> str:
        """Active inference device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Hugging Face model identifier."""
        return self._model_id

    def _from_pretrained(self, device: str) -> None:
        """Load model + tokenizer onto a specific device."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            revision=self._revision,
            device_map={"": device},
            cache_dir=self._cache_dir,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            revision=self._revision,
            cache_dir=self._cache_dir,
        )
        self._device = device
        self._loaded = True

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        First run downloads ~3.7 GB and can take several minutes. Falls back
        to CPU if the preferred accelerator fails.

        Raises:
            Exception: If loading fails on both the preferred device and CPU.
        """
        if self._loaded:
            return

        logger.info(
            f"Loading VLM: {self._model_id} (revision={self._revision}) "
            f"on device '{self._device}'..."
        )

        try:
            self._from_pretrained(self._device)
            logger.info(f"VLM loaded successfully on '{self._device}'.")
            return
        except Exception as exc:
            logger.error(f"Failed to load VLM on '{self._device}': {exc}")
            if self._device == "cpu":
                raise

        logger.warning("Retrying VLM load on CPU...")
        try:
            self._from_pretrained("cpu")
            logger.info("VLM loaded on CPU (fallback).")
        except Exception as cpu_exc:
            logger.error(f"VLM CPU fallback also failed: {cpu_exc}")
            raise

    def query(self, image: Image.Image, question: str) -> str:
        """Ask a question about an image.

        Args:
            image: PIL Image to analyze.
            question: Plain-English question.

        Returns:
            Raw text response from the model.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        return self.answer(self.encode(image), question)

    def encode(self, image: Image.Image) -> object:
        """Run the vision tower once and return its embedding.

        Moondream is already two calls internally, so exposing the seam costs
        nothing and lets the worker amortize the expensive half. Measured on an
        M2: three rules about one subject took 11.50 s re-encoding per rule
        against 9.33 s encoding once, for byte-identical verdicts.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        return self._model.encode_image(image)

    def answer(self, handle: object, question: str) -> str:
        """Ask one question against an embedding from :meth:`encode`.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        return self._model.answer_question(handle, question, self._tokenizer)

    def query_batch(
        self, images: list[Image.Image], questions: list[str]
    ) -> list[str]:
        """Answer every pair in one vision pass and one ``generate()``.

        The pinned revision already ships this as ``batch_answer``: it runs the
        vision tower over the whole list at once — ``VisionEncoder.forward``
        branches on ``isinstance(images, list)`` — then left-pads the per-image
        prompt embeddings and decodes the batch in a single call. Each crop
        keeps its own full-resolution embedding, so a batched verdict is not a
        lower-resolution one; row index maps back to subject.

        ``max_new_tokens`` is passed deliberately. ``batch_answer`` defaults to
        512 and a batched ``generate()`` runs until *every* row hits EOS, so one
        model that ignores "answer Yes or No" and rambles would hold the whole
        batch to its own token budget. :class:`BinaryVerdictParser` reads only
        the leading word.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        if len(images) != len(questions):
            raise ValueError(
                f"query_batch needs one question per image, got "
                f"{len(images)} image(s) and {len(questions)} question(s)."
            )

        if not images:
            return []

        # A one-element batch would still pay batch_answer's padding and
        # stacking, for a batch that cannot benefit from either.
        if len(images) == 1:
            return [self.query(images[0], questions[0])]

        answers = self._model.batch_answer(
            images=[image.convert("RGB") for image in images],
            prompts=list(questions),
            tokenizer=self._tokenizer,
            max_new_tokens=self._max_new_tokens,
        )
        return [answer.strip() for answer in answers]

    def request_stop(self) -> None:
        """No-op — ``answer_question()`` exposes no stopping hook.

        Moondream's remote-code wrapper drives ``generate()`` internally and
        takes no ``stopping_criteria``, so a timed-out inference here runs to
        completion and is merely ignored. Use the ``transformers`` backend if
        interruptible generation matters for your deployment.
        """

    def cleanup(self) -> None:
        """Release accelerator memory after inference (best-effort).

        Apple's unified memory architecture makes this matter more than on
        discrete GPUs — the VLM competes with YOLO for the same pool.
        """
        try:
            import torch

            if self._device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass  # Cleanup is best-effort

    def unload(self) -> None:
        """Unload the model from memory entirely."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self.cleanup()

        import gc

        gc.collect()
        logger.info("VLM model unloaded.")

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"MoondreamBackend(model='{self._model_id}', device='{self._device}', {status})"


# ── Generic Transformers Vision-to-Text ──────────────────────────


def _stop_on_event_criteria(stop_event: threading.Event):
    """Build a ``StoppingCriteria`` that halts generation when the event is set.

    ``transformers`` is imported lazily so this module stays importable (and
    unit-testable) on machines that never load an in-process model.

    Args:
        stop_event: Event the worker sets when an inference exceeds its budget.

    Returns:
        A ``StoppingCriteriaList`` to hand to ``model.generate()``.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnEvent(StoppingCriteria):
        """Consulted once per generated token."""

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return stop_event.is_set()

    return StoppingCriteriaList([_StopOnEvent()])


class TransformersVLMBackend:
    """Any chat-templated vision-to-sequence model from Hugging Face.

    Unlike :class:`MoondreamBackend`, which targets Moondream's bespoke
    ``encode_image()``/``answer_question()`` API, this uses the standard
    ``AutoProcessor`` + ``AutoModelForVision2Seq`` path, so it runs the current
    generation of small open VLMs without per-model code.

    Models that fit an edge device, on the pinned ``transformers<4.50``:

    ==============================  =====  ==========================
    Model ID                        Size   Notes
    ==============================  =====  ==========================
    Qwen/Qwen2-VL-2B-Instruct       2B     Strong VQA; the default here
    Qwen/Qwen2.5-VL-3B-Instruct     3B     Better, needs transformers>=4.49
    HuggingFaceTB/SmolVLM-Instruct  2B     Lightest; Idefics3 architecture
    ==============================  =====  ==========================

    SmolVLM**2** and anything else whose ``model_type`` landed in
    transformers 4.50+ will not load until that pin is raised — the pin exists
    for :class:`MoondreamBackend`'s revision, so lifting it means retesting
    Moondream too.

    Generation is greedy (``do_sample=False``): a binary verdict must be
    reproducible, and sampling would make the same frame answer differently
    between runs.

    Args:
        model_id: Hugging Face model identifier.
        device: PyTorch device string.
        max_new_tokens: Generation cap. Verdicts are short.
        cache_dir: Local directory for downloaded weights.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 128,
        cache_dir: str = "models/vlm",
        probe_threshold: float = 0.5,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._cache_dir = cache_dir
        self._probe_threshold = probe_threshold
        self._model = None
        self._processor = None
        self._loaded = False
        self._stop_event = threading.Event()
        # Resolved once at load, against the tokenizer that was actually used.
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident in memory."""
        return self._loaded

    @property
    def device(self) -> str:
        """Active inference device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Hugging Face model identifier."""
        return self._model_id

    def request_stop(self) -> None:
        """Abandon the in-flight generation at the next token boundary.

        ``generate()`` consults the stopping criteria once per token, so this
        returns the executor thread within roughly one token of latency
        instead of leaving it pinned until the full response is produced.
        """
        self._stop_event.set()

    def _from_pretrained(self, device: str) -> None:
        """Load processor + model onto a specific device."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # fp32 on CPU: half precision is slower there and can produce NaNs.
        dtype = torch.float32 if device == "cpu" else torch.float16

        self._processor = AutoProcessor.from_pretrained(
            self._model_id, cache_dir=self._cache_dir
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            cache_dir=self._cache_dir,
        ).to(device)
        self._model.eval()

        # Resolved here rather than lazily so an unprobeable tokenizer is
        # reported at load, next to the rest of the model's setup, and not on
        # the first verdict of a live deployment. It is not fatal: an empty id
        # set turns probing off for this backend and the worker falls back to
        # generate-and-parse, which is exactly what it did before probing
        # existed. Refusing to load would be a far worse trade.
        try:
            self._yes_ids, self._no_ids = resolve_yes_no_ids(
                self._processor.tokenizer
            )
        except ProbeUnsupportedError as exc:
            self._yes_ids, self._no_ids = [], []
            logger.warning(f"Binary probing unavailable for {self._model_id}: {exc}")

        self._device = device
        self._loaded = True

    def load(self) -> None:
        """Load the model, falling back to CPU if the accelerator refuses."""
        if self._loaded:
            return

        logger.info(f"Loading VLM: {self._model_id} on device '{self._device}'...")

        try:
            self._from_pretrained(self._device)
            logger.info(f"VLM loaded successfully on '{self._device}'.")
            return
        except Exception as exc:
            logger.error(f"Failed to load VLM on '{self._device}': {exc}")
            if self._device == "cpu":
                raise

        logger.warning("Retrying VLM load on CPU...")
        self._from_pretrained("cpu")
        logger.info("VLM loaded on CPU (fallback).")

    def _chat_prompt(self, question: str) -> str:
        """Render one image+text user turn through the model's chat template.

        Shared by the generate and probe paths so the two cannot drift. A probe
        run against a different prompt than ``generate()`` would not be
        comparable to it, and the agreement check that justifies the swap would
        be measuring the prompt rather than the decision rule.
        """
        return self._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

    def query(self, image: Image.Image, question: str) -> str:
        """Ask a question about an image using the model's chat template.

        Args:
            image: PIL Image to analyze.
            question: Plain-English question.

        Returns:
            Raw text response, with the echoed prompt stripped.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        import torch

        prompt = self._chat_prompt(question)
        inputs = self._processor(
            text=[prompt], images=[image.convert("RGB")], return_tensors="pt"
        ).to(self._device)

        # Cleared per call so an abort from a previous, timed-out query cannot
        # kill this one before it starts.
        self._stop_event.clear()

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                stopping_criteria=_stop_on_event_criteria(self._stop_event),
            )

        # generate() returns prompt + completion; keep only the completion.
        prompt_len = inputs["input_ids"].shape[1]
        completion = generated[:, prompt_len:]

        return self._processor.batch_decode(
            completion, skip_special_tokens=True
        )[0].strip()

    def query_batch(
        self, images: list[Image.Image], questions: list[str]
    ) -> list[str]:
        """Answer every pair in one ``generate()`` over a padded batch.

        Padding is forced to the **left** so every row's completion begins at
        the same column, which is what lets the single ``prompt_len`` slice
        below be correct for all of them. Right padding would put each row's
        answer at a different offset and silently return prompt fragments.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        if len(images) != len(questions):
            raise ValueError(
                f"query_batch needs one question per image, got "
                f"{len(images)} image(s) and {len(questions)} question(s)."
            )

        if not images:
            return []

        import torch

        prompts = [self._chat_prompt(question) for question in questions]

        previous_side = self._processor.tokenizer.padding_side
        self._processor.tokenizer.padding_side = "left"
        try:
            inputs = self._processor(
                text=prompts,
                images=[image.convert("RGB") for image in images],
                return_tensors="pt",
                padding=True,
            ).to(self._device)
        finally:
            self._processor.tokenizer.padding_side = previous_side

        self._stop_event.clear()

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                stopping_criteria=_stop_on_event_criteria(self._stop_event),
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = generated[:, prompt_len:]

        return [
            text.strip()
            for text in self._processor.batch_decode(
                completions, skip_special_tokens=True
            )
        ]

    def probe_binary(self, handle: object, question: str) -> BinaryProbe:
        """Decide one yes/no question in a single forward pass.

        Routed through :meth:`probe_binary_batch` so a batch of one and a batch
        of many run the identical code. A system whose decision rule changes
        with ``vlm_batch_size`` would be almost impossible to reason about.
        """
        return self.probe_binary_batch([handle], [question])[0]

    def probe_binary_batch(
        self, handles: list[object], questions: list[str]
    ) -> list[BinaryProbe]:
        """Probe every pair in one forward pass over a padded batch.

        This is the whole optimisation. ``generate()`` runs the prefill to
        produce its first token and then keeps going, one full forward pass per
        additional token, until the model emits EOS or hits ``max_new_tokens``.
        The verdict was already decided at the first of those positions — every
        pass after it produces punctuation and justification that a string
        parser then throws away. Here the prefill runs once and the logits it
        already computed are read directly.

        Padding is forced **left**, as in :meth:`query_batch`: it is what makes
        position ``-1`` the true final prompt token for every row rather than
        whichever pad happened to land there.

        Raises:
            RuntimeError: If the model is not loaded.
            ProbeUnsupportedError: If the tokenizer yielded no usable Yes/No ids, or
                neither word drew any probability mass.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        if not self._yes_ids or not self._no_ids:
            raise ProbeUnsupportedError(
                f"No distinguishable Yes/No token ids for {self._model_id}."
            )

        if len(handles) != len(questions):
            raise ValueError(
                f"probe_binary_batch needs one question per image, got "
                f"{len(handles)} image(s) and {len(questions)} question(s)."
            )

        if not handles:
            return []

        import torch

        prompts = [self._chat_prompt(question) for question in questions]

        previous_side = self._processor.tokenizer.padding_side
        self._processor.tokenizer.padding_side = "left"
        try:
            inputs = self._processor(
                text=prompts,
                images=[image.convert("RGB") for image in handles],  # type: ignore[union-attr]
                return_tensors="pt",
                padding=True,
            ).to(self._device)
        finally:
            self._processor.tokenizer.padding_side = previous_side

        with torch.inference_mode():
            # No stopping criteria and no generation config: there is exactly
            # one pass, so there is no token boundary at which to interrupt.
            logits = self._model(**inputs).logits[:, -1, :]

        # Softmax in fp32 regardless of the model's dtype. On an fp16
        # accelerator the tail of the distribution underflows to zero, and the
        # Yes/No mass is frequently in that tail.
        probs = torch.softmax(logits.float(), dim=-1)
        yes_mass = probs[:, self._yes_ids].sum(dim=-1).tolist()
        no_mass = probs[:, self._no_ids].sum(dim=-1).tolist()

        return [
            decide_probe(float(y), float(n), self._probe_threshold)
            for y, n in zip(yes_mass, no_mass, strict=True)
        ]

    def encode(self, image: Image.Image) -> object:
        """Pass the image through — this backend has no shareable half.

        ``generate()`` consumes the processed image and the prompt together,
        with no exposed seam between the vision tower and the language model,
        so there is nothing to hoist. Splitting it would mean reaching into
        each architecture's internals — different for Qwen2-VL and SmolVLM —
        to cache vision features by hand, which is not worth the coupling.
        """
        return image

    def answer(self, handle: object, question: str) -> str:
        """Answer against the handle from :meth:`encode`, which is the image."""
        return self.query(handle, question)  # type: ignore[arg-type]

    def cleanup(self) -> None:
        """Release accelerator memory after inference (best-effort)."""
        try:
            import torch

            if self._device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self) -> None:
        """Unload the model from memory entirely."""
        self._model = None
        self._processor = None
        self._loaded = False
        self.cleanup()

        import gc

        gc.collect()
        logger.info("VLM model unloaded.")

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"TransformersVLMBackend(model='{self._model_id}', "
            f"device='{self._device}', {status})"
        )


# ── Local OpenAI-Compatible Server ───────────────────────────────


class OpenAICompatBackend:
    """VLM served by a local OpenAI-compatible HTTP endpoint.

    Works with Ollama, llama.cpp's server, vLLM, SGLang — anything exposing
    ``POST /v1/chat/completions`` with image content parts. The image is sent
    inline as a base64 data URI, so nothing is written to disk and nothing
    leaves the device.

    This is the recommended path on constrained edge hardware: a quantized
    model behind llama.cpp usually beats HF Transformers on CPU by a wide
    margin, and the weights live in the server process rather than here.

    Args:
        api_base: Base URL including the ``/v1`` suffix.
        model: Model name to request.
        api_key: Ignored by most local servers, but sent for compatibility.
        max_tokens: Generation cap. Verdicts are short — keep this small.
        timeout_s: Per-request read/write timeout — how long the server gets
            to answer once connected.
        connect_timeout_s: How long to wait for the connection itself. Kept
            short and separate from ``timeout_s``: a local server either
            accepts immediately or is not there.
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434/v1",
        model: str = "moondream",
        api_key: str = "EMPTY",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        probe_threshold: float = 0.5,
        connect_timeout_s: float = 2.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._api_key = api_key or "EMPTY"
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._probe_threshold = probe_threshold
        self._client = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the HTTP client is ready."""
        return self._loaded

    @property
    def device(self) -> str:
        """Reported as the server URL — inference happens out of process."""
        return f"remote:{self._api_base}"

    @property
    def model_name(self) -> str:
        """Model name requested from the server."""
        return self._model

    def load(self) -> None:
        """Create the HTTP client and verify the server is reachable.

        A failed reachability probe is logged but not fatal — the server may
        still be starting up, and each query retries the connection anyway.
        """
        if self._loaded:
            return

        import httpx

        # Connect is bounded far tighter than read. A read has to allow the
        # server a full prefill on a 729-token image, but a connect that has
        # not landed in two seconds is not going to — and the scalar form
        # applied timeout_s to both, so an unreachable (rather than actively
        # refused) host pinned the single-worker executor for a whole minute
        # before anything noticed.
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=self._connect_timeout_s,
                read=self._timeout_s,
                write=self._timeout_s,
                pool=self._timeout_s,
            )
        )
        self._loaded = True

        try:
            resp = self._client.get(f"{self._api_base}/models")
            resp.raise_for_status()
            logger.info(f"VLM server reachable at {self._api_base} (model='{self._model}').")
        except Exception as exc:
            logger.warning(
                f"VLM server at {self._api_base} did not answer /models ({exc}). "
                f"Continuing — queries will retry."
            )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image as a base64 JPEG data URI."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def query(self, image: Image.Image, question: str) -> str:
        """Ask a question about an image via the chat completions API.

        Args:
            image: PIL Image to analyze.
            question: Plain-English question.

        Returns:
            Raw text response from the model.

        Raises:
            RuntimeError: If the backend was never loaded.
            VLMUnavailableError: If the endpoint is unreachable or 5xx.
            httpx.HTTPError: If the request fails for any other reason.
        """
        if not self._loaded or self._client is None:
            raise RuntimeError("VLM backend is not loaded. Call load() first.")

        data = self._post(
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                # Pinned so the same crop and rule cannot answer differently
                # between runs. The transformers path fixes this with
                # do_sample=False; without this the HTTP path was the only
                # non-reproducible verdict in the appliance.
                "temperature": 0.0,
                "messages": self._messages(image, question),
            }
        )

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected VLM server response shape: {data}") from exc

    def _messages(self, image: Image.Image, question: str) -> list[dict]:
        """The one user turn, image inlined as a data URI."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(image)},
                    },
                ],
            }
        ]

    def _post(self, payload: dict) -> dict:
        """POST a chat completion and return the decoded body.

        Every request in this backend funnels through here, which makes it the
        one place that can tell "the server is gone" apart from "this query
        failed". The former is raised as :class:`VLMUnavailableError` so the
        worker can drop the whole batch instead of re-learning it per rule.

        Raises:
            VLMUnavailableError: If the endpoint is unreachable, timed out, or
                answered 5xx.
            httpx.HTTPStatusError: On a 4xx, which is a malformed request
                rather than an outage.
        """
        import httpx

        try:
            resp = self._client.post(  # type: ignore[union-attr]
                f"{self._api_base}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TransportError as exc:
            # ConnectError, ConnectTimeout and ReadTimeout all land here.
            raise VLMUnavailableError(
                f"VLM server at {self._api_base} is unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if resp.status_code >= 500:
            raise VLMUnavailableError(
                f"VLM server at {self._api_base} returned "
                f"{resp.status_code} — up but not serving."
            )

        resp.raise_for_status()
        return resp.json()

    def probe_binary(self, handle: object, question: str) -> BinaryProbe:
        """Decide a yes/no question from the server's own token probabilities.

        Asks for a single token with ``logprobs``, then reads the Yes and No
        mass out of the returned candidates. The server still runs the same
        prefill, but it stops there instead of decoding a sentence, and the
        verdict comes back calibrated rather than as prose to be parsed.

        Support is not universal, and that is the whole reason this raises a
        distinct error: **llama.cpp's server implements ``logprobs``, Ollama
        does not.** The caller is expected to latch on the first
        :class:`ProbeUnsupportedError` and stop asking for the rest of the session
        — retrying per query would pay for a useless request every time.

        Raises:
            RuntimeError: If the backend was never loaded.
            ProbeUnsupportedError: If the server returns no usable token
                probabilities, or neither word appears among the candidates.
        """
        if not self._loaded or self._client is None:
            raise RuntimeError("VLM backend is not loaded. Call load() first.")

        data = self._post(
            {
                "model": self._model,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": _TOP_LOGPROBS,
                "messages": self._messages(image=handle, question=question),  # type: ignore[arg-type]
            }
        )

        try:
            candidates = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProbeUnsupportedError(
                f"Server at {self._api_base} returned no per-token logprobs; "
                "it cannot be probed (Ollama, for example, does not implement "
                "them). Falling back to generate-and-parse."
            ) from exc

        p_yes = 0.0
        p_no = 0.0
        for entry in candidates:
            # Exact match after trimming whitespace and trailing punctuation.
            # Prefix matching would fold "not", "none" and "nothing" into the
            # No mass, which is a different claim than the model saying "No".
            word = str(entry.get("token", "")).strip().strip(".,!?;:\"'").lower()
            if word not in ("yes", "no"):
                continue
            try:
                prob = math.exp(float(entry["logprob"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if word == "yes":
                p_yes += prob
            else:
                p_no += prob

        if p_yes <= 0.0 and p_no <= 0.0:
            raise ProbeUnsupportedError(
                f"Neither 'Yes' nor 'No' appeared in the top {_TOP_LOGPROBS} "
                f"candidates from {self._model}; this model cannot be probed "
                "for a binary verdict."
            )

        return decide_probe(p_yes, p_no, self._probe_threshold)

    def probe_binary_batch(
        self, handles: list[object], questions: list[str]
    ) -> list[BinaryProbe]:
        """Probe each pair with its own request, for the reasons in
        :meth:`query_batch` — the server owns its own scheduling.
        """
        if len(handles) != len(questions):
            raise ValueError(
                f"probe_binary_batch needs one question per image, got "
                f"{len(handles)} image(s) and {len(questions)} question(s)."
            )
        return [
            self.probe_binary(handle, question)
            for handle, question in zip(handles, questions, strict=True)
        ]

    def encode(self, image: Image.Image) -> object:
        """Pass the image through — this backend has no shareable half.

        Inference happens in the server process and the image is re-sent as a
        base64 part on every request, so there is no local work to share. A
        server with prompt caching may already deduplicate this on its side.
        """
        return image

    def answer(self, handle: object, question: str) -> str:
        """Answer against the handle from :meth:`encode`, which is the image."""
        return self.query(handle, question)  # type: ignore[arg-type]

    def query_batch(
        self, images: list[Image.Image], questions: list[str]
    ) -> list[str]:
        """Answer each pair with its own request.

        Deliberately not batched. The server owns its own batching and
        scheduling, and this client is a synchronous ``httpx.Client`` driven
        from a single-worker executor, so issuing N requests concurrently from
        here would fight the scheduler rather than help it. Putting N images in
        one request is trivial on the wire but leaves no per-subject channel to
        read the answers back out of.
        """
        return sequential_query_batch(self, images, questions)

    def request_stop(self) -> None:
        """No-op — the request is bounded by the HTTP timeout instead.

        ``httpx`` offers no thread-safe way to cancel an in-flight synchronous
        request from outside, but inference runs in the server process, so a
        stuck call costs this process only a blocked socket read that
        ``timeout_s`` already bounds.
        """

    def cleanup(self) -> None:
        """No per-inference cleanup needed — the model lives elsewhere."""

    def unload(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._loaded = False
        logger.info("VLM HTTP client closed.")

    def __repr__(self) -> str:
        status = "ready" if self._loaded else "closed"
        return f"OpenAICompatBackend(base='{self._api_base}', model='{self._model}', {status})"


# ── Factory ──────────────────────────────────────────────────────


def build_backend(settings: Settings) -> VLMBackend:
    """Construct the VLM backend selected by ``EYEQ_VLM_BACKEND``.

    Args:
        settings: Application settings.

    Returns:
        An unloaded backend. Call ``load()`` from the thread that will use it.
    """
    if settings.vlm_backend == "openai_compat":
        return OpenAICompatBackend(
            api_base=settings.vlm_api_base,
            model=settings.vlm_api_model,
            api_key=settings.vlm_api_key,
            max_tokens=settings.vlm_max_tokens,
            timeout_s=settings.vlm_timeout_s,
            probe_threshold=settings.vlm_probe_threshold,
        )

    if settings.vlm_backend == "transformers":
        return TransformersVLMBackend(
            model_id=settings.vlm_model,
            device=settings.vlm_inference_device,
            max_new_tokens=settings.vlm_max_tokens,
            probe_threshold=settings.vlm_probe_threshold,
        )

    return MoondreamBackend(
        device=settings.vlm_inference_device,
        model_id=settings.vlm_model,
        revision=settings.vlm_revision,
        max_new_tokens=settings.vlm_max_tokens,
    )
