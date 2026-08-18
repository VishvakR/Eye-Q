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
import threading
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
    ) -> None:
        self._device = device
        self._model_id = model_id
        self._revision = revision
        self._cache_dir = cache_dir
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
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        enc_image = self._model.encode_image(image)
        return self._model.answer_question(enc_image, question, self._tokenizer)

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
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._loaded = False
        self._stop_event = threading.Event()

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
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
        timeout_s: Per-request HTTP timeout.
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434/v1",
        model: str = "moondream",
        api_key: str = "EMPTY",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._api_key = api_key or "EMPTY"
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
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

        self._client = httpx.Client(timeout=self._timeout_s)
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
            httpx.HTTPError: If the request fails.
        """
        if not self._loaded or self._client is None:
            raise RuntimeError("VLM backend is not loaded. Call load() first.")

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
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
            ],
        }

        resp = self._client.post(
            f"{self._api_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected VLM server response shape: {data}") from exc

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
        )

    if settings.vlm_backend == "transformers":
        return TransformersVLMBackend(
            model_id=settings.vlm_model,
            device=settings.vlm_inference_device,
            max_new_tokens=settings.vlm_max_tokens,
        )

    return MoondreamBackend(
        device=settings.vlm_inference_device,
        model_id=settings.vlm_model,
        revision=settings.vlm_revision,
    )
