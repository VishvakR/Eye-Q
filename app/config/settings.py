"""Centralized application configuration for Eye-Q.

All settings are loaded from environment variables (prefixed with EYEQ_),
a .env file in the project root, or their default values — in that priority order.

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.inference_device)  # "mps", "cuda", or "cpu"
"""

from __future__ import annotations

import functools
import logging
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def detect_inference_device() -> str:
    """Determine the best available inference device.

    Priority: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.

    Returns:
        Device string compatible with both PyTorch and Ultralytics.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed — defaulting to CPU inference device.")
        return "cpu"

    # Apple Silicon (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Verify MPS is actually functional, not just reported as available
            _test = torch.zeros(1, device="mps")
            del _test
            logger.info("Inference device: MPS (Apple Silicon)")
            return "mps"
        except Exception as exc:
            logger.warning(f"MPS reported available but non-functional: {exc}")

    # NVIDIA CUDA
    if torch.cuda.is_available():
        logger.info(f"Inference device: CUDA ({torch.cuda.get_device_name(0)})")
        return "cuda"

    logger.info("Inference device: CPU (no GPU acceleration available)")
    return "cpu"


class Settings(BaseSettings):
    """Centralized application configuration.

    Values are loaded in this priority (highest first):
        1. Environment variables (prefixed with ``EYEQ_``)
        2. ``.env`` file in the project root
        3. Default values defined below

    Example ``.env`` file::

        EYEQ_DEFAULT_SOURCE=rtsp://192.168.1.100:554/stream1
        EYEQ_BUFFER_CAPACITY=150
        EYEQ_DEVICE=mps
    """

    model_config = SettingsConfigDict(
        env_prefix="EYEQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Video ────────────────────────────────────────────────────
    default_source: str = Field(
        default="0",
        description="Default camera source: RTSP URI, file path, or device index (as string).",
    )
    frame_width: int = Field(default=1280, ge=160, le=7680, description="Capture frame width.")
    frame_height: int = Field(default=720, ge=120, le=4320, description="Capture frame height.")
    target_fps: int = Field(default=30, ge=1, le=120, description="Target capture frame rate.")

    # ── Buffer ───────────────────────────────────────────────────
    buffer_capacity: int = Field(
        default=150,
        ge=10,
        le=3000,
        description=(
            "Number of frames in the shared-memory ring buffer. "
            "At the 1280×720 default this is ~405 MB (5s of history at 30 FPS) — "
            "sized for edge devices. 1080p×300 would need 1.77 GB."
        ),
    )

    # ── ZMQ ──────────────────────────────────────────────────────
    zmq_endpoint: str = Field(
        default="tcp://127.0.0.1:5557",
        description="ZeroMQ endpoint for Stage 1 → Stage 2 'detection' events.",
    )
    zmq_result_endpoint: str = Field(
        default="tcp://127.0.0.1:5558",
        description=(
            "ZeroMQ endpoint for Stage 2 → control-plane 'vlm_result' events. "
            "Kept separate from zmq_endpoint so each publisher owns its socket "
            "in its own thread (ZMQ sockets are not thread-safe)."
        ),
    )

    # ── YOLO (Stage 1) ───────────────────────────────────────────
    yolo_model: str = Field(
        default="models/yolov8n.pt", description="Path or name of the YOLOv8 model."
    )
    yolo_confidence: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Minimum detection confidence."
    )
    yolo_imgsz: int = Field(default=640, description="YOLO input image size.")
    detection_cooldown_s: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Minimum seconds between triggers for the same subject. With "
            "tracking on this is per instance, so one worker's cooldown never "
            "suppresses the worker standing beside them."
        ),
    )

    watchdog_fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description=(
            "Maximum rate at which Stage 1 runs YOLO. The main CPU lever on a "
            "board with no GPU: the watchdog is paced independently of "
            "capture, so left at 30 it will use every cycle it can get and "
            "starve the VLM and the API. There is no benefit to setting it "
            "above the camera's frame rate — repeated frames are skipped "
            "without inference either way."
        ),
    )

    # ── Tracking (Stage 1) ───────────────────────────────────────
    track_enabled: bool = Field(
        default=True,
        description=(
            "Assign stable per-instance ids so every subject in a frame can be "
            "judged, not just the first of its label. Set false to restore the "
            "original one-per-label behaviour exactly."
        ),
    )
    track_iou_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum IoU for a detection to continue an existing track. Set "
            "deliberately low: a subject who is not matched gets a fresh id and "
            "is judged again, so a strict threshold shows up as wasted "
            "inferences. Measured on the bundled 6 FPS clip, 0.45 produced 4.2 "
            "ids per real person against 2.0 at this value. Two people in one "
            "frame can never share an id whatever this is set to, so lowering "
            "it cannot hide a subject. Raise it for high-frame-rate cameras "
            "where boxes barely move between frames."
        ),
    )
    track_max_age_frames: int = Field(
        default=60,
        ge=1,
        description=(
            "Frames a track survives unmatched before its id is retired. Long "
            "enough to ride out a subject passing behind an obstacle — who "
            "would otherwise return as a stranger and cost a fresh inference — "
            "and short enough that a departed subject stops costing memory."
        ),
    )

    yolo_pair_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How much of the smaller box must lie inside the larger for a "
            "rule's pair_label to count as a co-occurrence, measured as "
            "BoundingBox.overlap_ratio — intersection over the *smaller* "
            "area.\n\n"
            "Explicitly not IoU. A cell phone fully inside a person's box "
            "scores about 0.028 on IoU, not because the phone is barely inside "
            "but because the person is roughly thirty-six times larger and IoU "
            "divides by the union. Any IoU threshold loose enough to admit "
            "that pair also admits two boxes that merely graze at a corner. "
            "Dividing by the smaller area asks the question the operator means "
            "— 'is the phone in the person?' — and returns ~1.0 for full "
            "containment at any size ratio.\n\n"
            "0.5 reads as 'at least half the smaller object is inside the "
            "larger'. Lower it toward 0 for partners YOLO boxes loosely; "
            "raising it above ~0.9 demands near-total containment, which a "
            "phone held out to the side will fail. 0.0 accepts any touch at "
            "all, which on generous axis-aligned boxes is close to no gate — "
            "a phone on a desk two metres behind someone clips their box by a "
            "pixel, and at 30 FPS that is a firehose."
        ),
    )

    # ── VLM (Stage 2) ───────────────────────────────────────────
    vlm_backend: Literal[
        "moondream4bit", "moondream", "transformers", "openai_compat"
    ] = Field(
        default="moondream4bit",
        description=(
            "Which VLM backend to use. All four run locally — no cloud. "
            "'moondream4bit' — the default — loads the 4-bit 2025-04-14 "
            "Moondream in-process: a third the download of the fp32 revision, "
            "and the only in-process backend that can read a verdict off the "
            "prefill logits instead of generating text (see vlm_probe). "
            "'moondream' loads Moondream2 at the pinned 2024-08-26 revision "
            "in fp32 via its older bespoke API; it is the baseline the 4-bit "
            "backend is measured against and stays available if that "
            "comparison ever goes the wrong way on your hardware. "
            "'transformers' loads any chat-templated vision-to-text model "
            "in-process (Qwen2-VL-2B, SmolVLM, MiniCPM-V) — set vlm_model to "
            "the HF id. 'openai_compat' calls a local OpenAI-compatible server "
            "(Ollama, llama.cpp, vLLM), usually the fastest option on edge CPUs "
            "and it keeps model weights out of this process."
        ),
    )
    vlm_model: str = Field(
        default="moondream/moondream-2b-2025-04-14-4bit",
        description=(
            "Hugging Face model ID for the VLM. Set this and vlm_revision "
            "together — they are read by whichever in-process backend is "
            "selected, and the two Moondream backends want different pairs."
        ),
    )
    vlm_api_base: str = Field(
        default="http://127.0.0.1:11434/v1",
        description="Base URL of the local OpenAI-compatible VLM server (openai_compat backend).",
    )
    vlm_api_model: str = Field(
        default="moondream",
        description="Model name to request from the OpenAI-compatible server.",
    )
    vlm_api_key: str = Field(
        default="EMPTY", description="API key for the local VLM server (usually unused)."
    )
    vlm_max_tokens: int = Field(
        default=2,
        ge=1,
        description=(
            "Maximum tokens the VLM may generate per query. Verdicts are "
            "binary and BinaryVerdictParser only reads the leading word, so "
            "2 is enough for 'Yes' or 'No' plus its punctuation, and the rest "
            "is a worst-case guard against a model that ignores the "
            "'answer Yes or No' instruction and rambles — each extra token is "
            "another sequential forward pass. This is not a latency lever: it "
            "does not speed up a model that already stops at EOS, and Stage 2 "
            "is ~99% prefill regardless. It is moot on the probe path, which "
            "generates nothing at all. Raise it if a backend prefixes its "
            "answer, because a truncated leading word parses as 'no alert' "
            "and disappears silently."
        ),
    )
    vlm_revision: str = Field(
        default="a89c59223ef8b5bb7826780728eeec172727ca84",
        description=(
            "Model revision/commit for reproducibility. Pinned to a sha rather "
            "than a branch because the moondream4bit backend patches the "
            "repo's remote code — it replaces QuantizedLinear.unpack to skip a "
            "CUDA-only torchao kernel — and a moving revision could change "
            "what is being patched without anything failing loudly. "
            "vlm_backend='moondream' wants '2024-08-26' instead."
        ),
    )
    vlm_compute_dtype: Literal["", "float32", "float16", "bfloat16"] = Field(
        default="",
        description=(
            "Dtype the moondream4bit backend dequantizes its packed weights "
            "to. Empty — the default — picks per device: fp32 on CPU, fp16 on "
            "MPS, bf16 on CUDA. The CPU choice is the load-bearing one: torch "
            "has no Accelerate-backed bf16 GEMM on ARM, and Stage 2 is "
            "prefill-bound, so upstream's bf16 would buy memory at the "
            "expense of the thing being optimized. Override only against a "
            "bench/benchmark_vlm.py --compute-dtype sweep on your own board. "
            "Ignored by every other backend."
        ),
    )
    vlm_timeout_s: float = Field(
        default=60.0,
        ge=1.0,
        description=(
            "Hard cap on a single VLM inference. Measured ~8s for Moondream2 "
            "on Apple Silicon CPU, and edge CPUs are slower still, so this is "
            "a runaway guard rather than a latency target. Lower it well below "
            "60s only when running the openai_compat backend on a GPU."
        ),
    )
    vlm_staleness_s: float = Field(
        default=5.0,
        ge=0.5,
        description="Discard detection triggers older than this many seconds.",
    )
    vlm_default_prompt: str | None = Field(
        default=None,
        description=(
            "Optional fallback question asked when a detection matches no "
            "rule. Unset by default: an appliance should not spend inferences "
            "on questions the operator never asked, and the resulting alerts "
            "are unattributable. Set it only to demo the cascade with no "
            "rules configured."
        ),
    )
    vlm_crop_padding: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Extra padding around bounding box as fraction of bbox size.",
    )
    vlm_focus_dim: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Brightness multiplier applied outside the subject's own box "
            "within its crop. The padding above exists to give the VLM spatial "
            "context, but it also drags in whoever is standing nearby — and a "
            "helmeted neighbour in the frame can flip the verdict on a "
            "bare-headed subject. Dimming keeps the context readable while "
            "leaving no doubt who the question is about. 1.0 disables. "
            "Defaults to disabled: dimming is a synthetic edit no VLM saw in "
            "training, and vlm_square_edge now bounds the crop so context is "
            "cheaper than it was when this was introduced. The mechanism is "
            "kept because the failure it fixed is real — set this to 0.4 to "
            "restore it if crowded scenes start answering about the wrong "
            "person."
        ),
    )
    vlm_verdict_ttl_s: float = Field(
        default=300.0,
        ge=0.0,
        description=(
            "Seconds a per-instance verdict is reused before that subject is "
            "judged again. This is the appliance's load control, and the "
            "governing arithmetic is duty_cycle = subjects x inference_s / ttl "
            "— at ~6s per query on CPU, five subjects and a 30s TTL would "
            "saturate the VLM completely. Keep it well above 2 x subjects x "
            "inference_s. 0 disables reuse (every trigger is judged)."
        ),
    )
    vlm_rejudge_iou: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Re-judge a subject early when IoU against the box it was last "
            "judged in falls below this — they have moved far enough that the "
            "old verdict may no longer describe them."
        ),
    )
    vlm_min_rejudge_s: float = Field(
        default=20.0,
        ge=0.0,
        description=(
            "Floor between re-judgements of one subject, however far they "
            "moved. Without it a single walker re-judges roughly every 0.6s as "
            "their box slides off its own footprint, which alone would "
            "saturate an edge CPU."
        ),
    )
    vlm_max_instances_per_batch: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum subjects inferred from one drained batch. Never-judged "
            "subjects are taken first, so a crowd delays re-checks rather than "
            "first looks."
        ),
    )

    vlm_max_crop_edge: int = Field(
        default=511,
        ge=0,
        description=(
            "Longest side, in pixels, of the crop handed to the VLM. This "
            "tracks a cliff in Moondream's preprocessor, not a quality dial: "
            "at 512 px and above it stops encoding one 378 px image and starts "
            "encoding the image plus 2-4 tiles, which measured 3.16 s against "
            "1.42 s for the same one-word answer. Most crops are already well "
            "under it; the case it catches is vlm_crop_padding falling back to "
            "the whole frame for a subject filling most of it. Raise it only if "
            "you switch to a backend with no such threshold, and 0 disables."
        ),
    )

    vlm_batch_size: int = Field(
        default=1,
        ge=1,
        description=(
            "How many crops the VLM may answer about in one forward pass. "
            "1 — the default — dispatches one query at a time, which is what "
            "this appliance has always done. Raising it is measured, not "
            "assumed: Stage 2 carries 729 image tokens against a one-token "
            "answer, making it ~99% prefill, and prefill is compute-bound at "
            "roughly 280 FLOP/byte. Batching amortizes a weight read that "
            "prefill is not waiting on, so it buys little — while the KV cache "
            "grows linearly with this value (~295 MB per sequence for "
            "Moondream at fp32). Measure with bench/benchmark_vlm.py before "
            "raising it, and never raise it on a memory-bound board. The "
            "default moondream4bit backend cannot batch at all — its "
            "revision's batch_answer() is a sequential loop upstream — so "
            "there this only groups queries, it does not fuse them."
        ),
    )
    vlm_min_subject_px: int = Field(
        default=96,
        ge=0,
        description=(
            "Skip a detection whose own box, before padding, is smaller than "
            "this on its longer side. Governs BOTH stages: Stage 1 sheds "
            "against it before publishing (watchdog.detections_filtered_size) "
            "and Stage 2 keeps the same test as a backstop. One setting on "
            "purpose — both call BoundingBox.pixel_size on the same "
            "fixed-shape ring buffer, so the predicate is literally the same "
            "function and the two floors cannot drift by a pixel. The "
            "consequence to know when reading /api/status: "
            "vlm.instances_too_small should now read ZERO, which is the "
            "healthy steady state and not a regression — Stage 1 sheds first. "
            "A non-zero value there means something other than this watchdog "
            "is publishing on the detection endpoint. "
            "Such a subject cannot support the question: "
            "demo_cctv.py already records that at this camera distance a head "
            "is about 20 px and Moondream answers 'No' to every phrasing of a "
            "hard-hat question regardless of what is on the worker's head. "
            "Before this existed the crop fell back to the whole frame, so the "
            "least answerable subject bought the most expensive inference and "
            "the answer was filed against that person's box anyway. 0 disables. "
            "Raised from 24 to 96 to discard crops no model can solve — but "
            "note what it trades: a skipped subject is never judged at all, so "
            "this buys precision with recall. It is also measured in "
            "*captured* pixels, so it interacts with frame resolution: over "
            "cctv_demo.mp4 a floor of 96 skips 6.5% of detections at 1920x1080 "
            "but 38.7% at the 960x540 .env.pi5 profile, where 64 is the right "
            "value. Watch instances_too_small on /api/status after changing "
            "it; if it swallows most detections at your camera distance, lower "
            "it rather than accepting the blind spot."
        ),
    )

    vlm_square_edge: int = Field(
        default=378,
        ge=0,
        description=(
            "Edge, in pixels, of the square canvas each crop is letterboxed "
            "into before it reaches the VLM. Subjects are tall and narrow — "
            "the demo's crops run about 1:3 — and a model that resizes to a "
            "square input stretches them, so a person arrives roughly three "
            "times too wide. Letterboxing scales the long side to fit and pads "
            "the short one, preserving the shape the model was trained on. "
            "Because every crop then lands at the same size, the image-token "
            "count is constant and prefill cost stops varying with subject "
            "geometry. 378 is Moondream's native SigLIP edge and stays under "
            "the 512 px tiling cliff described on vlm_max_crop_edge; 392 "
            "(14x28) lands exactly on Qwen2-VL's 28 px grid and avoids its "
            "smart_resize rounding. 0 disables letterboxing and falls back to "
            "vlm_max_crop_edge downscaling."
        ),
    )

    # ── Verdict Probing ──────────────────────────────────────────
    vlm_probe: Literal["auto", "on", "off"] = Field(
        default="auto",
        description=(
            "Whether to read the verdict from the model's token probabilities "
            "instead of generating an answer and parsing it. A Stage 2 query "
            "is ~99% prefill and needs exactly one bit back, so generating "
            "prose spends a full forward pass per token on text that a string "
            "parser then discards — and that parser defaults to 'no alert' on "
            "anything it fails to recognise, turning a malformed answer into a "
            "silent false negative. 'auto' probes wherever the backend can and "
            "falls back permanently on the first backend that cannot: the "
            "moondream4bit and transformers backends always can, llama.cpp "
            "can via logprobs, Ollama and the pinned 2024-08-26 Moondream "
            "wrapper cannot. 'off' forces "
            "generate-and-parse; 'on' is the same as 'auto' but logs the "
            "fallback as a warning rather than an informational message."
        ),
    )
    vlm_probe_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Yes-probability above which a probed verdict counts as a "
            "violation, after renormalizing over the Yes/No pair alone. 0.5 is "
            "the neutral comparison and matches what generate-and-parse did "
            "implicitly. Raise it to make alerts harder to trigger — this is "
            "the dial the old text path never offered, because a parsed 'Yes' "
            "carried no strength."
        ),
    )

    # ── Inference Device ─────────────────────────────────────────
    device: Literal["auto", "mps", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Inference device for YOLO. 'auto' detects MPS → CUDA → CPU, then "
            "yields the accelerator to an in-process VLM and takes CPU — two "
            "threads on one MPS device segfaults PyTorch, and Stage 1 pays "
            "~10 ms/frame for the swap where Stage 2 would pay ~10 s/verdict. "
            "Set this explicitly to opt out of that arbitration."
        ),
    )
    vlm_device: Literal["auto", "mps", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Inference device for the VLM. 'auto' probes MPS -> CUDA -> CPU "
            "and wins it from Stage 1, which drops to CPU — see "
            "Settings.inference_device for why that arbitration exists and "
            "what it costs. "
            "This used to default to 'cpu' on the measurement that Moondream2 "
            "at the 2024-08-26 revision ran no faster on MPS (6.1s vs 5.6s per "
            "query) and loaded ~10s slower, so the accelerator only cost "
            "unified memory Stage 1 wanted. That finding does not carry to the "
            "2025-04-14 model now behind moondream4bit, whose vision tower "
            "dominates its query: measured on an M2 over one real crop, "
            "encode_image takes 7,313 ms on CPU against 1,672 ms on MPS at "
            "fp16. Set 'cpu' explicitly to restore the old behaviour, and note "
            "that vlm_backend='moondream' is still better off there."
        ),
    )

    # ── API Server ───────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host.")
    api_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI bind port.")

    # ── Pipeline Toggles ─────────────────────────────────────────
    # These are set by main.py's CLI flags via environment variables before
    # get_settings() is first called, so the lifespan can honour them.
    enable_pipeline: bool = Field(
        default=True,
        description="Start capture + Stage 1 + Stage 2. False = API/dashboard only.",
    )
    enable_vlm: bool = Field(
        default=True,
        description="Start the Stage 2 VLM worker. False = Stage 1 only (no model load).",
    )

    # ── Database & Media ─────────────────────────────────────────
    database_path: str = Field(
        default="data/eyeq.db", description="SQLite database file path."
    )
    snapshot_dir: str = Field(
        default="data/snapshots",
        description="Directory where alert evidence JPEGs are written.",
    )
    stream_fps: float = Field(
        default=12.0,
        gt=0.0,
        le=30.0,
        description="Frame rate of the MJPEG dashboard preview stream.",
    )
    stream_width: int = Field(
        default=1280, ge=160, description="Max width of MJPEG preview frames (downscaled)."
    )
    stream_quality: int = Field(
        default=70, ge=10, le=100, description="JPEG quality for the MJPEG preview stream."
    )

    # ── Webhooks ─────────────────────────────────────────────────
    webhook_url: str | None = Field(
        default=None, description="Optional webhook URL for alert dispatch."
    )
    webhook_timeout_s: float = Field(
        default=5.0, ge=0.5, description="Webhook HTTP request timeout."
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Application log level."
    )

    # Backends that run the model inside this process, and therefore contend
    # with Stage 1 for an accelerator. ``openai_compat`` holds its weights in
    # another process and never does.
    _IN_PROCESS_VLM_BACKENDS = ("moondream4bit", "moondream", "transformers")

    def _accelerator_contended(self) -> bool:
        """Whether both stages would land on the same accelerator by default.

        Only true when *neither* device was set explicitly. An operator who
        names a device gets that device, and a warning if the pairing is the
        unsafe one.
        """
        return (
            self.device == "auto"
            and self.vlm_device == "auto"
            and self.vlm_backend in self._IN_PROCESS_VLM_BACKENDS
            and detect_inference_device() != "cpu"
        )

    @property
    def inference_device(self) -> str:
        """Resolve Stage 1's device, yielding the accelerator to Stage 2.

        PyTorch's MPS backend is not safe to drive from two threads at once.
        With Stage 1 and an in-process VLM both on MPS, this process segfaults
        inside ``_mps_convolution_impl`` — one thread building an ``MPSGraph``
        while the other tears one down — usually before the VLM has even
        finished loading. It is a hard crash, not an exception: no traceback,
        no shutdown hook, and the ring buffer's shared memory is left behind.

        So the two `auto`s are arbitrated here rather than resolved
        independently, and the accelerator goes to Stage 2. The measurement is
        lopsided enough that it is not a close call — on an M2, per call:

        | | MPS | CPU | cost of losing it |
        |---|---|---|---|
        | Stage 1 (YOLOv8n @720p) | 16.9 ms | 26.8 ms | **9.9 ms** |
        | Stage 2 (Moondream 2B 4-bit) | 1,873 ms | 11,712 ms | **9,839 ms** |

        Stage 1 gives up ten milliseconds a frame; Stage 2 would give up ten
        seconds a verdict. Setting ``EYEQ_DEVICE`` explicitly opts out.
        """
        if self.device != "auto":
            return self.device
        if self._accelerator_contended():
            return "cpu"
        return detect_inference_device()

    @property
    def vlm_inference_device(self) -> str:
        """Resolve the VLM device. Falls back to CPU inside the backend's load().

        Wins the arbitration described on :attr:`inference_device`. Warns when
        an explicit pairing recreates the crash rather than silently overriding
        it — an operator who named both devices meant both devices.
        """
        resolved = (
            detect_inference_device() if self.vlm_device == "auto" else self.vlm_device
        )
        if (
            resolved != "cpu"
            and self.vlm_backend in self._IN_PROCESS_VLM_BACKENDS
            and self.device != "auto"
            and self.device == resolved
        ):
            logger.warning(
                f"EYEQ_DEVICE and EYEQ_VLM_DEVICE are both '{resolved}'. Driving "
                "it from Stage 1 and an in-process VLM at the same time "
                "segfaults inside PyTorch's MPS backend. Set EYEQ_DEVICE=cpu, "
                "or EYEQ_VLM_BACKEND=openai_compat to move the weights out of "
                "this process."
            )
        return resolved

    @property
    def video_source(self) -> str | int:
        """Parse ``default_source`` into an integer device index or a string URI/path."""
        try:
            return int(self.default_source)
        except ValueError:
            return self.default_source


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    Call this from anywhere in the codebase to access configuration.
    The instance is created once and reused for the process lifetime.
    """
    return Settings()
