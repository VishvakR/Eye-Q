"""Stage 2 — VLM Inference Worker: wake-on-event visual question answering.

This module runs in a dedicated daemon thread, listening for detection
triggers from the Stage 1 YOLO Watchdog via ZMQ. When triggered, it:
    1. Fetches the high-resolution frame from the ring buffer by timestamp
    2. Crops the image to the detected bounding box (with padding)
    3. Queries the VLM (Moondream2) with each matching rule's plain-English prompt
    4. Parses the response into a binary Yes/No verdict
    5. Publishes results and fires alerts for positive verdicts

Key design constraints:
    - **Staleness protection**: Triggers older than N seconds are discarded
    - **Memory cleanup**: ``torch.mps.empty_cache()`` after each inference
    - **Timeout enforcement**: No single inference exceeds the configured timeout
    - **Non-blocking**: VLM latency (500ms–2s) never blocks the video pipeline

Example:
    >>> worker = VLMWorker(
    ...     buffer=buffer,
    ...     zmq_endpoint="tcp://127.0.0.1:5557",
    ...     device="mps",
    ... )
    >>> worker.start()
    >>> # ... runs in background, processes triggers from YOLO ...
    >>> worker.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.core.buffer import RollingFrameBuffer
from app.core.ipc import EventPublisher, EventSubscriber
from app.pipelines.vlm_judge import BinaryVerdictParser
from app.schemas.events import BoundingBox, DetectionEvent, VLMQuery, VLMResult

logger = logging.getLogger(__name__)


# ── Frame Cropping ───────────────────────────────────────────────


def crop_detection(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    """Crop the frame to the bounding box with percentage-based padding.

    Padding adds context around the detection, which significantly
    improves VLM comprehension. The model needs spatial context to
    understand relationships like "blocking the exit" or "wearing a
    hard hat."

    Args:
        frame: Full-resolution frame as numpy array (H, W, C).
        bbox: Normalized bounding box coordinates.
        padding_ratio: Extra padding as fraction of bbox dimensions.
                       0.15 = 15% padding on each side.

    Returns:
        Cropped region as a new numpy array. Returns the full frame
        if the bbox covers most of the frame already.
    """
    h, w = frame.shape[:2]

    # If bbox covers >80% of the frame, return full frame
    if bbox.area > 0.8:
        return frame.copy()

    # Denormalize to pixel coordinates
    x1 = int(bbox.x1 * w)
    y1 = int(bbox.y1 * h)
    x2 = int(bbox.x2 * w)
    y2 = int(bbox.y2 * h)

    # Add padding
    pad_w = int((x2 - x1) * padding_ratio)
    pad_h = int((y2 - y1) * padding_ratio)

    # Clamp to frame bounds
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    # Ensure minimum crop size (at least 32×32)
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return frame.copy()

    return frame[y1:y2, x1:x2].copy()


# ── Moondream2 VLM Wrapper ───────────────────────────────────────


class MoondreamVLM:
    """Wrapper around vikhyatk/moondream2 for visual question answering.

    Moondream2 is a tiny (1.86B params) vision-language model optimized
    for edge deployment. It supports:
        - Visual Question Answering (VQA) — used here
        - Image Captioning
        - Object Detection
        - Visual Pointing

    The model is loaded lazily on first use and can be explicitly
    unloaded to free memory.

    Args:
        device: PyTorch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
        model_id: Hugging Face model identifier.
        revision: Model revision for reproducibility.
    """

    def __init__(
        self,
        device: str = "mps",
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2025-06-21",
    ) -> None:
        self._device = device
        self._model_id = model_id
        self._revision = revision
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        This can take 10–30 seconds on first run (downloads ~3.7GB).
        Subsequent loads are fast if the model is cached locally.
        """
        if self._loaded:
            return

        logger.info(
            f"Loading VLM model: {self._model_id} (revision={self._revision}) "
            f"on device '{self._device}'..."
        )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                trust_remote_code=True,
                revision=self._revision,
                device_map={"": self._device},
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                revision=self._revision,
            )
            self._loaded = True

            logger.info(
                f"VLM model loaded successfully on '{self._device}'. "
                f"Model ID: {self._model_id}"
            )

        except Exception as exc:
            logger.error(f"Failed to load VLM model: {exc}")

            # Try CPU fallback
            if self._device != "cpu":
                logger.warning("Retrying VLM load on CPU...")
                self._device = "cpu"
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    self._model = AutoModelForCausalLM.from_pretrained(
                        self._model_id,
                        trust_remote_code=True,
                        revision=self._revision,
                        device_map={"": "cpu"},
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_id,
                        revision=self._revision,
                    )
                    self._loaded = True
                    logger.info("VLM model loaded on CPU (fallback).")
                except Exception as cpu_exc:
                    logger.error(f"VLM CPU fallback also failed: {cpu_exc}")
                    raise

    def query(self, image: Image.Image, question: str) -> str:
        """Ask a yes/no question about an image.

        Args:
            image: PIL Image to analyze.
            question: Plain-English question (e.g., "Is a pallet blocking
                      the fire exit?").

        Returns:
            Raw text response from the VLM.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("VLM model is not loaded. Call load() first.")

        result = self._model.query(image, question)
        return result["answer"]

    def cleanup(self) -> None:
        """Release GPU memory after inference.

        On Apple MPS, this calls ``torch.mps.empty_cache()`` to free
        the Metal command buffer pool. On CUDA, it calls the equivalent.
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

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    @property
    def device(self) -> str:
        """Current inference device."""
        return self._device

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"MoondreamVLM(model='{self._model_id}', "
            f"device='{self._device}', {status})"
        )


# ── Worker Statistics ────────────────────────────────────────────


@dataclass
class VLMWorkerStats:
    """Snapshot of VLM worker state for monitoring dashboards."""

    is_running: bool
    model_loaded: bool
    device: str
    triggers_received: int
    triggers_processed: int
    triggers_stale: int
    triggers_failed: int
    verdicts_positive: int
    verdicts_negative: int
    avg_inference_ms: float
    last_inference_ms: float
    uptime_seconds: float


# ── VLM Worker ───────────────────────────────────────────────────


class VLMWorker:
    """Stage 2 wake-on-event VLM inference worker.

    Runs in a dedicated daemon thread, listening for detection triggers
    from the YOLO watchdog via ZMQ. On each trigger:
        1. Check staleness — discard if too old
        2. Fetch the frame from the ring buffer by timestamp
        3. Crop to the bounding box with padding
        4. Query the VLM with the configured prompt(s)
        5. Parse the response into a binary verdict
        6. Publish the result

    Args:
        buffer: Shared-memory ring buffer to read frames from.
        zmq_endpoint: ZMQ endpoint to subscribe to detection events.
        publisher: Optional ZMQ publisher for VLM results. If None,
                   results are only logged.
        device: Inference device (``"mps"``, ``"cuda"``, ``"cpu"``).
        model_id: Hugging Face model ID for the VLM.
        model_revision: Model revision for reproducibility.
        staleness_threshold_s: Discard triggers older than this (seconds).
        crop_padding: Extra padding around bbox as fraction of bbox size.
        default_prompt: Default prompt used when no rules are configured.
    """

    def __init__(
        self,
        buffer: RollingFrameBuffer,
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        publisher: EventPublisher | None = None,
        device: str = "cpu",
        model_id: str = "vikhyatk/moondream2",
        model_revision: str = "2025-06-21",
        staleness_threshold_s: float = 5.0,
        crop_padding: float = 0.15,
        default_prompt: str = "Is there anything unusual or potentially dangerous in this image? Answer with Yes or No and explain briefly.",
    ) -> None:
        self._buffer = buffer
        self._zmq_endpoint = zmq_endpoint
        self._publisher = publisher
        self._staleness_threshold = staleness_threshold_s
        self._crop_padding = crop_padding
        self._default_prompt = default_prompt

        # VLM model (loaded lazily in worker thread)
        self._vlm = MoondreamVLM(
            device=device,
            model_id=model_id,
            revision=model_revision,
        )

        # Verdict parser
        self._verdict_parser = BinaryVerdictParser()

        # Prompt rules: mapping of label -> list of prompts
        # In Phase 4, this will be loaded from SQLite
        self._rules: dict[str, list[dict]] = {}

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics
        self._triggers_received = 0
        self._triggers_processed = 0
        self._triggers_stale = 0
        self._triggers_failed = 0
        self._verdicts_positive = 0
        self._verdicts_negative = 0
        self._inference_times: list[float] = []
        self._last_inference_ms: float = 0.0
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def add_rule(
        self,
        label: str,
        prompt: str,
        rule_id: int = 0,
        rule_name: str = "",
    ) -> None:
        """Register a prompt rule for a specific detection label.

        When a detection with ``label`` is received, the VLM will be
        queried with ``prompt``.

        Args:
            label: COCO class label to match (e.g., ``"person"``).
            prompt: Plain-English question for the VLM.
            rule_id: Optional database ID for the rule.
            rule_name: Optional human-readable name.
        """
        if label not in self._rules:
            self._rules[label] = []

        self._rules[label].append({
            "rule_id": rule_id,
            "rule_name": rule_name,
            "prompt": prompt,
        })

        logger.info(
            f"VLM rule added: label='{label}', prompt='{prompt[:60]}...'"
        )

    def clear_rules(self) -> None:
        """Remove all registered prompt rules."""
        self._rules.clear()
        logger.info("All VLM rules cleared.")

    def start(self) -> None:
        """Start the VLM worker in a dedicated daemon thread.

        Raises:
            RuntimeError: If the worker is already running.
        """
        if self.is_running:
            raise RuntimeError("VLMWorker is already running.")

        self._shutdown_event.clear()
        self._start_time = time.monotonic()

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="VLMWorker",
            daemon=True,
        )
        self._thread.start()

        logger.info("VLMWorker started.")

    def stop(self, timeout: float = 15.0) -> None:
        """Signal the worker to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait (VLM inference can be slow).
        """
        if not self.is_running:
            return

        logger.info("Stopping VLMWorker...")
        self._shutdown_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"VLMWorker did not stop within {timeout}s "
                    f"(may be mid-inference)."
                )

        self._thread = None
        logger.info(
            f"VLMWorker stopped. "
            f"Processed {self._triggers_processed} triggers, "
            f"positive={self._verdicts_positive}, "
            f"negative={self._verdicts_negative}, "
            f"stale={self._triggers_stale}."
        )

    @property
    def stats(self) -> VLMWorkerStats:
        """Current worker statistics snapshot."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        avg_ms = (
            sum(self._inference_times[-50:]) / len(self._inference_times[-50:])
            if self._inference_times
            else 0.0
        )

        return VLMWorkerStats(
            is_running=self.is_running,
            model_loaded=self._vlm.is_loaded,
            device=self._vlm.device,
            triggers_received=self._triggers_received,
            triggers_processed=self._triggers_processed,
            triggers_stale=self._triggers_stale,
            triggers_failed=self._triggers_failed,
            verdicts_positive=self._verdicts_positive,
            verdicts_negative=self._verdicts_negative,
            avg_inference_ms=avg_ms,
            last_inference_ms=self._last_inference_ms,
            uptime_seconds=uptime,
        )

    # ── Internal Worker Loop ─────────────────────────────────────

    def _worker_loop(self) -> None:
        """Main worker loop running in a dedicated thread."""
        # Load the VLM model (this takes 10-30s on first run)
        try:
            self._vlm.load()
        except Exception as exc:
            logger.error(f"VLMWorker: failed to load model: {exc}")
            return

        # Create ZMQ subscriber (in the worker thread for thread safety)
        subscriber = EventSubscriber(
            self._zmq_endpoint,
            topic="detection",
        )

        try:
            while not self._shutdown_event.is_set():
                # Wait for a detection trigger (1s timeout for shutdown checks)
                msg = subscriber.receive(timeout_ms=1000)
                if msg is None:
                    continue

                topic, payload = msg
                self._triggers_received += 1

                try:
                    self._process_trigger(payload)
                except Exception as exc:
                    self._triggers_failed += 1
                    logger.error(
                        f"VLMWorker: trigger processing failed: {exc}",
                        exc_info=True,
                    )
                finally:
                    # Always clean up GPU memory
                    self._vlm.cleanup()

        finally:
            subscriber.close()
            self._vlm.unload()

    def _process_trigger(self, payload: dict) -> None:
        """Process a single detection trigger.

        Steps:
            1. Parse the detection event
            2. Check staleness
            3. Fetch frame from buffer
            4. Crop to bounding box
            5. Get matching rules (or use default prompt)
            6. Query VLM for each prompt
            7. Parse verdict and publish results
        """
        # Parse event
        event = DetectionEvent(**payload)

        # ── Staleness check ──────────────────────────────────────
        age = time.monotonic() - event.timestamp
        if age > self._staleness_threshold:
            self._triggers_stale += 1
            logger.warning(
                f"Discarding stale trigger: {event.label} "
                f"from {age:.1f}s ago (threshold: {self._staleness_threshold}s)"
            )
            return

        # ── Fetch frame from buffer ──────────────────────────────
        frame_result = self._buffer.read_by_timestamp(
            event.timestamp, tolerance_ms=500.0
        )

        if frame_result is None:
            # Try direct index access as fallback
            try:
                frame, ts = self._buffer.read_at_index(event.frame_index)
                frame_result = (frame, ts, event.frame_index)
            except (IndexError, RuntimeError):
                self._triggers_failed += 1
                logger.warning(
                    f"Frame not found for trigger at t={event.timestamp:.3f} "
                    f"(index={event.frame_index}). Buffer may have wrapped."
                )
                return

        frame, actual_ts, slot_idx = frame_result

        # ── Crop to bounding box ─────────────────────────────────
        cropped = crop_detection(frame, event.bbox, self._crop_padding)

        # Convert BGR (OpenCV) to RGB (PIL)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)

        # ── Get matching rules ───────────────────────────────────
        rules = self._rules.get(event.label, [])

        if not rules:
            # Use default prompt if no specific rules are configured
            rules = [{
                "rule_id": 0,
                "rule_name": "default",
                "prompt": self._default_prompt,
            }]

        # ── Query VLM for each rule ──────────────────────────────
        for rule in rules:
            prompt = rule["prompt"]
            rule_id = rule["rule_id"]
            rule_name = rule.get("rule_name", "")

            logger.info(
                f"VLM query: [{event.label}] '{prompt[:80]}...' "
                f"(rule_id={rule_id})"
            )

            # Run inference with timing
            start_time = time.monotonic()
            try:
                raw_response = self._vlm.query(pil_image, prompt)
            except Exception as exc:
                self._triggers_failed += 1
                logger.error(f"VLM inference failed: {exc}")
                continue
            inference_ms = (time.monotonic() - start_time) * 1000.0

            self._last_inference_ms = inference_ms
            self._inference_times.append(inference_ms)
            # Keep only last 100 timings
            if len(self._inference_times) > 100:
                self._inference_times = self._inference_times[-100:]

            # Parse verdict
            verdict = self._verdict_parser.parse(raw_response)
            self._triggers_processed += 1

            if verdict:
                self._verdicts_positive += 1
            else:
                self._verdicts_negative += 1

            # Create result
            query = VLMQuery(
                detection=event,
                rule_id=rule_id,
                rule_name=rule_name,
                prompt=prompt,
            )

            result = VLMResult(
                query=query,
                verdict=verdict,
                raw_response=raw_response,
                inference_time_ms=inference_ms,
                model_name=self._vlm._model_id,
            )

            # Log result
            verdict_str = "🔴 YES (VIOLATION)" if verdict else "🟢 NO (clear)"
            logger.info(
                f"VLM result: {verdict_str} | "
                f"inference={inference_ms:.0f}ms | "
                f"response='{raw_response[:100]}...'"
            )

            # Publish result via ZMQ
            if self._publisher is not None:
                self._publisher.publish(
                    "vlm_result",
                    result.model_dump(),
                )

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"VLMWorker({status}, "
            f"processed={self._triggers_processed}, "
            f"positive={self._verdicts_positive}, "
            f"negative={self._verdicts_negative})"
        )
