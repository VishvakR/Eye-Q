"""Stage 2 — VLM Inference Worker: wake-on-event visual question answering.

This module runs in a dedicated daemon thread, listening for detection
triggers from the Stage 1 YOLO Watchdog via ZMQ. When triggered, it:
    1. Fetches the high-resolution frame from the ring buffer by timestamp
    2. Crops the image to the detected bounding box (with padding)
    3. Filters out rules whose ROI does not contain the detection
    4. Queries the VLM with each matching rule's plain-English prompt
    5. Parses the response into a binary Yes/No verdict
    6. Saves an evidence JPEG and publishes the result for the control plane

Key design constraints:
    - **Staleness protection**: Triggers older than N seconds are discarded
    - **Timeout enforcement**: No single inference exceeds ``inference_timeout_s``
    - **Memory cleanup**: The backend's ``cleanup()`` runs after every trigger
    - **Non-blocking**: VLM latency (500ms–2s) never blocks the video pipeline
    - **Socket ownership**: The result publisher is created *inside* the worker
      thread, because ZMQ sockets are not safe to share across threads

Results are published on their own endpoint (``:5558`` by default), separate
from the detection endpoint (``:5557``), so Stage 1 and Stage 2 never share a
socket. :class:`~app.services.alert_sink.AlertSink` consumes the result topic.

Example:
    >>> worker = VLMWorker(buffer=buffer, device="cpu")
    >>> worker.add_rule("person", "Is anyone missing a hard hat?", rule_id=1)
    >>> worker.start()
    >>> # ... runs in background, processes triggers from YOLO ...
    >>> worker.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.core.buffer import RollingFrameBuffer
from app.core.ipc import EventPublisher, EventSubscriber
from app.core.roi import RegionOfInterest
from app.pipelines.vlm_backends import MoondreamBackend, VLMBackend
from app.pipelines.vlm_judge import BinaryVerdictParser
from app.schemas.events import BoundingBox, DetectionEvent, VLMQuery, VLMResult

logger = logging.getLogger(__name__)


# Appended to every rule prompt. BinaryVerdictParser is most reliable on its
# first-word strategy, so nudging the model toward a leading "Yes"/"No"
# materially improves verdict accuracy over free-form prose.
VERDICT_SUFFIX = " Answer with only Yes or No."

# Above this many memo entries, expired ones are swept. Keys are per subject
# and track ids never repeat, so the memo grows with footfall rather than with
# the number of rules.
_MEMO_PRUNE_AT = 1024

# Ceiling on how many triggers one drain may carry forward. Not the load
# control — that is max_instances_per_batch, applied after the memo has had its
# say — purely a guard against an unbounded backlog materialising as crops.
_BATCH_HARD_CAP = 64


# ── Frame Cropping ───────────────────────────────────────────────


def crop_bounds(
    frame_shape: tuple[int, ...],
    bbox: BoundingBox,
    padding_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    """Pixel bounds of the padded crop for a detection, as ``(x1, y1, x2, y2)``.

    Split out from :func:`crop_detection` so the focused variant can reuse the
    exact same geometry — the two must never disagree about where the crop
    starts, or the subject would be dimmed instead of highlighted.

    Both degenerate cases fall back to the whole frame: a bbox covering most of
    the frame has no meaningful context left to add, and a crop under 32x32 is
    too small for the VLM to read.

    Args:
        frame_shape: ``frame.shape`` of the source frame.
        bbox: Normalized bounding box coordinates.
        padding_ratio: Extra padding as a fraction of bbox dimensions.

    Returns:
        Pixel bounds, clamped to the frame.
    """
    h, w = frame_shape[:2]

    # If bbox covers >80% of the frame, use the full frame
    if bbox.area > 0.8:
        return 0, 0, w, h

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
        return 0, 0, w, h

    return x1, y1, x2, y2


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
    x1, y1, x2, y2 = crop_bounds(frame.shape, bbox, padding_ratio)
    return frame[y1:y2, x1:x2].copy()


def crop_detection_focused(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.15,
    dim: float = 0.4,
) -> np.ndarray:
    """Crop to the detection, dimming everything outside the subject's own box.

    Padding is what lets the VLM answer questions about relationships — is this
    person bending over, is that pallet blocking the exit — so it cannot simply
    be removed. But in a crowded frame the padding drags in the people standing
    nearby, and "is this person missing a hard hat?" asked over a crop
    containing two people has no single correct answer. The model will pick
    one, and which one is not something the operator controls.

    Dimming keeps the context legible while making the subject unambiguous.

    Args:
        frame: Full-resolution frame (H, W, C), BGR.
        bbox: Normalized box of the subject to highlight.
        padding_ratio: Extra context around the box, as a fraction of its size.
        dim: Brightness multiplier for the context. 1.0 returns a plain crop.

    Returns:
        A new array — the crop, with the subject at full brightness.
    """
    x1, y1, x2, y2 = crop_bounds(frame.shape, bbox, padding_ratio)
    crop = frame[y1:y2, x1:x2]

    if dim >= 1.0:
        return crop.copy()

    h, w = frame.shape[:2]
    crop_h, crop_w = crop.shape[:2]

    # The subject's box in coordinates local to the crop. Clamped because
    # crop_bounds falls back to the whole frame for very large or very small
    # boxes, in which case the subject may sit anywhere inside it.
    bx1 = max(0, min(crop_w, int(bbox.x1 * w) - x1))
    by1 = max(0, min(crop_h, int(bbox.y1 * h) - y1))
    bx2 = max(0, min(crop_w, int(bbox.x2 * w) - x1))
    by2 = max(0, min(crop_h, int(bbox.y2 * h) - y1))

    # One SIMD pass over a sub-megapixel crop — well under a millisecond, and
    # convertScaleAbs already returns a fresh array.
    out = cv2.convertScaleAbs(crop, alpha=dim)
    if bx2 > bx1 and by2 > by1:
        out[by1:by2, bx1:bx2] = crop[by1:by2, bx1:bx2]

    return out


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
    triggers_coalesced: int
    triggers_failed: int
    triggers_timed_out: int
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
        4. Select rules matching the label whose ROI contains the detection
        5. Query the VLM for each rule, under a hard timeout
        6. Parse the verdict, save evidence, publish the result

    Args:
        buffer: Shared-memory ring buffer to read frames from.
        zmq_endpoint: ZMQ endpoint to subscribe to detection events.
        result_endpoint: ZMQ endpoint to publish ``vlm_result`` events on.
                         Must differ from ``zmq_endpoint``.
        backend: VLM backend to use. Defaults to in-process Moondream2.
        device: Inference device for the default backend.
        model_id: Hugging Face model ID for the default backend.
        model_revision: Model revision for the default backend.
        staleness_threshold_s: Discard triggers older than this (seconds).
        crop_padding: Extra padding around bbox as fraction of bbox size.
        inference_timeout_s: Hard cap on a single VLM query.
        snapshot_dir: Directory for evidence JPEGs. None disables snapshots.
        default_prompt: Optional fallback question for detections no rule
            matches. ``None`` (the default) means an unmatched detection is
            simply ignored — the appliance should never spend an inference on
            a question the operator did not ask.
    """

    def __init__(
        self,
        buffer: RollingFrameBuffer,
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        result_endpoint: str = "tcp://127.0.0.1:5558",
        backend: VLMBackend | None = None,
        device: str = "cpu",
        model_id: str = "vikhyatk/moondream2",
        model_revision: str = "2024-08-26",
        staleness_threshold_s: float = 5.0,
        crop_padding: float = 0.15,
        inference_timeout_s: float = 30.0,
        snapshot_dir: str | None = "data/snapshots",
        default_prompt: str | None = None,
    ) -> None:
        self._buffer = buffer
        self._zmq_endpoint = zmq_endpoint
        self._result_endpoint = result_endpoint
        self._staleness_threshold = staleness_threshold_s
        self._crop_padding = crop_padding
        self._inference_timeout_s = inference_timeout_s
        self._snapshot_dir = snapshot_dir
        self._default_prompt = default_prompt

        # VLM backend (loaded lazily inside the worker thread)
        self._vlm: VLMBackend = backend or MoondreamBackend(
            device=device,
            model_id=model_id,
            revision=model_revision,
        )

        # Verdict parser
        self._verdict_parser = BinaryVerdictParser()

        # Prompt rules: label -> list of rule dicts. Synced from SQLite by the
        # control plane whenever a rule is created, updated, or deleted.
        # Guarded by a lock because the API event loop writes while the worker
        # thread reads.
        self._rules: dict[str, list[dict]] = {}
        self._rules_lock = threading.Lock()

        # Single-worker executor used purely to bound inference time. One
        # worker is deliberate: if an inference hangs, the next submit blocks
        # and times out fast rather than fanning out runaway threads.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VLMInfer")

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Statistics
        self._triggers_received = 0
        self._triggers_processed = 0
        self._triggers_stale = 0
        self._triggers_coalesced = 0
        self._triggers_failed = 0
        self._triggers_timed_out = 0
        self._verdicts_positive = 0
        self._verdicts_negative = 0
        self._inference_times: list[float] = []
        self._last_inference_ms: float = 0.0
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── Rule Management ──────────────────────────────────────────

    def add_rule(
        self,
        label: str,
        prompt: str,
        rule_id: int = 0,
        rule_name: str = "",
        roi: RegionOfInterest | None = None,
    ) -> None:
        """Register a prompt rule for a specific detection label.

        When a detection with ``label`` arrives, the VLM is queried with
        ``prompt`` — but only if ``roi`` (when given) contains the detection.

        Per-rule ROI is enforced here rather than in the watchdog because a
        single watchdog cannot hold N distinct polygons; the point-in-polygon
        test is pure geometry and costs nothing at trigger rates.

        Args:
            label: COCO class label to match, or ``"*"`` for any label.
            prompt: Plain-English question for the VLM.
            rule_id: Database ID for the rule.
            rule_name: Human-readable name.
            roi: Optional region of interest for this rule.
        """
        with self._rules_lock:
            self._rules.setdefault(label, []).append(
                {
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "prompt": prompt,
                    "roi": roi,
                }
            )

        logger.info(f"VLM rule added: label='{label}', prompt='{prompt[:60]}'")

    def clear_rules(self) -> None:
        """Remove all registered prompt rules."""
        with self._rules_lock:
            self._rules.clear()
        logger.info("All VLM rules cleared.")

    def _match_rules(self, event: DetectionEvent) -> list[dict]:
        """Select the rules that should run for a detection.

        Label-specific rules take precedence; wildcard (``"*"``) rules apply
        when no label-specific rule exists. Rules whose ROI excludes the
        detection are dropped.

        Returns an empty list when nothing matches, so no VLM call is made.
        A fallback prompt runs only if one was explicitly configured via
        ``default_prompt`` — otherwise a detection nobody wrote a rule for
        costs nothing and produces no alert.
        """
        with self._rules_lock:
            candidates = list(self._rules.get(event.label, []))
            if not candidates:
                candidates = list(self._rules.get("*", []))

        matched = [
            rule
            for rule in candidates
            if rule.get("roi") is None or rule["roi"].contains_bbox_center(event.bbox)
        ]

        if candidates and not matched:
            logger.debug(
                f"All {len(candidates)} rule(s) for '{event.label}' excluded by ROI."
            )
            return []

        if not matched:
            if not self._default_prompt:
                return []
            return [
                {
                    "rule_id": 0,
                    "rule_name": "default",
                    "prompt": self._default_prompt,
                    "roi": None,
                }
            ]

        return matched

    # ── Lifecycle ────────────────────────────────────────────────

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
            self._executor.shutdown(wait=False)
            return

        logger.info("Stopping VLMWorker...")
        self._shutdown_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"VLMWorker did not stop within {timeout}s (may be mid-inference)."
                )

        self._thread = None
        self._executor.shutdown(wait=False)

        logger.info(
            f"VLMWorker stopped. "
            f"Processed {self._triggers_processed} triggers, "
            f"positive={self._verdicts_positive}, "
            f"negative={self._verdicts_negative}, "
            f"stale={self._triggers_stale}, "
            f"coalesced={self._triggers_coalesced}."
        )

    @property
    def stats(self) -> VLMWorkerStats:
        """Current worker statistics snapshot."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        recent = self._inference_times[-50:]
        avg_ms = sum(recent) / len(recent) if recent else 0.0

        return VLMWorkerStats(
            is_running=self.is_running,
            model_loaded=self._vlm.is_loaded,
            device=self._vlm.device,
            triggers_received=self._triggers_received,
            triggers_processed=self._triggers_processed,
            triggers_stale=self._triggers_stale,
            triggers_coalesced=self._triggers_coalesced,
            triggers_failed=self._triggers_failed,
            triggers_timed_out=self._triggers_timed_out,
            verdicts_positive=self._verdicts_positive,
            verdicts_negative=self._verdicts_negative,
            avg_inference_ms=avg_ms,
            last_inference_ms=self._last_inference_ms,
            uptime_seconds=uptime,
        )

    # ── Internal Worker Loop ─────────────────────────────────────

    def _worker_loop(self) -> None:
        """Main worker loop running in a dedicated thread.

        Both ZMQ sockets are created here, in the thread that uses them —
        ZMQ sockets must not be shared across threads.
        """
        try:
            self._vlm.load()
        except Exception as exc:
            logger.error(f"VLMWorker: failed to load model: {exc}")
            return

        subscriber = EventSubscriber(self._zmq_endpoint, topic="detection")
        publisher = EventPublisher(self._result_endpoint)

        try:
            while not self._shutdown_event.is_set():
                for payload in self._collect_batch(subscriber):
                    if self._shutdown_event.is_set():
                        break
                    try:
                        self._process_trigger(payload, publisher)
                    except Exception as exc:
                        self._triggers_failed += 1
                        logger.error(
                            f"VLMWorker: trigger processing failed: {exc}",
                            exc_info=True,
                        )
                    finally:
                        # Always release accelerator memory back to YOLO
                        self._vlm.cleanup()

        finally:
            publisher.close()
            subscriber.close()
            self._vlm.unload()

    def _collect_batch(self, subscriber: EventSubscriber) -> list[dict]:
        """Take one trigger, then coalesce the whole backlog behind it.

        VLM inference (~1–8s) is far slower than Stage 1 produces triggers, so
        a queue builds up. Processing it in arrival order means always working
        on the *oldest* frame, which is exactly the one most likely to have
        aged past ``staleness_threshold`` by the time it is dequeued — the
        pipeline spends its budget on work it then throws away.

        Instead, drain everything pending and keep only the newest trigger per
        label, newest first. Older duplicates of the same label describe a
        scene that a fresher frame already supersedes.

        Returns:
            Triggers to process, newest first. Empty if nothing arrived.
        """
        # 1s timeout so shutdown stays responsive when the scene is quiet.
        first = subscriber.receive(timeout_ms=1000)
        if first is None:
            return []

        pending = [first[1]]
        pending.extend(payload for _topic, payload in subscriber.drain(max_messages=512))
        self._triggers_received += len(pending)

        newest: dict[str, dict] = {}
        for payload in pending:
            label = payload.get("label", "")
            current = newest.get(label)
            if current is None or payload.get("timestamp", 0.0) > current.get(
                "timestamp", 0.0
            ):
                newest[label] = payload

        coalesced = len(pending) - len(newest)
        if coalesced:
            self._triggers_coalesced += coalesced
            logger.debug(
                f"Coalesced {coalesced} superseded trigger(s); "
                f"processing {len(newest)} label(s)."
            )

        return sorted(
            newest.values(), key=lambda p: p.get("timestamp", 0.0), reverse=True
        )

    def _infer(self, image: Image.Image, prompt: str) -> str:
        """Run a single VLM query under a hard timeout.

        Args:
            image: Cropped detection image.
            prompt: Question to ask.

        Returns:
            Raw VLM text response.

        Raises:
            concurrent.futures.TimeoutError: If inference exceeds the timeout.
        """
        future = self._executor.submit(self._vlm.query, image, prompt)
        return future.result(timeout=self._inference_timeout_s)

    def _save_snapshot(self, image_bgr: np.ndarray) -> str | None:
        """Write an evidence JPEG for an alert.

        Args:
            image_bgr: The cropped region the VLM was shown, in BGR order.

        Returns:
            Path to the written file, or None if snapshots are disabled or
            the write failed (evidence is best-effort — never fail a verdict
            because the disk is full).
        """
        if not self._snapshot_dir:
            return None

        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            path = os.path.join(self._snapshot_dir, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return path
        except Exception as exc:
            logger.warning(f"Failed to write alert snapshot: {exc}")
            return None

    def _process_trigger(self, payload: dict, publisher: EventPublisher) -> None:
        """Process a single detection trigger end to end."""
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

        # ── Select rules before doing any expensive work ─────────
        rules = self._match_rules(event)
        if not rules:
            return

        # ── Fetch frame from buffer ──────────────────────────────
        frame_result = self._buffer.read_by_timestamp(event.timestamp, tolerance_ms=500.0)

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

        frame, _actual_ts, _slot_idx = frame_result

        # ── Crop to bounding box ─────────────────────────────────
        cropped = crop_detection(frame, event.bbox, self._crop_padding)
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # One evidence JPEG per trigger, shared by every rule it fires.
        snapshot_path: str | None = None
        snapshot_written = False

        # ── Query VLM for each matching rule ─────────────────────
        for rule in rules:
            prompt = rule["prompt"]
            rule_id = rule["rule_id"]
            rule_name = rule.get("rule_name", "")

            logger.info(f"VLM query: [{event.label}] '{prompt[:80]}' (rule_id={rule_id})")

            start_time = time.monotonic()
            try:
                raw_response = self._infer(pil_image, prompt + VERDICT_SUFFIX)
            except FutureTimeoutError:
                self._triggers_timed_out += 1
                # Ask the backend to abandon generation. Without this the
                # executor's single worker stays pinned on the dead inference
                # and the *next* trigger waits behind it too.
                self._vlm.request_stop()
                logger.error(
                    f"VLM inference exceeded {self._inference_timeout_s}s "
                    f"for rule_id={rule_id}. Aborting and skipping trigger."
                )
                continue
            except Exception as exc:
                self._triggers_failed += 1
                logger.error(f"VLM inference failed: {exc}")
                continue
            inference_ms = (time.monotonic() - start_time) * 1000.0

            self._last_inference_ms = inference_ms
            self._inference_times.append(inference_ms)
            if len(self._inference_times) > 100:
                self._inference_times = self._inference_times[-100:]

            # Parse verdict
            verdict = self._verdict_parser.parse(raw_response)
            self._triggers_processed += 1

            if verdict:
                self._verdicts_positive += 1
            else:
                self._verdicts_negative += 1

            if not snapshot_written:
                snapshot_path = self._save_snapshot(cropped)
                snapshot_written = True

            result = VLMResult(
                query=VLMQuery(
                    detection=event,
                    rule_id=rule_id,
                    rule_name=rule_name,
                    prompt=prompt,
                ),
                verdict=verdict,
                raw_response=raw_response,
                inference_time_ms=inference_ms,
                model_name=self._vlm.model_name,
                snapshot_path=snapshot_path,
            )

            verdict_str = "🔴 YES (VIOLATION)" if verdict else "🟢 NO (clear)"
            logger.info(
                f"VLM result: {verdict_str} | "
                f"inference={inference_ms:.0f}ms | "
                f"response='{raw_response[:100]}'"
            )

            # Hand off to the control plane (persist + broadcast + webhook)
            publisher.publish("vlm_result", result.model_dump())

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"VLMWorker({status}, "
            f"processed={self._triggers_processed}, "
            f"positive={self._verdicts_positive}, "
            f"negative={self._verdicts_negative})"
        )
