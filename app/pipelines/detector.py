"""Stage 1 — YOLO Watchdog: always-on lightweight object detector.

This module runs a YOLOv8n (nano) model in a dedicated daemon thread,
reading the latest frame from the shared-memory ring buffer at up to
30 FPS. When a relevant object is detected inside the configured ROI,
it publishes a ``DetectionEvent`` to the ZMQ PUB socket for the Stage 2
VLM worker to consume.

Key design principles:
    - **Lightweight**: YOLOv8n inference is ~2-4ms on Apple MPS, making
      it feasible to run at full 30 FPS without thermal throttling.
    - **Non-blocking**: Runs in its own thread, never blocks capture or VLM.
    - **Trigger throttling**: Per-label cooldown prevents flooding the VLM
      with redundant detections of the same object.
    - **ROI filtering**: Only detections whose center falls within the
      configured polygon are forwarded.
    - **Display vs. trigger scope**: label/ROI/cooldown gate what wakes the
      VLM, not what the dashboard draws. Every confident detection is recorded
      for the live overlay so a narrow rule never blinds the operator.

Example:
    >>> watchdog = YOLOWatchdog(
    ...     buffer=buffer,
    ...     publisher=publisher,
    ...     device="mps",
    ... )
    >>> watchdog.start()
    >>> # ... runs in background until ...
    >>> watchdog.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.core.buffer import RollingFrameBuffer
from app.core.ipc import EventPublisher
from app.core.roi import RegionOfInterest
from app.pipelines.tracker import GreedyIoUTracker
from app.schemas.events import BoundingBox, DetectionEvent

logger = logging.getLogger(__name__)


# ── Cooldown Gate ────────────────────────────────────────────────


# Above this many live keys the gate prunes. Keys are per *instance* now, so
# the map grows with track churn rather than with the handful of COCO labels
# an operator watches.
_COOLDOWN_PRUNE_AT = 512

# Backstop for the pathological case: a very long cooldown means almost nothing
# ages out, so age-based pruning alone would not bound the map.
_COOLDOWN_HARD_CAP = 4096


class CooldownGate:
    """Per-key cooldown to prevent trigger flooding.

    After a key triggers a detection event, further detections under the same
    key are suppressed for ``cooldown_seconds``. This prevents the VLM worker
    from being overwhelmed by redundant triggers (e.g., a person standing in
    view generates 30 detections per second without cooldown).

    The key is opaque to this class. Stage 1 keys it by *instance*
    (``"person#7"``) when tracking is on, and by bare label (``"person"``)
    when it is off — which is what makes a frame of five workers produce five
    triggers rather than one.

    Args:
        cooldown_seconds: Minimum seconds between triggers for the same key.
    """

    def __init__(self, cooldown_seconds: float = 5.0) -> None:
        self._cooldown = cooldown_seconds
        self._last_trigger: dict[str, float] = {}

    def should_trigger(self, key: str) -> bool:
        """Check if a key is eligible to trigger (cooldown elapsed).

        If eligible, updates the last-trigger timestamp for this key.

        Args:
            key: Cooldown key — a label, or a label/instance pair.

        Returns:
            True if the cooldown period has elapsed since the last trigger.
        """
        now = time.monotonic()
        last = self._last_trigger.get(key, 0.0)

        if now - last >= self._cooldown:
            if len(self._last_trigger) >= _COOLDOWN_PRUNE_AT:
                self._prune(now)
            self._last_trigger[key] = now
            return True
        return False

    def _prune(self, now: float) -> None:
        """Drop keys that can no longer suppress anything.

        An entry older than the cooldown would pass ``should_trigger`` anyway,
        so discarding it is exact rather than lossy — the gate behaves
        identically with or without the prune. Without it, per-instance keying
        would leak one entry per track for the life of the process, which on an
        appliance meant to run for weeks is a real leak.
        """
        self._last_trigger = {
            key: ts for key, ts in self._last_trigger.items() if now - ts < self._cooldown
        }

        if len(self._last_trigger) > _COOLDOWN_HARD_CAP:
            # Nothing aged out (a very long cooldown). Keep the most recent,
            # since those are the only ones still able to suppress.
            newest = sorted(self._last_trigger.items(), key=lambda kv: kv[1], reverse=True)
            self._last_trigger = dict(newest[:_COOLDOWN_HARD_CAP])

    def is_known(self, key: str) -> bool:
        """Whether this key has ever triggered within the current window."""
        return key in self._last_trigger

    @property
    def tracked_keys(self) -> int:
        """Number of keys currently held, for leak monitoring."""
        return len(self._last_trigger)

    def reset(self, key: str | None = None) -> None:
        """Reset cooldown for a specific key or all keys.

        Args:
            key: If provided, reset only this key. If None, reset all.
        """
        if key is None:
            self._last_trigger.clear()
        else:
            self._last_trigger.pop(key, None)

    @property
    def cooldown_seconds(self) -> float:
        """Current cooldown duration in seconds."""
        return self._cooldown

    @cooldown_seconds.setter
    def cooldown_seconds(self, value: float) -> None:
        """Update the cooldown duration."""
        self._cooldown = max(0.0, value)


# ── Watchdog Statistics ──────────────────────────────────────────


@dataclass
class WatchdogStats:
    """Snapshot of YOLO watchdog state for monitoring dashboards."""

    is_running: bool
    device: str
    model_name: str
    frames_processed: int
    detections_total: int
    detections_published: int
    detections_filtered_roi: int
    detections_filtered_confidence: int
    detections_filtered_cooldown: int
    detections_filtered_label: int
    current_fps: float
    uptime_seconds: float
    tracks_active: int = 0
    cooldown_keys: int = 0
    frames_repeated: int = 0


# ── YOLO Watchdog ────────────────────────────────────────────────


class YOLOWatchdog:
    """Stage 1 always-on lightweight object detector.

    Reads frames from the ring buffer, runs YOLOv8n inference, filters
    detections by label/confidence/ROI/cooldown, and publishes qualifying
    triggers to the ZMQ PUB socket.

    Args:
        buffer: Shared-memory ring buffer to read frames from.
        publisher: ZMQ event publisher for detection triggers.
        device: Inference device (``"mps"``, ``"cuda"``, or ``"cpu"``).
        model_name: YOLO model filename (e.g., ``"yolov8n.pt"``).
        confidence_threshold: Minimum confidence to accept a detection.
        imgsz: YOLO input image size (pixels, square).
        target_labels: Set of COCO labels to detect. If None, all labels pass.
        roi: Region of interest polygon. If None, full-frame is used.
        cooldown_seconds: Per-instance trigger cooldown.
        camera_id: Camera identifier attached to published events.
        target_fps: Maximum detector frame rate.
        track_enabled: Assign stable per-instance ids so every subject in a
            frame can trigger, not just the first of its label. Turning this
            off restores the original per-label behaviour exactly.
        track_iou_threshold: Minimum IoU to continue an existing track.
        track_max_age_frames: Frames a track survives unmatched.
    """

    def __init__(
        self,
        buffer: RollingFrameBuffer,
        publisher: EventPublisher,
        device: str = "cpu",
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        imgsz: int = 640,
        target_labels: set[str] | None = None,
        roi: RegionOfInterest | None = None,
        cooldown_seconds: float = 5.0,
        camera_id: str = "default",
        target_fps: float = 30.0,
        track_enabled: bool = True,
        track_iou_threshold: float = 0.2,
        track_max_age_frames: int = 60,
    ) -> None:
        self._buffer = buffer
        self._publisher = publisher
        self._device = device
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._imgsz = imgsz
        self._target_labels = target_labels
        self._roi = roi or RegionOfInterest.full_frame()
        self._cooldown = CooldownGate(cooldown_seconds)
        self._camera_id = camera_id
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        # Instance identity. Without it the cooldown below is keyed by label,
        # so a frame of five workers wakes the VLM about exactly one of them.
        self._tracker = (
            GreedyIoUTracker(
                iou_threshold=track_iou_threshold,
                max_age_frames=track_max_age_frames,
            )
            if track_enabled
            else None
        )

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

        # YOLO model (loaded lazily in the detector thread)
        self._model = None

        # Statistics
        self._frames_processed = 0
        self._detections_total = 0
        self._detections_published = 0
        self._detections_filtered_roi = 0
        self._detections_filtered_confidence = 0
        self._detections_filtered_cooldown = 0
        self._detections_filtered_label = 0
        self._start_time: float = 0.0
        self._fps_counter = _WatchdogFPSCounter(window_size=30)
        self._last_timestamp: float | None = None
        self._frames_repeated = 0

        # Recently seen detections, for the dashboard's live overlay. Holds
        # every detection above the confidence threshold — including labels no
        # rule watches, and detections suppressed by ROI or cooldown — so the
        # live feed shows the whole scene rather than only what triggers.
        self._recent: deque[tuple[DetectionEvent, float, bool]] = deque(maxlen=256)
        self._recent_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the watchdog thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def recent_detections(
        self, max_age_s: float = 1.0
    ) -> list[tuple[DetectionEvent, bool]]:
        """Detections seen within the last ``max_age_s`` seconds.

        Args:
            max_age_s: Maximum age of returned detections.

        Returns:
            List of ``(event, triggered)`` pairs, where ``triggered`` marks
            detections that actually woke the VLM. Untriggered entries include
            out-of-scope labels, out-of-ROI positions, and cooldown holds.
        """
        cutoff = time.monotonic() - max_age_s
        with self._recent_lock:
            return [(evt, fired) for evt, ts, fired in self._recent if ts >= cutoff]

    @property
    def roi(self) -> RegionOfInterest:
        """Current region of interest, for overlay rendering."""
        return self._roi

    def start(self) -> None:
        """Start the YOLO watchdog in a dedicated daemon thread.

        Raises:
            RuntimeError: If the watchdog is already running.
        """
        if self.is_running:
            raise RuntimeError("YOLOWatchdog is already running.")

        self._shutdown_event.clear()
        self._start_time = time.monotonic()

        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="YOLOWatchdog",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            f"YOLOWatchdog started: model={self._model_name}, "
            f"device={self._device}, target_fps={self._target_fps}"
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the watchdog to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        if not self.is_running:
            return

        logger.info("Stopping YOLOWatchdog...")
        self._shutdown_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"YOLOWatchdog did not stop within {timeout}s."
                )

        self._thread = None
        logger.info(
            f"YOLOWatchdog stopped. Processed {self._frames_processed} frames, "
            f"published {self._detections_published} triggers."
        )

    @property
    def stats(self) -> WatchdogStats:
        """Current watchdog statistics snapshot."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        return WatchdogStats(
            is_running=self.is_running,
            device=self._device,
            model_name=self._model_name,
            frames_processed=self._frames_processed,
            detections_total=self._detections_total,
            detections_published=self._detections_published,
            detections_filtered_roi=self._detections_filtered_roi,
            detections_filtered_confidence=self._detections_filtered_confidence,
            detections_filtered_cooldown=self._detections_filtered_cooldown,
            detections_filtered_label=self._detections_filtered_label,
            current_fps=self._fps_counter.fps,
            uptime_seconds=uptime,
            tracks_active=self._tracker.active_count if self._tracker else 0,
            cooldown_keys=self._cooldown.tracked_keys,
            frames_repeated=self._frames_repeated,
        )

    # ── Configuration Mutators ───────────────────────────────────

    def update_roi(self, roi: RegionOfInterest) -> None:
        """Update the region of interest at runtime.

        Thread-safe: the ROI is read atomically in the watchdog loop.
        """
        self._roi = roi
        logger.info(f"YOLOWatchdog ROI updated to: {roi}")

    def update_target_labels(self, labels: set[str] | None) -> None:
        """Update the label whitelist at runtime.

        Args:
            labels: New set of labels to detect, or None for all labels.
        """
        self._target_labels = labels
        logger.info(f"YOLOWatchdog target labels updated to: {labels}")

    def update_cooldown(self, seconds: float) -> None:
        """Update the per-label cooldown duration at runtime."""
        self._cooldown.cooldown_seconds = seconds
        logger.info(f"YOLOWatchdog cooldown updated to: {seconds}s")

    def update_confidence(self, threshold: float) -> None:
        """Update the minimum detection confidence at runtime.

        Args:
            threshold: New confidence threshold, clamped to [0, 1].
        """
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"YOLOWatchdog confidence updated to: {self._confidence_threshold}")

    # ── Internal Watchdog Loop ───────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Main watchdog loop running in a dedicated thread."""
        # Load the YOLO model (this can take a few seconds on first run)
        if not self._load_model():
            logger.error("YOLOWatchdog: failed to load model. Exiting.")
            return

        logger.info(
            f"YOLOWatchdog model loaded: {self._model_name} on {self._device}"
        )

        last_process_time = 0.0

        while not self._shutdown_event.is_set():
            try:
                # Frame-rate control
                now = time.monotonic()
                elapsed = now - last_process_time
                if elapsed < self._frame_interval:
                    # Sleep for the remaining time (interruptible)
                    remaining = self._frame_interval - elapsed
                    self._shutdown_event.wait(timeout=remaining)
                    if self._shutdown_event.is_set():
                        break
                    continue

                # Read the latest frame from the buffer
                try:
                    frame, timestamp, frame_index = self._buffer.read_latest()
                except RuntimeError:
                    # Buffer is empty — wait a bit and retry
                    self._shutdown_event.wait(timeout=0.1)
                    continue

                # Never run inference twice on the same frame. The loop is
                # paced independently of capture, so whenever the camera is
                # slower than target_fps — a 6 FPS clip, an RTSP stream that
                # drops, any CPU-bound board — the buffer keeps returning the
                # frame already processed. Detecting the same objects again
                # costs a full YOLO pass and changes nothing.
                if timestamp == self._last_timestamp:
                    self._frames_repeated += 1
                    self._shutdown_event.wait(timeout=min(0.02, self._frame_interval))
                    continue
                self._last_timestamp = timestamp

                # Run YOLO inference
                self._process_frame(frame, timestamp, frame_index)

                last_process_time = time.monotonic()
                self._frames_processed += 1
                self._fps_counter.tick()

            except Exception as exc:
                logger.error(f"YOLOWatchdog error: {exc}", exc_info=True)
                # Prevent tight error loops
                self._shutdown_event.wait(timeout=0.5)

    def _load_model(self) -> bool:
        """Load the YOLO model. Returns True on success.

        Attempts to use the configured device (MPS/CUDA), falling back
        to CPU if the preferred device fails.
        """
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)

            # Warm up the model with a dummy inference to initialize device
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            try:
                self._model.predict(
                    dummy,
                    device=self._device,
                    imgsz=self._imgsz,
                    verbose=False,
                )
                logger.info(
                    f"YOLO warm-up successful on device '{self._device}'."
                )
            except Exception as device_exc:
                logger.warning(
                    f"YOLO failed on device '{self._device}': {device_exc}. "
                    f"Falling back to CPU."
                )
                self._device = "cpu"
                self._model.predict(
                    dummy,
                    device="cpu",
                    imgsz=self._imgsz,
                    verbose=False,
                )
                logger.info("YOLO warm-up successful on CPU (fallback).")

            return True

        except ImportError:
            logger.error(
                "ultralytics package not installed. "
                "Run: uv add ultralytics"
            )
            return False
        except Exception as exc:
            logger.error(f"Failed to load YOLO model '{self._model_name}': {exc}")
            return False

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> None:
        """Run YOLO inference on a single frame and publish qualifying detections.

        Two passes, because tracking is a whole-frame operation:

            1. Decode every box and apply the confidence filter, collecting the
               survivors.
            2. Assign track ids to that whole set at once.
            3. Per detection, apply label / ROI / cooldown gating and publish.

        The pass split is what makes multi-instance alerting work. The cooldown
        is keyed by instance, so five workers in one frame produce five
        triggers — but only if identity is resolved before the gate, and
        identity can only be resolved by looking at the frame's boxes together.
        """
        if self._model is None:
            return

        # Run inference
        results = self._model.predict(
            frame,
            device=self._device,
            imgsz=self._imgsz,
            conf=self._confidence_threshold,
            stream=True,
            verbose=False,
        )

        frame_h, frame_w = frame.shape[:2]

        # ── Pass 1: decode and filter by confidence ──────────────────
        candidates: list[tuple[str, BoundingBox, float]] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # One device-to-host transfer for the whole frame, rather than
            # three per box. Every `.item()` and `.tolist()` is a
            # synchronization point on an accelerator, so the per-box form
            # measured 5.4 ms for eleven boxes on MPS against 0.15 ms here —
            # a third of Stage 1's entire per-frame budget spent fetching
            # sixty-six numbers. Columns are [x1, y1, x2, y2, conf, cls];
            # read the last two from the right, since Ultralytics inserts a
            # track id before them when its own tracker is driving.
            detections = result.boxes.data.cpu().numpy()
            self._detections_total += len(detections)

            # Normalize and clamp every box at once.
            scale = np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float64)
            xyxy_norm = np.clip(detections[:, :4] / scale, 0.0, 1.0)

            for box_norm, confidence, cls_id in zip(
                xyxy_norm, detections[:, -2], detections[:, -1]
            ):
                confidence = float(confidence)

                # ── Confidence threshold ─────────────────────────
                # (Already filtered by YOLO `conf` param, but double-check
                #  for safety since threshold may be updated at runtime)
                if confidence < self._confidence_threshold:
                    self._detections_filtered_confidence += 1
                    continue

                cls_id = int(cls_id)
                label = result.names.get(cls_id, f"class_{cls_id}")

                bbox = BoundingBox(
                    x1=float(box_norm[0]),
                    y1=float(box_norm[1]),
                    x2=float(box_norm[2]),
                    y2=float(box_norm[3]),
                )

                candidates.append((label, bbox, confidence))

        if not candidates:
            # Still age the tracker, or a subject that walks out of frame would
            # keep its id forever and be treated as already-judged on return.
            if self._tracker is not None:
                self._tracker.update([])
            return

        # ── Pass 2: resolve identity for the whole frame at once ─────
        if self._tracker is not None:
            track_ids = self._tracker.update([(lbl, bb) for lbl, bb, _ in candidates])
        else:
            track_ids = [-1] * len(candidates)

        # ── Pass 3: gate and publish ─────────────────────────────────
        for (label, bbox, confidence), track_id in zip(candidates, track_ids):
            event = DetectionEvent(
                timestamp=timestamp,
                camera_id=self._camera_id,
                label=label,
                confidence=confidence,
                bbox=bbox,
                frame_index=frame_index,
                track_id=track_id,
            )

            # ── Trigger gating ───────────────────────────────
            # Scope (label whitelist), ROI, and cooldown decide whether to
            # WAKE THE VLM. They deliberately do not decide what the live
            # feed shows: a rule watching "person" must not blind the
            # operator to the forklift beside them. YOLO scores every class
            # regardless, so displaying them all is free.
            #
            # The cooldown key carries the instance, so one worker's cooldown
            # cannot suppress the worker standing next to them. With tracking
            # off the key degrades to the bare label, which is exactly the
            # original behaviour.
            key = f"{label}#{track_id}" if track_id >= 0 else label

            fired = False
            if self._target_labels is not None and label not in self._target_labels:
                self._detections_filtered_label += 1
            elif not self._roi.contains_bbox_center(bbox):
                self._detections_filtered_roi += 1
            elif not self._cooldown.should_trigger(key):
                self._detections_filtered_cooldown += 1
            else:
                fired = True

            # Record every confident detection for the live overlay —
            # in-scope or not, triggering or not.
            with self._recent_lock:
                self._recent.append((event, time.monotonic(), fired))

            if not fired:
                continue

            # ── Publish detection event ──────────────────────
            success = self._publisher.publish(
                "detection",
                event.model_dump(),
            )

            if success:
                self._detections_published += 1
                logger.info(
                    f"Detection published: {key} ({confidence:.2f}) "
                    f"at [{bbox.x1:.2f}, {bbox.y1:.2f}, {bbox.x2:.2f}, {bbox.y2:.2f}]"
                )
            else:
                logger.warning(
                    f"Failed to publish detection for {key} (ZMQ send dropped)."
                )

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"YOLOWatchdog(model={self._model_name}, device={self._device}, "
            f"{status}, processed={self._frames_processed}, "
            f"published={self._detections_published})"
        )


class _WatchdogFPSCounter:
    """Sliding-window FPS calculator for the watchdog."""

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record a frame processing timestamp."""
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]

    @property
    def fps(self) -> float:
        """Current frames per second over the sliding window."""
        if len(self._timestamps) < 2:
            return 0.0
        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / duration
