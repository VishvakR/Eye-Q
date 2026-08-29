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
import math
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.core.buffer import RollingFrameBuffer
from app.core.ipc import EventPublisher, EventSubscriber
from app.core.roi import RegionOfInterest
from app.pipelines.vlm_backends import (
    BinaryProbe,
    BinaryProbeBackend,
    Moondream4BitBackend,
    ProbeUnsupportedError,
    VLMBackend,
    VLMUnavailableError,
)
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

# Floor on either side of a crop handed to the VLM. Below roughly this the
# model has nothing to read; the old behaviour of substituting the whole frame
# made such a detection the most expensive query in the system instead.
_MIN_CROP_PX = 64

# Fill for the bars added by :func:`letterbox_square`. Mid-grey, the same value
# YOLO's own letterbox uses: black bars read as unlit scene and the model will
# describe them, while neutral grey reads as nothing at all.
_LETTERBOX_GREY = 114


def _percentile(samples, pct: float) -> float:
    """Nearest-rank percentile of ``samples``, or 0.0 when empty.

    Hand-rolled rather than pulled from numpy: the sample window is 100 floats,
    so sorting it costs nothing, and this keeps the stats path free of any
    dependency that a stripped-down edge install might not carry.
    """
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(0, min(len(ordered) - 1, rank - 1))]


# ── Frame Cropping ───────────────────────────────────────────────


def _widen(lo: int, hi: int, minimum: int, limit: int) -> tuple[int, int]:
    """Grow a span to ``minimum`` around its own centre, staying inside ``limit``.

    Used by :func:`crop_bounds` so a detection too small to crop is enlarged
    where it actually is, rather than replaced by the whole frame.

    Args:
        lo: Low edge of the span, in pixels.
        hi: High edge of the span, in pixels.
        minimum: Smallest acceptable span.
        limit: Frame extent along this axis.

    Returns:
        The widened ``(lo, hi)``, clamped to ``[0, limit]``. When the frame
        itself is narrower than ``minimum`` the whole extent is returned.
    """
    if hi - lo >= minimum:
        return lo, hi
    if limit <= minimum:
        return 0, limit

    centre = (lo + hi) // 2
    lo = centre - minimum // 2
    hi = lo + minimum

    if lo < 0:
        return 0, minimum
    if hi > limit:
        return limit - minimum, limit
    return lo, hi


def prepare_vlm_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    *,
    crop_padding: float = 0.20,
    focus_dim: float = 1.0,
    square_edge: int = 378,
    max_crop_edge: int = 378,
) -> np.ndarray:
    """Turn a frame and a box into the exact array Stage 2 is shown.

    The whole crop sequence in one place, because it had drifted. Stage 2's
    inputs were produced here with one set of arguments and by
    ``bench/make_fixtures.py`` with another — that harness called
    :func:`crop_detection_focused` bare, taking its *signature* default of
    ``dim=0.4``, and never letterboxed at all. Production dims at
    ``vlm_focus_dim`` (1.0, i.e. off) and always letterboxes. So every verdict
    number ever measured on those fixtures described a crop the appliance does
    not generate.

    That is not a small discrepancy. Dimming changes what the model can see
    outside the subject's own box, and letterboxing changes the aspect ratio
    and the image-token count. An accuracy measurement taken on the wrong
    inputs is worse than none, because it looks like evidence.

    Callers pass their own settings; the defaults here mirror the shipped ones
    so a caller that wants "whatever production does" can simply omit them.

    Args:
        frame: Full BGR frame.
        bbox: Normalised subject box.
        crop_padding: Context added around the box, as a fraction of its size.
        focus_dim: Brightness multiplier outside the subject's own box. 1.0
            disables dimming.
        square_edge: Letterbox canvas edge. 0 falls back to ``max_crop_edge``.
        max_crop_edge: Longest-side cap, used **only** when ``square_edge`` is
            0. At the shipped default it is unreachable, so a deployment
            tuning it must set ``EYEQ_VLM_SQUARE_EDGE=0`` as well or the
            change will silently do nothing.

    Returns:
        The crop as it will be handed to the backend, in BGR.
    """
    focused = crop_detection_focused(frame, bbox, crop_padding, focus_dim)
    # Letterboxing subsumes the max-edge cap: it already scales the long side
    # to a fixed value, so nothing can reach the tiling cliff
    # downscale_for_vlm exists to dodge. The old path stays for square_edge=0,
    # which is how a deployment opts out.
    if square_edge > 0:
        return letterbox_square(focused, square_edge)
    return downscale_for_vlm(focused, max_crop_edge)


def subject_pixel_size(
    frame_shape: tuple[int, ...], bbox: BoundingBox
) -> tuple[int, int]:
    """The subject's own box in pixels, before any padding.

    Separate from :func:`crop_bounds` because the question it answers is not
    "where do I cut" but "is there anything here worth asking about" — and the
    padded crop is the wrong thing to measure for that.

    A thin adapter over :meth:`BoundingBox.pixel_size`, which Stage 1's own size
    floor also calls. The two floors must agree exactly about the same box or
    Stage 2's becomes unreachable in a way no counter would show, so the
    arithmetic lives in one place and both stages borrow it.
    """
    h, w = frame_shape[:2]
    return bbox.pixel_size(w, h)


def crop_bounds(
    frame_shape: tuple[int, ...],
    bbox: BoundingBox,
    padding_ratio: float = 0.20,
) -> tuple[int, int, int, int]:
    """Pixel bounds of the padded crop for a detection, as ``(x1, y1, x2, y2)``.

    Split out from :func:`crop_detection` so the focused variant can reuse the
    exact same geometry — the two must never disagree about where the crop
    starts, or the subject would be dimmed instead of highlighted.

    Both degenerate cases used to fall back to the entire frame, and both were
    the wrong way round once Stage 2's cost was measured. The prompt carries
    729 image tokens against a single-token answer, so a whole-frame crop is
    the *most* expensive inference the appliance can run:

        - A subject filling most of the frame produced a padded box that was
          already almost the whole frame, so returning the frame gained nothing
          and cost :func:`crop_detection_focused` its geometry — the subject
          stopped being dimmed apart from its surroundings exactly when the
          scene was busiest.
        - A box too small to crop became the frame, which spent the largest
          inference in the system on the subject least able to support an
          answer.

    Now the padded box is always honoured and simply grown to
    ``_MIN_CROP_PX`` around its own centre when it is undersized.
    :func:`downscale_for_vlm` caps the other end.

    Args:
        frame_shape: ``frame.shape`` of the source frame.
        bbox: Normalized bounding box coordinates.
        padding_ratio: Extra padding as a fraction of bbox dimensions.

    Returns:
        Pixel bounds, clamped to the frame.
    """
    h, w = frame_shape[:2]

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

    # Enlarge an undersized crop in place instead of discarding its geometry.
    x1, x2 = _widen(x1, x2, _MIN_CROP_PX, w)
    y1, y2 = _widen(y1, y2, _MIN_CROP_PX, h)

    return x1, y1, x2, y2


def crop_detection(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.20,
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
                       0.20 = 20% padding on each side.

    Returns:
        Cropped region as a new numpy array.
    """
    x1, y1, x2, y2 = crop_bounds(frame.shape, bbox, padding_ratio)
    return frame[y1:y2, x1:x2].copy()


def crop_detection_focused(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.20,
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
    # crop_bounds widens an undersized box around its centre, after which the
    # subject may sit anywhere inside the returned bounds.
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


def downscale_for_vlm(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Shrink a crop so its longest side fits ``max_edge``, or pass it through.

    This exists because of a tiling cliff inside Moondream's preprocessor
    rather than any generic "smaller is faster" intuition. The cliff is real,
    but *which* cliff depends on which model is loaded, and this docstring
    used to describe the wrong one.

    The **2024** model (``vikhyatk/moondream2``) branched on
    ``if max_dim < 512``: below that it resized to 378x378 and ran the vision
    tower once, at or above it resized to 756 px and pushed the image plus
    every tile through — 3 to 5 passes. Measured on an M2 against *that*
    model: 1.42 s at 360 px versus 3.16 s at 792 px for the same one-word
    answer. That is where the old default of 511 (= 512 - 1) came from.

    The **shipped 2025 4-bit** model tiles by ``select_tiling()`` in
    ``image_crops.py``, which returns a single tile only when
    ``min(height, width) <= 378`` — a test on the *short* side, not the long
    one, derived from ``crop_window_size = (378//14 - 2*4) * 14 = 266`` plus
    ``total_margin_pixels = 112``. Under that rule 511 is single-tile only for
    very elongated crops: 511x383 tiles (2,4) into 9 encoder passes where 378
    would take 2. Hence the default moved to 378, which is single-tile at
    every aspect ratio.

    Crops normally land well under the threshold, so this usually does nothing.
    Superseded by :func:`letterbox_square` when ``square_edge`` is set, which
    fixes the long side outright and so cannot reach the cliff at all; this
    remains the path for ``square_edge=0``.

    Args:
        image: The crop, in BGR.
        max_edge: Longest permitted side in pixels. 0 or less disables.

    Returns:
        The original array when it already fits, otherwise a new smaller one.
    """
    if max_edge <= 0:
        return image

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image

    scale = max_edge / longest
    return cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        # INTER_AREA is the correct filter for shrinking — it averages the
        # source pixels rather than sampling them, so thin features like a vest
        # strap survive the reduction instead of aliasing away.
        interpolation=cv2.INTER_AREA,
    )


def letterbox_square(
    image: np.ndarray, edge: int, pad_value: int = _LETTERBOX_GREY
) -> np.ndarray:
    """Fit a crop into a square canvas without changing its proportions.

    Every subject this appliance asks about is a standing person, so the crops
    are tall and narrow — the demo footage produces roughly 1:3. A VLM that
    resizes its input to a square receives that person stretched to three times
    their width, which is not a shape anything in its training data had.
    Scaling the long side to ``edge`` and padding the short one leaves the
    person the shape they actually are.

    The stretch is not hypothetical. ``overlap_crop_image`` resizes the global
    view with an unconditional ``pil_img.resize((378, 378), LANCZOS)`` that
    does not preserve aspect ratio, so without this a 1:3 person really does
    reach the encoder three times too wide.

    That is the whole justification, and it is worth being precise about what
    is *not*. An earlier version of this note claimed the fixed size also held
    the image-token count constant and so stabilised prefill. It does not: the
    count is always 729 whatever the input, because ``vision_projection``
    adaptive-pools to 27x27 and hard-codes ``.view(729, ...)``. Tiling scales
    the vision encoder's batch dimension, never the token count — and per
    EVALUATION-4bit.md the encoder is 88.8% of a query against prefill's
    11.1%, so the batch is what matters anyway.

    Padding with mid-grey rather than black is deliberate — black bars read as
    part of the scene, as unlit floor or a dark doorway, and invite the model
    to describe them. Neutral grey reads as absence.

    Args:
        image: The crop, in BGR.
        edge: Side of the output square in pixels. Must be positive; callers
            treat 0 as "disabled" and skip this entirely.
        pad_value: Fill for the bars.

    Returns:
        A new ``(edge, edge, 3)`` array. Never the input, and never a view of
        it — the source frame is reused by other subjects in the same batch.
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    scale = edge / longest
    new_w = max(1, min(edge, round(w * scale)))
    new_h = max(1, min(edge, round(h * scale)))

    # INTER_AREA averages when shrinking, which is the common case here;
    # INTER_LINEAR is the better choice on the rare upscale of a distant
    # subject, where averaging has nothing to average.
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.full((edge, edge, image.shape[2]), pad_value, dtype=image.dtype)
    top = (edge - new_h) // 2
    left = (edge - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


# ── Worker Statistics ────────────────────────────────────────────


@dataclass
class _Subject:
    """One cropped subject, plus the evidence JPEG shared by all its rules.

    Exists because batching breaks the assumption that a subject's questions
    are answered together: grouping by prompt can put two rules about the same
    person in different dispatches, and the snapshot must still be written
    once. Holding that state on the subject rather than in a loop variable
    makes it independent of how the questions are scheduled.
    """

    event: DetectionEvent
    cropped: np.ndarray
    image: Image.Image
    snapshot_path: str | None = None
    snapshot_written: bool = False


@dataclass(frozen=True)
class _Answer:
    """One decided verdict, however it was obtained.

    The two paths that produce a verdict — probing the model's probabilities
    and generating text to parse — converge here, so everything downstream
    records a verdict identically and cannot tell which one ran. Without this
    the recording tail would need to know about both.

    Attributes:
        verdict: The decision.
        raw: Text to publish. The model's own words on the generate path, and
            the winning word on the probe path, so consumers of
            ``VLMResult.raw_response`` keep working either way.
        confidence: Yes-probability, when a probe produced it. ``None`` on the
            parsed path — the parser cannot know how strongly a word was
            meant, and inventing a number there would be worse than admitting
            it is unknown.
    """

    verdict: bool
    raw: str
    confidence: float | None = None


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
    # Units differ, deliberately: instances_judged counts *subjects* while
    # instances_memoised counts *rules*. Two enabled rules make one judged
    # subject two forward passes, so the two are not directly comparable.
    instances_judged: int
    instances_memoised: int
    instances_shed: int
    instances_too_small: int
    # Batches abandoned because the endpoint was down. This is now the only
    # place an outage is counted — the failure counters below deliberately
    # ignore VLMUnavailableError, so a dead endpoint reads as one problem
    # rather than as many.
    batches_dropped_unavailable: int
    backend_unavailable: bool
    avg_inference_ms: float
    last_inference_ms: float
    # Capture-to-verdict, which is what an operator actually waits through.
    # Reported as percentiles because the mean hides exactly the case that
    # matters: most subjects answered promptly while a few wait out a batch.
    avg_queued_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    uptime_seconds: float
    # Reuse recovered by position after the id changed. Defaulted so the
    # dataclass keeps its positional contract for existing callers.
    instances_memoised_spatial: int = 0
    # Why the memo missed. new_track dominating means identity is churning and
    # the TTL is irrelevant however high it is set — that was the measured
    # state of this appliance before the spatial fallback existed.
    memo_miss_new_track: int = 0
    memo_miss_ttl: int = 0
    memo_miss_moved: int = 0
    # triggers_failed split by cause. It was incremented from ten sites, so a
    # wrapped ring buffer, a malformed payload and a crashed model were one
    # number; it now stays at zero and these carry the meaning.
    triggers_malformed: int = 0
    triggers_frame_lost: int = 0
    triggers_backend_error: int = 0
    triggers_internal_error: int = 0


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
        backend: VLM backend to use. Defaults to the in-process 4-bit
            Moondream (``2025-04-14``).
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
        focus_dim: Brightness applied outside the subject's box in its crop.
        verdict_ttl_s: Seconds a per-instance verdict is reused before that
            subject is judged again. The load control; see the memo below.
        rejudge_iou: Re-judge early below this IoU against the judged box.
        min_rejudge_s: Floor between re-judgements of one subject.
        crossid_reuse_s: Window for reusing a verdict across a *different*
            track id, by position. 0 disables the spatial fallback.
        crossid_iou: Minimum overlap for that reuse. Deliberately far above
            ``track_iou_threshold`` — see :meth:`_spatial_verdict`.
        max_instances_per_batch: Cap on subjects inferred per drained batch.
        max_crop_edge: Longest side a crop may have when handed to the VLM.
        min_subject_px: Skip a subject whose own box is smaller than this on
            its longer side — too little pixel evidence to answer about.
        batch_size: Crops answered in one VLM forward pass. 1 dispatches
            singly, which is the default and the measured-safe choice.
            See :func:`downscale_for_vlm` for why 512 is a cliff and not a
            preference.
        square_edge: Side of the square canvas each crop is letterboxed into,
            via :func:`letterbox_square`. 0 disables it and falls back to
            ``max_crop_edge`` downscaling.
        probe: ``"auto"``, ``"on"`` or ``"off"``. Whether to read verdicts
            from the model's token probabilities instead of generating an
            answer and parsing it. Falls back permanently on any backend that
            cannot support it.
        probe_threshold: Yes-probability at or above which a probed verdict
            counts as a violation.
    """

    def __init__(
        self,
        buffer: RollingFrameBuffer,
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        result_endpoint: str = "tcp://127.0.0.1:5558",
        backend: VLMBackend | None = None,
        device: str = "cpu",
        model_id: str = "moondream/moondream-2b-2025-04-14-4bit",
        model_revision: str = "a89c59223ef8b5bb7826780728eeec172727ca84",
        staleness_threshold_s: float = 5.0,
        crop_padding: float = 0.20,
        inference_timeout_s: float = 30.0,
        snapshot_dir: str | None = "data/snapshots",
        default_prompt: str | None = None,
        focus_dim: float = 1.0,
        verdict_ttl_s: float = 300.0,
        rejudge_iou: float = 0.5,
        min_rejudge_s: float = 20.0,
        crossid_reuse_s: float = 45.0,
        crossid_iou: float = 0.7,
        max_instances_per_batch: int = 8,
        max_crop_edge: int = 378,
        min_subject_px: int = 96,
        batch_size: int = 1,
        square_edge: int = 378,
        probe: str = "auto",
        probe_threshold: float = 0.5,
    ) -> None:
        self._buffer = buffer
        self._zmq_endpoint = zmq_endpoint
        self._result_endpoint = result_endpoint
        self._staleness_threshold = staleness_threshold_s
        self._crop_padding = crop_padding
        self._inference_timeout_s = inference_timeout_s
        self._snapshot_dir = snapshot_dir
        self._default_prompt = default_prompt
        self._focus_dim = focus_dim
        self._verdict_ttl_s = verdict_ttl_s
        self._rejudge_iou = rejudge_iou
        self._min_rejudge_s = min_rejudge_s
        self._crossid_reuse_s = crossid_reuse_s
        self._crossid_iou = crossid_iou
        self._max_instances_per_batch = max_instances_per_batch
        self._max_crop_edge = max_crop_edge
        self._min_subject_px = min_subject_px
        self._batch_size = max(1, batch_size)
        self._square_edge = square_edge
        self._probe_mode = probe
        self._probe_threshold = probe_threshold

        # Tri-state: None means "not tried yet", False means a backend told us
        # it cannot probe and we must stop asking. Latching matters — without
        # it, a server without logprobs would pay for a doomed probe request
        # before every single generate, doubling its request count forever.
        self._probe_supported: bool | None = None

        # Verdict memo:
        #   (track_id, rule_id) -> (verdict, judged_at, judged_bbox, label)
        #
        # This is the appliance's load control. Judging every tracked subject on
        # every cooldown would multiply inference demand by the size of the
        # crowd; reusing a verdict until the subject moves or the TTL expires
        # spends the same budget covering everyone instead of re-asking about
        # one person. Guarded by a lock because the batch loop reads and writes
        # it while rules are synced from the API thread.
        #
        # The label is carried in the entry rather than the key because the
        # spatial fallback below matches on position, and position alone would
        # happily hand a forklift's verdict to the person driving it.
        self._memo: dict[tuple[int, int], tuple[bool, float, BoundingBox, str]] = {}
        self._judged_tracks: set[int] = set()
        self._memo_lock = threading.Lock()

        # VLM backend (loaded lazily inside the worker thread)
        self._vlm: VLMBackend = backend or Moondream4BitBackend(
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
        # Retired in favour of the four below. Kept at zero and still
        # reported so an existing dashboard does not break; it was ten
        # unrelated causes wearing one name, which made a wrapped ring buffer
        # indistinguishable from a crashed model.
        self._triggers_failed = 0
        # Stage 1 published something that will not parse. Schema drift or a
        # foreign publisher — nothing to do with the VLM.
        self._triggers_malformed = 0
        # The source frame was gone before Stage 2 reached it. A CAPACITY
        # signal, not an error: raise buffer_capacity or cut VLM latency.
        self._triggers_frame_lost = 0
        # The model call raised, or returned a different number of answers
        # than it was asked for. Check device, dtype and memory.
        self._triggers_backend_error = 0
        # A bug in Eye-Q. Both sites log exc_info=True. The only two of the
        # original ten that should ever wake a developer.
        self._triggers_internal_error = 0
        self._triggers_timed_out = 0
        self._verdicts_positive = 0
        self._verdicts_negative = 0
        # How many verdicts came from a probe rather than a parsed sentence.
        # Compared against triggers_processed it says whether the fast path is
        # actually in force, which a config file alone cannot tell you.
        self._verdicts_probed = 0
        self._instances_judged = 0
        self._instances_memoised = 0
        # Reuse that only happened because of the spatial fallback. If this
        # dominates _instances_memoised, identity is churning badly and the
        # exact memo is carrying almost none of the load.
        self._instances_memoised_spatial = 0
        # Why the memo missed. Before these existed the three branches were
        # indistinguishable, and diagnosing a saturated VLM meant patching the
        # code by hand: new_track dominating means the TTL is irrelevant no
        # matter how high it is set.
        self._memo_miss_new_track = 0
        self._memo_miss_ttl = 0
        self._memo_miss_moved = 0
        self._instances_shed = 0
        self._instances_too_small = 0
        # How many batches were abandoned because the backend's endpoint was
        # down. Distinct from triggers_failed, which counts individual queries
        # that failed on their own merits.
        self._batches_dropped_unavailable = 0
        # Latched while the endpoint is unreachable, so a sustained outage
        # logs once instead of once per subject per collection cycle.
        self._backend_unavailable = False
        self._inference_times: deque[float] = deque(maxlen=100)
        self._queued_times: deque[float] = deque(maxlen=100)
        self._latencies: deque[float] = deque(maxlen=100)
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

    def _rules_to_judge(self, event: DetectionEvent) -> list[dict]:
        """Rules for this detection that are not already answered.

        Wraps :meth:`_match_rules` rather than replacing it, so rule matching
        stays a pure function of the rule table while the memo — which is
        stateful and time-dependent — is layered on top and testable alone.

        Two tiers, tried in order.

        **Exact**, on ``(track_id, rule_id)``. A rule is skipped when this exact
        subject was judged for it recently enough that the answer is still
        expected to hold — the verdict is younger than the TTL, and the subject
        has not moved far from where it was judged unless it moved within the
        re-judge floor, which stops a walking subject from demanding a fresh
        inference every stride.

        **Spatial**, on position, when the exact lookup misses because the id is
        new. This exists because the exact tier alone was measurably not
        working: over a 300 s replay of the bundled clip the tracker minted 80
        ids for about six real people, and 122 of 124 memo misses were a new id
        rather than an expired verdict — zero were TTL expiries. Every gate in
        the pipeline is keyed on ``track_id``, so an id change resets the Stage
        1 cooldown and this memo in the same instant, and inference demand ran
        roughly ten times what stable identity would need. See
        :meth:`_spatial_verdict` for the safety argument.

        An untracked detection (``track_id < 0``) is nobody in particular and
        is never memoised — treating it as somebody would blind the appliance
        to everyone after the first person it ever saw.
        """
        matched = self._match_rules(event)
        if not matched or event.track_id < 0 or self._verdict_ttl_s <= 0:
            return matched

        now = time.monotonic()
        unjudged = []

        with self._memo_lock:
            for rule in matched:
                rule_id = rule["rule_id"]
                entry = self._memo.get((event.track_id, rule_id))

                if entry is None:
                    # The dominant miss. Before paying for an inference, ask
                    # whether this box is somewhere we already have an answer.
                    if self._spatial_verdict(event, rule_id, now):
                        continue
                    self._memo_miss_new_track += 1
                    unjudged.append(rule)
                    continue

                _verdict, judged_at, judged_bbox, _label = entry
                age = now - judged_at

                if age >= self._verdict_ttl_s:
                    self._memo_miss_ttl += 1
                    unjudged.append(rule)
                    continue

                moved = event.bbox.iou(judged_bbox) < self._rejudge_iou
                if moved and age >= self._min_rejudge_s:
                    self._memo_miss_moved += 1
                    unjudged.append(rule)
                    continue

                self._instances_memoised += 1

        return unjudged

    def _spatial_verdict(
        self, event: DetectionEvent, rule_id: int, now: float
    ) -> bool:
        """Reuse a recent verdict from a *different* id at the same position.

        Caller holds ``_memo_lock``.

        The exact memo is keyed on identity, and identity is the thing that
        breaks: a subject re-detected after an occlusion, a greedy id swap when
        two people cross, or a seek in the source video all mint a fresh id for
        a person the appliance already has an answer about. This recovers that
        answer from geometry instead.

        The safety argument, because this is the one place the appliance
        trades correctness for load. Reusing a verdict across identities can
        hand person B the answer earned by person A, and a wrongly reused "No"
        is a missed alert — the bad direction. Three things bound it:

            - ``vlm_crossid_iou`` (0.7) is far stricter than the 0.2 the
              tracker itself uses to assert two boxes are the same object, so
              this never claims an identity the tracker would have rejected;
            - ``vlm_crossid_reuse_s`` (45 s) is a separate, much shorter window
              than the TTL, so cross-identity reuse can be shortened or turned
              off without touching same-subject reuse;
            - hits are counted separately, so an operator can see how much of
              their shedding rests on this.

        Returns:
            True when a verdict was reused. The match is also written through
            under the new key, so the next frame takes the O(1) exact path
            rather than scanning again.
        """
        if self._crossid_reuse_s <= 0:
            return False

        best_iou = self._crossid_iou
        best: tuple[bool, float, BoundingBox, str] | None = None

        for (_tid, entry_rule_id), entry in self._memo.items():
            if entry_rule_id != rule_id:
                continue
            verdict, judged_at, judged_bbox, label = entry
            # Cheap rejections first, so IoU is only computed on candidates
            # that could actually match.
            if label != event.label:
                continue
            if now - judged_at >= self._crossid_reuse_s:
                continue
            iou = event.bbox.iou(judged_bbox)
            if iou >= best_iou:
                best_iou = iou
                best = entry

        if best is None:
            return False

        verdict, judged_at, judged_bbox, label = best
        self._memo[(event.track_id, rule_id)] = (verdict, judged_at, event.bbox, label)
        self._judged_tracks.add(event.track_id)
        self._instances_memoised_spatial += 1
        logger.debug(
            f"Reusing verdict for [{event.label}#{event.track_id}] rule {rule_id} "
            f"from a prior subject at IoU {best_iou:.2f}, "
            f"{now - judged_at:.1f}s old."
        )
        return True

    def _remember_verdict(
        self, event: DetectionEvent, rule_id: int, verdict: bool
    ) -> None:
        """Record a verdict so this subject is not re-asked immediately."""
        if event.track_id < 0 or self._verdict_ttl_s <= 0:
            return

        with self._memo_lock:
            self._memo[(event.track_id, rule_id)] = (
                verdict,
                time.monotonic(),
                event.bbox,
                event.label,
            )
            self._judged_tracks.add(event.track_id)

            if len(self._memo) > _MEMO_PRUNE_AT:
                self._prune_memo()

    def _prune_memo(self) -> None:
        """Drop expired memo entries. Caller holds the lock.

        Entries past the TTL would be re-judged anyway, so dropping them is
        exact. Track ids never repeat, so without this the memo would grow for
        the life of the process.
        """
        now = time.monotonic()
        self._memo = {
            key: entry
            for key, entry in self._memo.items()
            if now - entry[1] < self._verdict_ttl_s
        }
        self._judged_tracks = {track_id for track_id, _rule_id in self._memo}

    def _too_small(self, frame_shape: tuple[int, ...], event: DetectionEvent) -> bool:
        """Whether a subject carries too few pixels to be worth a query.

        Measured on the subject's own box rather than the padded crop: padding
        adds context, not evidence, and a distant figure surrounded by more
        floor is no more answerable for it.

        The test is on the *longer* side deliberately. A person is tall and
        narrow, so requiring both sides to clear the floor would drop subjects
        that are perfectly legible.

        Takes the frame it will actually be cropped from rather than the
        buffer's declared shape, so the decision and the crop can never be
        measuring different pixels.
        """
        if self._min_subject_px <= 0:
            return False

        box_w, box_h = subject_pixel_size(frame_shape, event.bbox)
        return max(box_w, box_h) < self._min_subject_px

    def _is_new_subject(self, track_id: int) -> bool:
        """Whether this subject has never been judged.

        Drives batch priority: a person nobody has looked at yet outranks a
        re-check of someone already cleared, so a crowd delays re-checks rather
        than first looks.
        """
        if track_id < 0:
            return True
        with self._memo_lock:
            return track_id not in self._judged_tracks

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
        recent = list(self._inference_times)[-50:]
        avg_ms = sum(recent) / len(recent) if recent else 0.0

        queued = list(self._queued_times)
        avg_queued = sum(queued) / len(queued) if queued else 0.0

        return VLMWorkerStats(
            is_running=self.is_running,
            model_loaded=self._vlm.is_loaded,
            device=self._vlm.device,
            triggers_received=self._triggers_received,
            triggers_processed=self._triggers_processed,
            triggers_stale=self._triggers_stale,
            triggers_coalesced=self._triggers_coalesced,
            triggers_failed=self._triggers_failed,
            triggers_malformed=self._triggers_malformed,
            triggers_frame_lost=self._triggers_frame_lost,
            triggers_backend_error=self._triggers_backend_error,
            triggers_internal_error=self._triggers_internal_error,
            triggers_timed_out=self._triggers_timed_out,
            verdicts_positive=self._verdicts_positive,
            verdicts_negative=self._verdicts_negative,
            instances_judged=self._instances_judged,
            instances_memoised=self._instances_memoised,
            instances_shed=self._instances_shed,
            instances_too_small=self._instances_too_small,
            instances_memoised_spatial=self._instances_memoised_spatial,
            memo_miss_new_track=self._memo_miss_new_track,
            memo_miss_ttl=self._memo_miss_ttl,
            memo_miss_moved=self._memo_miss_moved,
            batches_dropped_unavailable=self._batches_dropped_unavailable,
            backend_unavailable=self._backend_unavailable,
            avg_inference_ms=avg_ms,
            last_inference_ms=self._last_inference_ms,
            avg_queued_ms=avg_queued,
            latency_p50_ms=_percentile(self._latencies, 50),
            latency_p95_ms=_percentile(self._latencies, 95),
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

        self._report_probe_capability()

        subscriber = EventSubscriber(self._zmq_endpoint, topic="detection")
        publisher = EventPublisher(self._result_endpoint)

        try:
            while not self._shutdown_event.is_set():
                batch = self._collect_batch(subscriber)
                if not batch:
                    continue
                try:
                    self._process_batch(batch, publisher, subscriber)
                except Exception as exc:
                    self._triggers_internal_error += 1
                    logger.error(
                        f"VLMWorker: batch processing failed: {exc}",
                        exc_info=True,
                    )

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
        return self._coalesce(pending)

    def _drain_now(self, subscriber: EventSubscriber) -> list[dict]:
        """Take whatever has arrived since the last look, without waiting.

        Called between subjects mid-batch so a person who walks in does not
        have to wait out every re-check queued ahead of them. Returns an empty
        list when the scene is quiet, which is the common case and costs one
        non-blocking poll.
        """
        pending = [payload for _topic, payload in subscriber.drain(max_messages=512)]
        return self._coalesce(pending) if pending else []

    def _coalesce(self, pending: list[dict]) -> list[dict]:
        """Reduce a burst of triggers to the newest one per subject.

        Args:
            pending: Raw payloads in arrival order.

        Returns:
            One payload per subject, newest first, capped at the hard ceiling.
        """
        if not pending:
            return []

        self._triggers_received += len(pending)

        # Keyed by *subject*, not by label. Keying on the label alone was the
        # defect that made a frame of five workers yield one judgement: four
        # of the five were discarded here as "superseded" when they were in
        # fact different people. An untracked payload carries track_id -1, so
        # all of them share one key and the old per-label behaviour is
        # reproduced exactly.
        newest: dict[tuple[str, int], dict] = {}
        for payload in pending:
            key = (payload.get("label", ""), payload.get("track_id", -1))
            current = newest.get(key)
            if current is None or payload.get("timestamp", 0.0) > current.get(
                "timestamp", 0.0
            ):
                newest[key] = payload

        coalesced = len(pending) - len(newest)
        if coalesced:
            self._triggers_coalesced += coalesced
            logger.debug(
                f"Coalesced {coalesced} superseded trigger(s); "
                f"processing {len(newest)} subject(s)."
            )

        batch = sorted(
            newest.values(), key=lambda p: p.get("timestamp", 0.0), reverse=True
        )

        if len(batch) > _BATCH_HARD_CAP:
            logger.warning(
                f"Backlog of {len(batch)} subjects exceeds the batch ceiling; "
                f"carrying {_BATCH_HARD_CAP} forward."
            )
            batch = batch[:_BATCH_HARD_CAP]

        return batch

    def _encode(self, image: Image.Image) -> object:
        """Run the backend's shareable per-image work under a hard timeout.

        Returns:
            An opaque handle to pass to :meth:`_answer`.

        Raises:
            concurrent.futures.TimeoutError: If encoding exceeds the timeout.
        """
        future = self._executor.submit(self._vlm.encode, image)
        return future.result(timeout=self._inference_timeout_s)

    def _answer(self, handle: object, prompt: str) -> str:
        """Ask one question against an encoded image, under a hard timeout.

        The encode and each answer are bounded separately, so a subject with
        several rules can in principle spend more than ``inference_timeout_s``
        in total. That is intended: the timeout is a runaway guard on a single
        wedged model call, not a budget for the subject.

        Raises:
            concurrent.futures.TimeoutError: If inference exceeds the timeout.
        """
        future = self._executor.submit(self._vlm.answer, handle, prompt)
        return future.result(timeout=self._inference_timeout_s)

    def _infer_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[str]:
        """Answer a whole group in one dispatch, under a scaled timeout.

        The budget grows with the batch because a flat per-query timeout
        applied to a batch of four would fire on work that is progressing
        normally. It grows sub-linearly because the point of batching is that
        N rows cost less than N separate calls — if they did not, the timeout
        would be the least of the problems.

        Raises:
            concurrent.futures.TimeoutError: If the batch exceeds its budget.
        """
        budget = self._inference_timeout_s * (1 + 0.5 * (len(images) - 1))
        future = self._executor.submit(self._vlm.query_batch, images, prompts)
        return future.result(timeout=budget)

    # ── Verdict Acquisition ──────────────────────────────────────

    def _report_probe_capability(self) -> None:
        """Say once, at startup, which verdict path this backend will take.

        Without this the fallback is invisible: a backend that cannot probe is
        never *asked*, so it never refuses, so nothing is ever logged — and an
        operator who set ``EYEQ_VLM_PROBE=on`` would watch it silently not
        happen. Announcing it at load is the only point where the answer is
        both known and worth one line.
        """
        if self._probe_mode == "off":
            logger.info("Verdict probing disabled; using generate-and-parse.")
            return

        if isinstance(self._vlm, BinaryProbeBackend):
            logger.info(
                f"Verdict probing enabled on {self._vlm.model_name} "
                f"(threshold {self._probe_threshold:.2f})."
            )
            return

        message = (
            f"{self._vlm.model_name} exposes no token probabilities; using "
            f"generate-and-parse. Note this path defaults to 'no alert' on any "
            f"answer it cannot recognise."
        )
        if self._probe_mode == "on":
            logger.warning(message)
        else:
            logger.info(message)

    def _probing_enabled(self) -> bool:
        """Whether the next query should try the probe path."""
        if self._probe_mode == "off" or self._probe_supported is False:
            return False
        return isinstance(self._vlm, BinaryProbeBackend)

    def _disable_probing(self, reason: object) -> None:
        """Stop probing for the rest of this session.

        Called once, on the first backend refusal. The backend cannot start
        supporting logprobs halfway through a run, so re-checking would only
        buy a wasted round trip per verdict.
        """
        if self._probe_supported is False:
            return

        self._probe_supported = False
        message = (
            f"Verdict probing unavailable on {self._vlm.model_name}; using "
            f"generate-and-parse for the rest of this session. ({reason})"
        )
        # 'on' is an explicit request for probing, so failing it is worth a
        # warning. Under 'auto' the fallback is expected on two of the three
        # backends and is merely informational.
        if self._probe_mode == "on":
            logger.warning(message)
        else:
            logger.info(message)

    def _note_unavailable(self, reason: object) -> None:
        """Record a dropped batch and log the outage once, not per subject.

        ``_collect_batch`` hands over a new batch roughly every second, and a
        refused connection fails fast, so an endpoint that stays down would
        otherwise emit an identical ERROR several times a second for as long
        as the outage lasts. Latch on the first one; :meth:`_record_verdict`
        clears the latch when traffic comes back.
        """
        self._batches_dropped_unavailable += 1

        if self._backend_unavailable:
            return

        self._backend_unavailable = True
        logger.error(
            f"VLM backend unavailable; dropping batches until it returns. "
            f"({reason})"
        )

    def _probe(self, handle: object, prompt: str) -> BinaryProbe:
        """Probe one verdict under the same hard timeout as an answer."""
        future = self._executor.submit(self._vlm.probe_binary, handle, prompt)  # type: ignore[attr-defined]
        return future.result(timeout=self._inference_timeout_s)

    def _probe_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[BinaryProbe]:
        """Probe a whole group under the scaled batch budget."""
        budget = self._inference_timeout_s * (1 + 0.5 * (len(images) - 1))
        future = self._executor.submit(
            self._vlm.probe_binary_batch, images, prompts  # type: ignore[attr-defined]
        )
        return future.result(timeout=budget)

    def _decide(self, handle: object, prompt: str) -> _Answer:
        """Get one verdict, by whichever route this backend supports.

        Probing asks the model for the single bit Stage 2 actually needs and
        reads it off the distribution the prefill already produced. Generating
        asks for prose and then guesses the bit back out of it — and the guess
        has a failure mode measured on this project's own fixtures: Moondream
        answers "Crouching" to "is this person bending over or crouching?",
        which is plainly yes, and the parser's conservative default files it as
        no-alert. A probe cannot produce that class of error, because it never
        has to recognise the wording.

        Raises:
            concurrent.futures.TimeoutError: If inference exceeds the timeout.
        """
        if self._probing_enabled():
            try:
                probe = self._probe(handle, prompt)
                self._verdicts_probed += 1
                return _Answer(probe.verdict, probe.raw, probe.p_yes)
            except ProbeUnsupportedError as exc:
                # Fall through and answer this query the old way rather than
                # dropping it — the transition itself should cost nothing.
                self._disable_probing(exc)

        raw = self._answer(handle, prompt)
        return _Answer(self._verdict_parser.parse(raw), raw)

    def _decide_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[_Answer]:
        """Get a whole group's verdicts, probed or parsed.

        Mirrors :meth:`_decide` exactly, so batch size never changes which
        decision rule is in force — only how many questions ride together.
        """
        if self._probing_enabled():
            try:
                probes = self._probe_batch(images, prompts)
                self._verdicts_probed += len(probes)
                return [_Answer(p.verdict, p.raw, p.p_yes) for p in probes]
            except ProbeUnsupportedError as exc:
                self._disable_probing(exc)

        return [
            _Answer(self._verdict_parser.parse(raw), raw)
            for raw in self._infer_batch(images, prompts)
        ]

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
        """Process a single detection trigger end to end.

        A one-element batch. Kept as its own entry point because a single
        trigger is the natural unit to reason about and to test.
        """
        self._process_batch([payload], publisher)

    def _read_frame(self, event: DetectionEvent) -> np.ndarray | None:
        """Fetch the frame a detection came from, or None if it is gone."""
        frame_result = self._buffer.read_by_timestamp(event.timestamp, tolerance_ms=500.0)
        if frame_result is not None:
            return frame_result[0]

        # Try direct index access as fallback. frame_index is a ring-buffer
        # *slot*, not a monotonic frame number, so once the buffer wraps that
        # slot holds a different frame entirely. Verify the timestamp before
        # trusting it — judging an unrelated frame and filing the verdict
        # against this detection's bbox is worse than judging nothing.
        try:
            frame, ts = self._buffer.read_at_index(event.frame_index)
        except (IndexError, RuntimeError):
            return None

        if frame is None or abs(ts - event.timestamp) > 0.5:
            return None
        return frame

    def _process_batch(
        self,
        payloads: list[dict],
        publisher: EventPublisher,
        subscriber: EventSubscriber | None = None,
    ) -> None:
        """Judge a whole drained batch: crop everything first, then infer.

        The ordering here is the point of the method, not an implementation
        detail. Inference takes seconds; the ring buffer holds seconds. So if
        the frame were fetched immediately before each inference — as it was
        when a trigger was the unit of work — then the second subject in a
        group would find its frame already overwritten and its trigger already
        past the staleness threshold, and would be dropped without ever being
        looked at. That is invisible in a single-subject scene and total in a
        crowded one.

        Cropping every subject up front costs ~1 ms each and a couple of
        megabytes, and makes the buffer irrelevant by the time the slow part
        starts.

        Args:
            payloads: Coalesced detection triggers.
            publisher: Where verdicts are published.
            subscriber: When given, the batch re-polls it between subjects so a
                newly arrived person can overtake queued re-checks. Omitted by
                the single-trigger path and by tests that drive one batch.
        """
        prepared = self._prepare(payloads)
        if not prepared:
            return

        self._judge_prepared(prepared, publisher, subscriber)

    def _prepare(
        self, payloads: list[dict]
    ) -> list[tuple[DetectionEvent, list[dict], np.ndarray, bool]]:
        """Turn raw payloads into croppped, judgeable work items.

        Everything cheap happens here and everything slow happens to the
        result: parse, shed stale groups, consult the memo, read each source
        frame once, and crop every surviving subject. See :meth:`_process_batch`
        for why the cropping cannot be deferred to inference time.

        Returns:
            ``(event, rules, crop, is_new_subject)`` tuples, unordered.
        """
        events: list[DetectionEvent] = []
        for payload in payloads:
            try:
                events.append(DetectionEvent(**payload))
            except Exception as exc:
                self._triggers_malformed += 1
                logger.warning(f"Discarding malformed detection payload: {exc}")

        if not events:
            return []

        # Subjects from one frame share a timestamp exactly — it is copied, not
        # recomputed, per box — so this groups by source frame.
        groups: dict[float, list[DetectionEvent]] = {}
        for event in events:
            groups.setdefault(event.timestamp, []).append(event)

        prepared: list[tuple[DetectionEvent, list[dict], np.ndarray, bool]] = []

        for timestamp in sorted(groups, reverse=True):
            group = groups[timestamp]

            # ── Staleness, once for the whole frame ──────────────
            age = time.monotonic() - timestamp
            if age > self._staleness_threshold:
                self._triggers_stale += len(group)
                logger.warning(
                    f"Discarding {len(group)} stale trigger(s) from {age:.1f}s ago "
                    f"(threshold: {self._staleness_threshold}s)"
                )
                continue

            # ── Decide what is worth asking before touching pixels ──
            pending = [(event, rules) for event in group if (rules := self._rules_to_judge(event))]
            if not pending:
                continue

            frame = self._read_frame(group[0])
            if frame is None:
                self._triggers_frame_lost += len(pending)
                logger.warning(
                    f"Frame not found for {len(pending)} trigger(s) at "
                    f"t={timestamp:.3f}. Buffer has wrapped."
                )
                continue

            for event, rules in pending:
                # Nothing legible to answer about. Skipped rather than judged,
                # and deliberately not remembered — the same person will be
                # worth asking about once they are closer, and a memoised
                # verdict here would suppress that question.
                if self._too_small(frame.shape, event):
                    self._instances_too_small += 1
                    logger.debug(
                        f"Skipping [{event.label}#{event.track_id}]: subject "
                        f"under {self._min_subject_px} px."
                    )
                    continue

                cropped = prepare_vlm_crop(
                    frame,
                    event.bbox,
                    crop_padding=self._crop_padding,
                    focus_dim=self._focus_dim,
                    square_edge=self._square_edge,
                    max_crop_edge=self._max_crop_edge,
                )
                prepared.append(
                    (event, rules, cropped, self._is_new_subject(event.track_id))
                )

            del frame

        return prepared

    def _judge_prepared(
        self,
        prepared: list[tuple[DetectionEvent, list[dict], np.ndarray, bool]],
        publisher: EventPublisher,
        subscriber: EventSubscriber | None = None,
    ) -> None:
        """Judge a prepared batch, letting new arrivals overtake re-checks.

        The priority rule is the same one the sort has always expressed —
        never-judged subjects ahead of re-checks — but it used to be applied
        once, to whatever happened to be in hand when the batch started. With
        inference at seconds per subject and a cap of eight, a batch can run
        for a minute; a person walking in a second after it began was invisible
        to that sort and waited out every re-check ahead of them. Re-polling
        between subjects lets them take the place the sort always intended.

        The cap still bounds how many subjects are judged per batch, so this
        changes *which* subjects get the budget, never how large it is.
        """
        judged = 0
        while prepared and judged < self._max_instances_per_batch:
            if self._shutdown_event.is_set():
                break

            if subscriber is not None:
                prepared.extend(self._prepare(self._newcomers(subscriber, prepared)))

            # Never-judged subjects first, then freshest. Re-applied on every
            # pass so a late arrival is ranked against what is still waiting,
            # not against the queue as it looked when the batch began.
            prepared.sort(key=lambda item: (not item[3], -item[0].timestamp))

            # One subject at a time unless batching is switched on. Taking
            # the chunk after the sort keeps the priority rule intact: a chunk
            # is the front of the queue, never an arbitrary slice of it.
            take = min(self._batch_size, self._max_instances_per_batch - judged)
            chunk = prepared[:take]
            del prepared[:take]
            judged += len(chunk)

            try:
                if len(chunk) == 1:
                    event, rules, cropped, _is_new = chunk[0]
                    ok = self._judge_instance(event, rules, cropped, publisher)
                else:
                    ok = self._judge_chunk(chunk, publisher)

                if not ok:
                    # Either a hung inference — the executor has a single
                    # worker and Moondream cannot be interrupted, so everything
                    # behind it would queue on a thread that is never coming
                    # back — or an endpoint that is down. The outage case has
                    # already logged itself, and throttled; saying "timeout"
                    # again here would both misname it and defeat that.
                    if not self._backend_unavailable:
                        logger.error(
                            "Abandoning the rest of the batch after a timeout."
                        )
                    return
            except Exception as exc:
                self._triggers_internal_error += 1
                logger.error(f"VLMWorker: judging failed: {exc}", exc_info=True)
            finally:
                # Always release accelerator memory back to YOLO
                self._vlm.cleanup()

        if prepared:
            self._instances_shed += len(prepared)
            logger.info(
                f"Judged {judged} subject(s); {len(prepared)} deferred to the "
                f"next batch."
            )

    def _newcomers(
        self,
        subscriber: EventSubscriber,
        prepared: list[tuple[DetectionEvent, list[dict], np.ndarray, bool]],
    ) -> list[dict]:
        """Freshly arrived triggers for subjects not already queued in this batch.

        Filtering by track id matters: the memo only records a verdict once the
        subject has been judged, so a subject still waiting its turn would
        otherwise be prepared a second time and judged twice in one batch.
        """
        waiting = {event.track_id for event, _rules, _crop, _new in prepared}
        return [
            payload
            for payload in self._drain_now(subscriber)
            if payload.get("track_id", -1) not in waiting
        ]

    def _judge_instance(
        self,
        event: DetectionEvent,
        rules: list[dict],
        cropped: np.ndarray,
        publisher: EventPublisher,
    ) -> bool:
        """Ask every pending rule about one subject and publish the verdicts.

        The unbatched path, and the default one. Encodes the crop once and
        answers each rule against that embedding, which is strictly cheaper
        than a batch of one and keeps the vision tower's output shared across
        the subject's rules.

        Returns:
            False if an inference timed out and the batch should be abandoned.
        """
        subject = _Subject(
            event=event,
            cropped=cropped,
            image=Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)),
        )

        self._instances_judged += 1

        # Encode once for the whole subject rather than once per rule. Every
        # rule matching this subject asks about the identical crop, so the
        # vision tower's output is identical too; re-deriving it per rule was
        # pure repetition. Backends with no separable vision half return the
        # image here and lose nothing.
        subject_start = time.monotonic()
        try:
            handle = self._encode(subject.image)
        except FutureTimeoutError:
            self._triggers_timed_out += 1
            self._vlm.request_stop()
            logger.error(
                f"VLM image encoding exceeded {self._inference_timeout_s}s "
                f"for [{event.label}#{event.track_id}]."
            )
            return False
        except Exception as exc:
            self._triggers_backend_error += 1
            logger.error(f"VLM image encoding failed: {exc}")
            return True

        # Charged to the first rule below, since that is the verdict whose
        # production actually paid for it. Later rules are genuinely cheaper
        # and their timings should say so.
        unbilled_encode_ms = (time.monotonic() - subject_start) * 1000.0

        for rule in rules:
            prompt = rule["prompt"]
            rule_id = rule["rule_id"]

            logger.info(
                f"VLM query: [{event.label}#{event.track_id}] "
                f"'{prompt[:80]}' (rule_id={rule_id})"
            )

            start_time = time.monotonic()
            # Measured from the frame's own timestamp, so this covers the whole
            # journey — Stage 1's gating, the ZMQ hop, the drain, the crop, and
            # every subject judged ahead of this one — not just the model call.
            # The first rule's clock starts at the encode, which was work done
            # on its behalf; later rules start at their own query.
            work_start = subject_start if unbilled_encode_ms else start_time
            queued_ms = max(0.0, (work_start - event.timestamp) * 1000.0)
            try:
                answer = self._decide(handle, prompt + VERDICT_SUFFIX)
            except FutureTimeoutError:
                self._triggers_timed_out += 1
                # Ask the backend to abandon generation. Without this the
                # executor's single worker stays pinned on the dead inference
                # and the *next* trigger waits behind it too.
                self._vlm.request_stop()
                logger.error(
                    f"VLM inference exceeded {self._inference_timeout_s}s "
                    f"for rule_id={rule_id}."
                )
                return False
            except VLMUnavailableError as exc:
                # The endpoint is down, not this query. Every rule and every
                # subject behind this one would pay its own doomed round-trip
                # to discover the same thing, so stop here.
                # Deliberately not counted as a failure: an outage is already
                # described by backend_unavailable and
                # batches_dropped_unavailable. Counting it here as well made
                # a single dead endpoint read as many unrelated faults.
                self._note_unavailable(exc)
                return False
            except Exception as exc:
                self._triggers_backend_error += 1
                logger.error(f"VLM inference failed: {exc}")
                continue
            # The encode is billed to whichever rule triggered it, so a
            # subject's total cost stays honest no matter how it is split.
            inference_ms = (time.monotonic() - start_time) * 1000.0 + unbilled_encode_ms
            unbilled_encode_ms = 0.0

            self._record_verdict(
                subject, rule, answer, queued_ms, inference_ms, publisher
            )

        return True

    def _judge_chunk(
        self,
        chunk: list[tuple[DetectionEvent, list[dict], np.ndarray, bool]],
        publisher: EventPublisher,
    ) -> bool:
        """Judge several subjects in one dispatch per distinct prompt.

        Only reached when ``vlm_batch_size`` is above 1. Questions are grouped
        by prompt text before dispatch, for a reason specific to how batching
        is implemented: the backend left-pads every row to the longest prompt
        in the batch, so a same-prompt group pads by zero and its inputs are
        identical to what the sequential path would have built. Mixed prompts
        still work, they simply spend compute on padding.

        The trade this path accepts is the encode hoisting in
        :meth:`_judge_instance`: a subject matched by two rules appears in two
        prompt groups and has its crop encoded twice. With one rule per label,
        which is the norm, nothing is lost.

        Returns:
            False if an inference timed out and the batch should be abandoned.
        """
        subjects = [
            _Subject(
                event=event,
                cropped=cropped,
                image=Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)),
            )
            for event, _rules, cropped, _is_new in chunk
        ]
        self._instances_judged += len(subjects)

        groups: dict[str, list[tuple[_Subject, dict]]] = {}
        for subject, (_event, rules, _cropped, _is_new) in zip(
            subjects, chunk, strict=True
        ):
            for rule in rules:
                groups.setdefault(rule["prompt"], []).append((subject, rule))

        for prompt, units in groups.items():
            logger.info(
                f"VLM batch: {len(units)} subject(s) "
                f"[{', '.join(f'{s.event.label}#{s.event.track_id}' for s, _ in units)}] "
                f"'{prompt[:80]}'"
            )

            start_time = time.monotonic()
            # One clock for the whole group. Every unit in it waited the same
            # amount of time for the dispatch to begin.
            queued = [
                max(0.0, (start_time - subject.event.timestamp) * 1000.0)
                for subject, _rule in units
            ]

            try:
                answers = self._decide_batch(
                    [subject.image for subject, _rule in units],
                    [prompt + VERDICT_SUFFIX] * len(units),
                )
            except FutureTimeoutError:
                # A batched timeout costs every subject in the group, not one.
                # This is the reason vlm_batch_size defaults to 1 and the
                # timeout budget scales with the batch.
                self._triggers_timed_out += len(units)
                self._vlm.request_stop()
                logger.error(
                    f"VLM batch of {len(units)} exceeded its budget; "
                    f"abandoning the rest of the batch."
                )
                return False
            except VLMUnavailableError as exc:
                # Deliberately not counted as a failure: an outage is already
                # described by backend_unavailable and
                # batches_dropped_unavailable. Counting it here as well made
                # a single dead endpoint read as many unrelated faults.
                self._note_unavailable(exc)
                return False
            except Exception as exc:
                self._triggers_backend_error += len(units)
                logger.error(f"VLM batch inference failed: {exc}")
                continue

            batch_ms = (time.monotonic() - start_time) * 1000.0
            # Shared evenly so avg_inference_ms stays comparable to the
            # sequential baseline. The batch is the honest unit of cost; per
            # subject it is an average and nothing finer is knowable.
            per_unit_ms = batch_ms / len(units)

            if len(answers) != len(units):
                self._triggers_backend_error += len(units)
                logger.error(
                    f"VLM batch returned {len(answers)} answer(s) for "
                    f"{len(units)} question(s); discarding the group rather "
                    f"than risk misattributing a verdict."
                )
                continue

            for (subject, rule), answer, queued_ms in zip(
                units, answers, queued, strict=True
            ):
                self._record_verdict(
                    subject,
                    rule,
                    answer,
                    queued_ms,
                    per_unit_ms,
                    publisher,
                    # Not queued + per_unit: nothing in this group existed
                    # until the whole dispatch returned.
                    latency_ms=queued_ms + batch_ms,
                )

        return True

    def _record_verdict(
        self,
        subject: _Subject,
        rule: dict,
        answer: _Answer,
        queued_ms: float,
        inference_ms: float,
        publisher: EventPublisher,
        latency_ms: float | None = None,
    ) -> None:
        """Remember one decided verdict and publish it.

        The tail shared by the sequential and batched paths, and now by the
        probed and parsed ones. All four must record a verdict identically —
        the claim of every alternative path is that it changes only how the
        answer was obtained, so the decision itself is made before this is
        called and this does not know which route produced it.

        Args:
            subject: The cropped subject and its shared evidence JPEG.
            rule: The rule this answer belongs to.
            answer: The decided verdict, its text, and its confidence if known.
            queued_ms: Frame timestamp to the start of the work.
            inference_ms: Model time *charged* to this verdict. For a batch
                this is the batch's cost divided by its size, which is the
                only per-subject figure that means anything.
            publisher: Where the verdict goes.
            latency_ms: Capture to verdict, when it is not simply the sum of
                the two above. A batched verdict is not available until the
                whole batch finishes, so every subject in it waits the full
                batch — amortizing that would understate the wait an operator
                actually sits through by up to the batch size.
        """
        # A verdict landed, so whatever outage was latched is over. Logging
        # the recovery matters as much as logging the failure — otherwise the
        # last thing in the log is an ERROR and nothing says it ended.
        if self._backend_unavailable:
            self._backend_unavailable = False
            logger.info("VLM backend is answering again; resuming batches.")

        event = subject.event
        if latency_ms is None:
            latency_ms = queued_ms + inference_ms

        self._last_inference_ms = inference_ms
        self._inference_times.append(inference_ms)
        self._queued_times.append(queued_ms)
        self._latencies.append(latency_ms)

        verdict = answer.verdict
        raw_response = answer.raw
        self._triggers_processed += 1

        if verdict:
            self._verdicts_positive += 1
        else:
            self._verdicts_negative += 1

        self._remember_verdict(event, rule["rule_id"], verdict)

        # One evidence JPEG per subject, shared by every rule asked about them.
        if not subject.snapshot_written:
            subject.snapshot_path = self._save_snapshot(subject.cropped)
            subject.snapshot_written = True

        result = VLMResult(
            query=VLMQuery(
                detection=event,
                rule_id=rule["rule_id"],
                rule_name=rule.get("rule_name", ""),
                prompt=rule["prompt"],
            ),
            verdict=verdict,
            raw_response=raw_response,
            confidence=answer.confidence,
            inference_time_ms=inference_ms,
            queued_ms=queued_ms,
            latency_ms=latency_ms,
            model_name=self._vlm.model_name,
            snapshot_path=subject.snapshot_path,
        )

        verdict_str = "🔴 YES (VIOLATION)" if verdict else "🟢 NO (clear)"
        confidence_str = (
            f" | p(yes)={answer.confidence:.2f}"
            if answer.confidence is not None
            else ""
        )
        logger.info(
            f"VLM result: {verdict_str} | "
            f"[{event.label}#{event.track_id}] | "
            f"inference={inference_ms:.0f}ms | "
            f"queued={queued_ms:.0f}ms | "
            f"latency={latency_ms:.0f}ms{confidence_str} | "
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
