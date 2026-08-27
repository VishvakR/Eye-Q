"""Pydantic schemas for inter-component event messaging.

These models define the JSON structure of events exchanged between
the YOLO watchdog (Stage 1), the VLM worker (Stage 2), and the
control plane via ZeroMQ PUB/SUB.

All coordinates use **normalized** values (0.0 to 1.0) relative to
frame dimensions, making them resolution-independent.

Example:
    >>> event = DetectionEvent(
    ...     timestamp=1723.456,
    ...     camera_id="warehouse_cam_1",
    ...     label="person",
    ...     confidence=0.87,
    ...     bbox=BoundingBox(x1=0.12, y1=0.34, x2=0.56, y2=0.78),
    ...     frame_index=42,
    ... )
    >>> event.model_dump()  # JSON-serializable dict
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0).

    Coordinates are relative to the frame dimensions, making them
    resolution-independent. ``(x1, y1)`` is the top-left corner
    and ``(x2, y2)`` is the bottom-right corner.
    """

    x1: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalized).")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalized).")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right edge (normalized).")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (normalized).")

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box as ``(cx, cy)``."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        """Normalized width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Normalized height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Normalized area of the bounding box."""
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box.

        Used to decide whether two boxes in consecutive frames describe the
        same object, and whether a tracked subject has moved far enough since
        its last verdict to be worth re-judging.

        Returns:
            Overlap ratio in [0, 1]. Zero when the boxes are disjoint, and
            zero when either box is degenerate — a box with no area cannot
            meaningfully overlap anything, and returning 0.0 keeps callers
            free of a division guard.
        """
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0

        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def overlap_ratio(self, other: BoundingBox) -> float:
        """Intersection over the *smaller* box's area — containment, not similarity.

        :meth:`iou` answers "are these the same object", which is the right
        question for tracking and the wrong one for co-occurrence. A cell phone
        fully inside a person's box scores about 0.028 on IoU — not because the
        phone is barely inside the person, but because the person is roughly
        thirty-six times larger and IoU divides by the union. Any IoU threshold
        loose enough to admit that pair is also loose enough to admit two boxes
        that merely graze at a corner.

        Dividing by the smaller area asks what the operator means — "is the
        phone in the person?" — and returns ~1.0 for full containment at any
        size ratio. Symmetric, because ``min`` is, so a rule never depends on
        which box the caller named first.

        Returns:
            Ratio in [0, 1]. Zero when the boxes are disjoint, and zero when
            either box is degenerate — matching :meth:`iou` so callers need no
            division guard.
        """
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0

        smaller = min(self.area, other.area)
        return inter / smaller if smaller > 0.0 else 0.0

    def pixel_size(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """The box's own extent in captured pixels, before any crop padding.

        Both stages need this and neither should own it. Stage 1 asks whether a
        detection is worth publishing at all; Stage 2 asks whether a subject is
        worth an inference. Computed separately the two could disagree about
        the same box by a pixel, and Stage 2's floor would either fire on
        something Stage 1 passed or go permanently dead — neither of which any
        counter would reveal.

        Truncating rather than rounding is deliberate: it is what
        ``subject_pixel_size`` has always done, and a floor is a floor.
        """
        return int(self.width * frame_w), int(self.height * frame_h)


class DetectionEvent(BaseModel):
    """Published by Stage 1 (YOLO Watchdog) when a relevant object is detected.

    This event triggers the Stage 2 VLM worker to fetch the corresponding
    frame from the buffer, crop it, and run VLM inference.

    ZMQ Topic: ``"detection"``
    """

    timestamp: float = Field(
        ..., description="Monotonic timestamp (time.monotonic()) of the captured frame."
    )
    camera_id: str = Field(..., description="Unique identifier of the source camera.")
    label: str = Field(..., description="COCO class label (e.g., 'person', 'car').")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score."
    )
    bbox: BoundingBox = Field(..., description="Normalized bounding box of the detection.")
    frame_index: int = Field(
        ..., ge=0, description="Slot index in the ring buffer for direct access."
    )
    # Deliberately optional and deliberately unconstrained: -1 must validate.
    # Stage 2 reconstructs events with DetectionEvent(**payload), so a default
    # keeps an untracked publisher wire-compatible — and -1 is the value that
    # makes every per-instance code path downstream fall back to today's
    # per-label behaviour, which is how tracking is switched off.
    track_id: int = Field(
        default=-1,
        description=(
            "Stable per-instance id assigned by Stage 1's tracker, so each "
            "subject is judged rather than only the first of its label. "
            "-1 when tracking is disabled."
        ),
    )


class VLMQuery(BaseModel):
    """Internal model binding a detection event to a user-defined rule.

    This is created by the VLM worker when it matches a detection
    against active rules from the database.
    """

    detection: DetectionEvent = Field(
        ..., description="The detection event that triggered this query."
    )
    rule_id: int = Field(..., description="Database ID of the matching rule.")
    rule_name: str = Field(default="", description="Human-readable name of the rule.")
    prompt: str = Field(
        ..., description="User's plain-English question for the VLM."
    )


class VLMResult(BaseModel):
    """Published by Stage 2 (VLM Worker) after inference.

    Contains the binary verdict (violation or clear), the raw VLM
    response text, and performance metrics.

    ZMQ Topic: ``"vlm_result"``
    """

    query: VLMQuery = Field(..., description="The query that produced this result.")
    verdict: bool = Field(
        ..., description="True if the VLM confirms a violation, False otherwise."
    )
    raw_response: str = Field(..., description="Full text output from the VLM.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Probability the model assigned to 'Yes', renormalized over the "
            "Yes/No pair, when the verdict was read from its token "
            "probabilities. None when the verdict came from parsing generated "
            "text, which carries no usable strength — a parsed 'Yes' and a "
            "barely-preferred 'Yes' are the same string. Lets an operator see "
            "which alerts were close calls."
        ),
    )
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Wall-clock inference time in milliseconds."
    )
    queued_ms: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Milliseconds between the frame being captured and this inference "
            "starting. Everything the appliance chose to do before answering "
            "this question — draining, coalescing, cropping, and every subject "
            "judged earlier in the same batch — lands here, so it is the term "
            "that grows under load while inference_time_ms stays flat."
        ),
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Milliseconds from frame capture to verdict: queued_ms plus the "
            "inference. This is the number an operator experiences, and the "
            "only one that reveals a backlog."
        ),
    )
    model_name: str = Field(
        default="moondream-2b-2025-04-14-4bit",
        description="Name of the VLM model used.",
    )
    snapshot_path: str | None = Field(
        default=None,
        description="Path to the saved evidence JPEG for this verdict, if written.",
    )


class SystemStatus(BaseModel):
    """System health status broadcast on the ``"status"`` ZMQ topic
    and exposed via the ``GET /api/status`` REST endpoint.
    """

    buffer_occupancy: float = Field(
        ..., ge=0.0, le=1.0, description="Ring buffer fill ratio."
    )
    detector_fps: float = Field(
        ..., ge=0.0, description="Current YOLO inference throughput."
    )
    vlm_queue_depth: int = Field(
        ..., ge=0, description="Number of pending VLM inference requests."
    )
    active_cameras: int = Field(
        ..., ge=0, description="Number of connected camera streams."
    )
    active_rules: int = Field(
        ..., ge=0, description="Number of enabled alert rules."
    )
    uptime_seconds: float = Field(
        ..., ge=0.0, description="Time since system startup."
    )
    inference_device: str = Field(
        default="cpu", description="Active inference device (mps/cuda/cpu)."
    )
