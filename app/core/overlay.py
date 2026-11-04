"""Frame annotation helpers shared by the MJPEG stream and the CLI demo.

Both the dashboard's live preview (``/api/stream/{camera_id}.mjpg``) and
[demo.py](demo.py) draw the same thing on top of raw frames: detection boxes,
the ROI polygon, and a stats bar. These live here so there is exactly one
implementation of that visual language.

All functions mutate the frame in place and expect BGR (OpenCV) byte order.

Example:
    >>> draw_roi(frame, RegionOfInterest.full_frame())
    >>> draw_detection(frame, bbox, "person", 0.87, triggered=True)
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.roi import RegionOfInterest
from app.schemas.events import BoundingBox

# ── Colors (BGR for OpenCV) ──────────────────────────────────────

COLOR_DETECTION = (0, 255, 0)      # Green — active detection
COLOR_TRIGGERED = (0, 100, 255)    # Orange — detection that triggered the VLM
COLOR_ROI = (255, 200, 0)          # Cyan — ROI boundary
COLOR_TEXT_BG = (0, 0, 0)          # Black — text outline
COLOR_TEXT = (255, 255, 255)       # White — text
COLOR_STATS_BG = (30, 30, 30)      # Dark gray — stats bar
COLOR_FPS = (0, 255, 200)          # Mint — FPS text


def draw_detection(
    frame: np.ndarray,
    bbox: BoundingBox,
    label: str,
    confidence: float,
    triggered: bool = False,
    track_id: int = -1,
) -> None:
    """Draw a bounding box with a label on the frame.

    Args:
        frame: BGR frame to draw on (modified in-place).
        bbox: Normalized bounding box coordinates.
        label: Detection class label.
        confidence: Detection confidence [0, 1].
        triggered: If True, use the triggered color and heavier stroke.
    """
    h, w = frame.shape[:2]
    x1 = int(bbox.x1 * w)
    y1 = int(bbox.y1 * h)
    x2 = int(bbox.x2 * w)
    y2 = int(bbox.y2 * h)

    color = COLOR_TRIGGERED if triggered else COLOR_DETECTION
    thickness = 3 if triggered else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Corner accents
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_thick = thickness + 1
    for cx, cy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, corner_thick)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, corner_thick)

    # The subject id makes tracking legible on the live feed: an id that flickers
    # or swaps as people cross is visible here long before it shows up as a
    # duplicate alert.
    subject = f"{label}#{track_id}" if track_id >= 0 else label
    text = f"{subject} {confidence:.0%}"
    if triggered:
        text = f"[TRIGGER] {text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (tw, th), _baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    label_y = max(y1 - 8, th + 8)
    cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), color, -1)
    cv2.putText(
        frame, text, (x1 + 4, label_y - 2),
        font, font_scale, COLOR_TEXT_BG, font_thickness + 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x1 + 4, label_y - 2),
        font, font_scale, COLOR_TEXT, font_thickness, cv2.LINE_AA,
    )


def draw_roi(frame: np.ndarray, roi: RegionOfInterest) -> None:
    """Draw the ROI polygon overlay on the frame.

    Args:
        frame: BGR frame to draw on (modified in-place).
        roi: Region of interest to visualize.
    """
    h, w = frame.shape[:2]
    pts = np.array([(int(x * w), int(y * h)) for x, y in roi.vertices], dtype=np.int32)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (255, 200, 0))
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    cv2.polylines(frame, [pts], isClosed=True, color=COLOR_ROI, thickness=2)

    centroid_x = int(sum(x for x, _ in roi.vertices) / len(roi.vertices) * w)
    centroid_y = int(sum(y for _, y in roi.vertices) / len(roi.vertices) * h)
    cv2.putText(
        frame, f"ROI: {roi.name}", (centroid_x - 30, centroid_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1, cv2.LINE_AA,
    )


def draw_stats_bar(
    frame: np.ndarray,
    fps: float,
    frames_processed: int,
    detections: int,
    triggers: int,
    device: str,
    paused: bool = False,
) -> None:
    """Draw a stats overlay bar along the bottom of the frame.

    Args:
        frame: BGR frame to draw on (modified in-place).
        fps: Current YOLO FPS.
        frames_processed: Total frames processed by YOLO.
        detections: Total raw detections.
        triggers: Total published triggers.
        device: Inference device name.
        paused: Whether the display is paused.
    """
    h, w = frame.shape[:2]
    bar_h = 36

    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_STATS_BG, -1)

    parts = [
        f"YOLO: {fps:.1f} FPS",
        f"Frames: {frames_processed}",
        f"Detections: {detections}",
        f"Triggers: {triggers}",
        f"Device: {device.upper()}",
    ]
    if paused:
        parts.insert(0, "|| PAUSED")

    cv2.putText(
        frame, "  |  ".join(parts), (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FPS, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Eye-Q", (w - 70, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2, cv2.LINE_AA,
    )


def draw_title_bar(frame: np.ndarray, source: str, hint: str = "") -> None:
    """Draw a title bar along the top of the frame.

    Args:
        frame: BGR frame to draw on (modified in-place).
        source: Video source description shown in the bar.
        hint: Optional right-aligned hint text (e.g., keyboard controls).
    """
    h, w = frame.shape[:2]
    bar_h = 32

    cv2.rectangle(frame, (0, 0), (w, bar_h), COLOR_STATS_BG, -1)

    src_display = source if len(source) < 60 else f"...{source[-57:]}"
    cv2.putText(
        frame, f"Eye-Q Stage 1 | Source: {src_display}", (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
    )

    if hint:
        cv2.putText(
            frame, hint, (w - 280, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
        )
