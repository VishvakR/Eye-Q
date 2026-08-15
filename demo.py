"""Eye-Q Visual Demo Runner — live bounding box visualization.

Runs the full Stage 1 pipeline with an OpenCV window showing:
    - Live video feed with YOLO detection bounding boxes
    - ROI polygon overlay (green border)
    - Detection labels, confidence scores, and trigger indicators
    - Real-time FPS and stats overlay

Supports:
    - Webcam (device index)
    - Video file (MP4, AVI, etc.)
    - RTSP stream

Usage:
    # Webcam (default)
    uv run python demo.py

    # Video file
    uv run python demo.py --source /path/to/video.mp4

    # RTSP stream
    uv run python demo.py --source rtsp://192.168.1.100:554/stream1

    # Custom settings
    uv run python demo.py --source video.mp4 --device mps --labels person,car --cooldown 2.0

Controls:
    - Press 'q' or ESC to quit
    - Press 'r' to reset cooldowns
    - Press SPACE to pause/resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from app.core.buffer import RollingFrameBuffer
from app.core.capture import VideoCapture
from app.core.ipc import EventPublisher, EventSubscriber
from app.core.roi import RegionOfInterest
from app.pipelines.detector import YOLOWatchdog
from app.schemas.events import BoundingBox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ── Colors (BGR for OpenCV) ──────────────────────────────────────

COLOR_DETECTION = (0, 255, 0)      # Green — active detection
COLOR_TRIGGERED = (0, 100, 255)    # Orange — detection that triggered VLM
COLOR_ROI = (255, 200, 0)          # Cyan — ROI boundary
COLOR_TEXT_BG = (0, 0, 0)          # Black — text background
COLOR_TEXT = (255, 255, 255)       # White — text
COLOR_STATS_BG = (30, 30, 30)     # Dark gray — stats bar
COLOR_FPS = (0, 255, 200)         # Mint — FPS text


def draw_detection(
    frame: np.ndarray,
    bbox: BoundingBox,
    label: str,
    confidence: float,
    triggered: bool = False,
) -> None:
    """Draw a bounding box with label on the frame.

    Args:
        frame: BGR frame to draw on (modified in-place).
        bbox: Normalized bounding box coordinates.
        label: Detection class label.
        confidence: Detection confidence [0, 1].
        triggered: If True, use the triggered color (orange).
    """
    h, w = frame.shape[:2]
    x1 = int(bbox.x1 * w)
    y1 = int(bbox.y1 * h)
    x2 = int(bbox.x2 * w)
    y2 = int(bbox.y2 * h)

    color = COLOR_TRIGGERED if triggered else COLOR_DETECTION
    thickness = 3 if triggered else 2

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw corner accents (modern style)
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_thick = thickness + 1
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)

    # Label text
    text = f"{label} {confidence:.0%}"
    if triggered:
        text = f"[TRIGGER] {text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Text background
    label_y = max(y1 - 8, th + 8)
    cv2.rectangle(
        frame,
        (x1, label_y - th - 6),
        (x1 + tw + 8, label_y + 4),
        color,
        -1,
    )
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
    pts = np.array(
        [(int(x * w), int(y * h)) for x, y in roi.vertices],
        dtype=np.int32,
    )

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (255, 200, 0, 40))
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    # Border
    cv2.polylines(frame, [pts], isClosed=True, color=COLOR_ROI, thickness=2)

    # Label
    centroid_x = int(sum(x for x, _ in roi.vertices) / len(roi.vertices) * w)
    centroid_y = int(sum(y for _, y in roi.vertices) / len(roi.vertices) * h)
    cv2.putText(
        frame,
        f"ROI: {roi.name}",
        (centroid_x - 30, centroid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_ROI,
        1,
        cv2.LINE_AA,
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
    """Draw a stats overlay bar at the bottom of the frame.

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

    # Background bar
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_STATS_BG, -1)

    # Stats text
    parts = [
        f"YOLO: {fps:.1f} FPS",
        f"Frames: {frames_processed}",
        f"Detections: {detections}",
        f"Triggers: {triggers}",
        f"Device: {device.upper()}",
    ]
    if paused:
        parts.insert(0, "|| PAUSED")

    text = "  |  ".join(parts)

    cv2.putText(
        frame,
        text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_FPS,
        1,
        cv2.LINE_AA,
    )

    # Eye-Q branding
    cv2.putText(
        frame,
        "Eye-Q",
        (w - 70, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 100, 255),
        2,
        cv2.LINE_AA,
    )


def draw_title_bar(frame: np.ndarray, source: str) -> None:
    """Draw a title bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 32

    cv2.rectangle(frame, (0, 0), (w, bar_h), COLOR_STATS_BG, -1)

    src_display = source if len(source) < 60 else f"...{source[-57:]}"
    cv2.putText(
        frame,
        f"Eye-Q Stage 1 | Source: {src_display}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    # Controls hint
    cv2.putText(
        frame,
        "Q:quit  R:reset  SPACE:pause",
        (w - 280, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


def run_demo(args: argparse.Namespace) -> None:
    """Run the visual demo."""
    source: str | int = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Parse target labels
    target_labels: set[str] | None = None
    if args.labels:
        target_labels = {l.strip() for l in args.labels.split(",")}
        logger.info(f"Target labels: {target_labels}")

    # Set up ROI
    if args.roi:
        parts = [float(x) for x in args.roi.split(",")]
        if len(parts) == 4:
            roi = RegionOfInterest.from_rect(*parts, name="custom")
        else:
            logger.error("ROI must be 4 values: x1,y1,x2,y2")
            sys.exit(1)
    else:
        roi = RegionOfInterest.full_frame()

    # Resolution
    res_w, res_h = args.width, args.height

    logger.info(f"Starting demo: source={source}, resolution={res_w}×{res_h}")

    # ── Initialize components ────────────────────────────────────
    buf = RollingFrameBuffer(
        capacity=150,
        frame_height=res_h,
        frame_width=res_w,
        buffer_id="demo_vis",
    )

    cap = VideoCapture(
        source=source,
        buffer=buf,
        camera_id="demo",
        target_fps=30,
    )
    cap.start()

    zmq_port = args.zmq_port
    endpoint = f"tcp://127.0.0.1:{zmq_port}"
    pub = EventPublisher(endpoint)
    sub = EventSubscriber(endpoint, topic="detection")
    time.sleep(0.3)

    watchdog = YOLOWatchdog(
        buffer=buf,
        publisher=pub,
        device=args.device,
        model_name=args.model,
        confidence_threshold=args.confidence,
        target_labels=target_labels,
        roi=roi,
        cooldown_seconds=args.cooldown,
        camera_id="demo",
        target_fps=args.yolo_fps,
    )
    watchdog.start()

    # Wait for model to load
    logger.info("Waiting for YOLO model to load...")
    time.sleep(2.0)

    # ── Main visualization loop ──────────────────────────────────
    window_name = "Eye-Q Stage 1 — YOLO Watchdog"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(res_w, 1280), min(res_h, 720))

    paused = False
    recent_triggers: list[dict] = []
    trigger_display_duration = 2.0  # Show trigger bbox for 2 seconds

    source_str = str(source)
    logger.info("Demo running. Press 'q' to quit.")

    try:
        while True:
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("r"):
                watchdog._cooldown.reset()
                logger.info("Cooldowns reset.")
            elif key == ord(" "):
                paused = not paused

            if paused:
                # Still show the last frame
                time.sleep(0.03)
                continue

            # Read latest frame from buffer
            try:
                frame, ts, idx = buf.read_latest()
            except RuntimeError:
                time.sleep(0.05)
                continue

            # Check for new detection events
            while True:
                event = sub.receive(timeout_ms=0)
                if event is None:
                    break
                _, data = event
                data["_received_at"] = time.monotonic()
                recent_triggers.append(data)

            # Clean up old triggers
            now = time.monotonic()
            recent_triggers = [
                t for t in recent_triggers
                if now - t["_received_at"] < trigger_display_duration
            ]

            # ── Draw on frame ────────────────────────────────────

            display = frame.copy()

            # Draw ROI
            if roi.name != "full_frame":
                draw_roi(display, roi)

            # Run YOLO on this frame for visualization (non-trigger detections)
            # We use the watchdog's model directly for real-time bbox drawing
            if watchdog._model is not None:
                results = watchdog._model.predict(
                    frame,
                    device=watchdog._device,
                    imgsz=watchdog._imgsz,
                    conf=args.confidence,
                    verbose=False,
                )
                frame_h, frame_w = frame.shape[:2]

                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        label = result.names.get(cls_id, f"class_{cls_id}")

                        xyxy = box.xyxy[0].tolist()
                        bbox = BoundingBox(
                            x1=max(0.0, min(1.0, xyxy[0] / frame_w)),
                            y1=max(0.0, min(1.0, xyxy[1] / frame_h)),
                            x2=max(0.0, min(1.0, xyxy[2] / frame_w)),
                            y2=max(0.0, min(1.0, xyxy[3] / frame_h)),
                        )

                        # Check if this detection is a recent trigger
                        is_triggered = any(
                            abs(t["bbox"]["x1"] - bbox.x1) < 0.05
                            and abs(t["bbox"]["y1"] - bbox.y1) < 0.05
                            and t["label"] == label
                            for t in recent_triggers
                        )

                        # Filter by labels for display if set
                        if target_labels and label not in target_labels:
                            # Draw non-target detections faintly
                            draw_detection(display, bbox, label, confidence, triggered=False)
                        else:
                            draw_detection(display, bbox, label, confidence, triggered=is_triggered)

            # Draw stats bar
            stats = watchdog.stats
            draw_stats_bar(
                display,
                fps=stats.current_fps,
                frames_processed=stats.frames_processed,
                detections=stats.detections_total,
                triggers=stats.detections_published,
                device=stats.device,
                paused=paused,
            )

            # Draw title bar
            draw_title_bar(display, source_str)

            # Show frame
            cv2.imshow(window_name, display)

            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    # ── Cleanup ──────────────────────────────────────────────────
    logger.info("Shutting down...")
    cv2.destroyAllWindows()
    watchdog.stop()
    cap.stop()
    pub.close()
    sub.close()
    buf.close()
    buf.unlink()
    logger.info("Clean shutdown.")


def main() -> None:
    """Parse CLI arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Eye-Q Stage 1 Visual Demo — YOLO Watchdog with bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python demo.py                                    # Webcam
    uv run python demo.py --source video.mp4                 # Video file
    uv run python demo.py --source video.mp4 --labels person,car
    uv run python demo.py --source 0 --roi 0.1,0.1,0.9,0.9  # Custom ROI
    uv run python demo.py --device cpu --confidence 0.5       # CPU mode, higher threshold
        """,
    )

    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: device index (0), file path, or RTSP URI (default: 0)",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Inference device (default: mps)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLO model file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated target labels (default: all). E.g.: person,car,dog",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Minimum detection confidence (default: 0.25)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0,
        help="Per-label trigger cooldown in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--roi", type=str, default=None,
        help="ROI as 'x1,y1,x2,y2' normalized coords (default: full frame)",
    )
    parser.add_argument(
        "--yolo-fps", type=float, default=15.0,
        help="YOLO inference target FPS (default: 15.0)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture height (default: 720)",
    )
    parser.add_argument(
        "--zmq-port", type=int, default=5558,
        help="ZMQ port for IPC (default: 5558)",
    )

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
