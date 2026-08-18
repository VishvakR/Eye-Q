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

from app.core.buffer import RollingFrameBuffer
from app.core.capture import VideoCapture
from app.core.ipc import EventPublisher, EventSubscriber
from app.core.overlay import (
    draw_detection,
    draw_roi,
    draw_stats_bar,
    draw_title_bar,
)
from app.core.roi import RegionOfInterest
from app.pipelines.detector import YOLOWatchdog
from app.schemas.events import BoundingBox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


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
        target_labels = {lbl.strip() for lbl in args.labels.split(",")}
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
            draw_title_bar(display, source_str, hint="Q:quit  R:reset  SPACE:pause")

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
