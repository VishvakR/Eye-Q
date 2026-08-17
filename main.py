"""Eye-Q — Main Orchestrator & CLI Entry Point.

Wires all system components together and manages the full lifecycle:
    1. Initialize database
    2. Create shared-memory frame buffer
    3. Start video capture thread(s)
    4. Start YOLO watchdog (Stage 1)
    5. Start VLM worker (Stage 2)
    6. Start FastAPI server
    7. Handle graceful shutdown on SIGINT/SIGTERM

Usage:
    # Start the full system
    uv run python main.py

    # With custom settings
    EYEQ_DEVICE=mps EYEQ_DEFAULT_SOURCE=0 uv run python main.py

    # Start just the API server (no video pipeline)
    uv run python main.py --api-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eyeq")


def main() -> None:
    """Parse CLI arguments and start the Eye-Q system."""
    parser = argparse.ArgumentParser(
        description="Eye-Q — Open-vocabulary VLM Retrofit Appliance",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI server (no video pipeline)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="API server bind address (default: from settings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="API server port (default: from settings)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM worker startup (Stage 1 only mode)",
    )

    args = parser.parse_args()

    if args.api_only:
        _run_api_only(args)
    else:
        _run_full_system(args)


def _run_api_only(args: argparse.Namespace) -> None:
    """Start only the FastAPI server without the video pipeline."""
    from app.config.settings import get_settings

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    logger.info(f"Starting Eye-Q API server (api-only mode) on {host}:{port}")

    uvicorn.run(
        "app.api.server:app",
        host=host,
        port=port,
        log_level="info",
    )


def _run_full_system(args: argparse.Namespace) -> None:
    """Start the full Eye-Q system with all components."""
    from app.config.settings import get_settings
    from app.core.buffer import RollingFrameBuffer
    from app.core.capture import VideoCapture
    from app.core.ipc import EventPublisher
    from app.pipelines.detector import YOLOWatchdog

    settings = get_settings()

    logger.info("=" * 60)
    logger.info("  Eye-Q — Open-Vocabulary VLM Retrofit Appliance")
    logger.info("=" * 60)
    logger.info(f"  Device:     {settings.inference_device}")
    logger.info(f"  Source:     {settings.default_source}")
    logger.info(f"  Resolution: {settings.frame_width}×{settings.frame_height}")
    logger.info(f"  Buffer:     {settings.buffer_capacity} frames")
    logger.info(f"  YOLO:       {settings.yolo_model}")
    logger.info(f"  VLM:        {settings.vlm_model}")
    logger.info(f"  API:        http://{settings.api_host}:{settings.api_port}")
    logger.info("=" * 60)

    # ── Initialize components ────────────────────────────────────

    # 1. Frame buffer
    buffer = RollingFrameBuffer(
        capacity=settings.buffer_capacity,
        frame_height=settings.frame_height,
        frame_width=settings.frame_width,
        buffer_id="main",
    )

    # 2. ZMQ publisher
    publisher = EventPublisher(settings.zmq_endpoint)

    # 3. Video capture
    source: str | int = settings.default_source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    capture = VideoCapture(
        source=source,
        buffer=buffer,
        camera_id="default",
        target_fps=settings.target_fps,
    )
    capture.start()

    # 4. YOLO watchdog
    watchdog = YOLOWatchdog(
        buffer=buffer,
        publisher=publisher,
        device=settings.inference_device,
        model_name=settings.yolo_model,
        confidence_threshold=settings.yolo_confidence,
        imgsz=settings.yolo_imgsz,
        cooldown_seconds=settings.detection_cooldown_s,
        camera_id="default",
    )
    watchdog.start()

    # 5. VLM worker (optional)
    vlm_worker = None
    if not args.no_vlm:
        from app.pipelines.vlm_worker import VLMWorker

        vlm_worker = VLMWorker(
            buffer=buffer,
            zmq_endpoint=settings.zmq_endpoint,
            publisher=publisher,
            device=settings.inference_device,
            model_id=settings.vlm_model,
            model_revision=settings.vlm_revision,
            staleness_threshold_s=settings.vlm_staleness_s,
            crop_padding=settings.vlm_crop_padding,
        )
        vlm_worker.start()

    # ── Attach components to FastAPI app state ───────────────────
    from app.api.server import app

    app.state.buffer = buffer
    app.state.watchdog = watchdog
    app.state.vlm_worker = vlm_worker
    app.state.captures = [capture]

    # ── Shutdown handler ─────────────────────────────────────────
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ── Start FastAPI server in a thread ─────────────────────────
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    server_config = uvicorn.Config(
        "app.api.server:app",
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(
        target=server.run,
        name="UvicornServer",
        daemon=True,
    )
    server_thread.start()

    logger.info(
        f"Eye-Q fully operational. Dashboard: http://{host}:{port}"
    )

    # ── Wait for shutdown signal ─────────────────────────────────
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass

    # ── Graceful shutdown ────────────────────────────────────────
    logger.info("Shutting down Eye-Q...")

    if vlm_worker is not None:
        vlm_worker.stop()

    watchdog.stop()
    capture.stop()
    publisher.close()

    # Server shutdown
    server.should_exit = True
    server_thread.join(timeout=5.0)

    buffer.close()
    buffer.unlink()

    logger.info("Eye-Q shut down cleanly.")


if __name__ == "__main__":
    main()
