"""Eye-Q — CLI entry point.

All runtime components (buffer, capture, Stage 1 watchdog, Stage 2 VLM
worker, alert sink) are owned by the FastAPI lifespan in
``app/api/server.py``. This module only translates CLI flags into ``EYEQ_*``
environment overrides and hands control to uvicorn, which means there is
exactly one place that starts and stops the pipeline.

Usage:
    # Start the full system (webcam by default)
    uv run python main.py

    # Point at a video file or RTSP stream
    EYEQ_DEFAULT_SOURCE=demovideo.mp4 uv run python main.py
    uv run python main.py --source rtsp://192.168.1.100:554/stream1

    # Dashboard and API only, no video pipeline
    uv run python main.py --api-only

    # Stage 1 only — runs YOLO without loading the VLM
    uv run python main.py --no-vlm
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

logger = logging.getLogger("eyeq")


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Translate CLI flags into ``EYEQ_*`` environment variables.

    This must run **before** the first ``get_settings()`` call — settings are
    cached with ``lru_cache``, so a later write would be ignored.
    """
    if args.source is not None:
        os.environ["EYEQ_DEFAULT_SOURCE"] = args.source
    if args.host is not None:
        os.environ["EYEQ_API_HOST"] = args.host
    if args.port is not None:
        os.environ["EYEQ_API_PORT"] = str(args.port)
    if args.device is not None:
        os.environ["EYEQ_DEVICE"] = args.device
    if args.api_only:
        os.environ["EYEQ_ENABLE_PIPELINE"] = "false"
    if args.no_vlm:
        os.environ["EYEQ_ENABLE_VLM"] = "false"


def main() -> None:
    """Parse CLI arguments and start the Eye-Q system."""
    parser = argparse.ArgumentParser(
        description="Eye-Q — Open-vocabulary VLM Retrofit Appliance",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: RTSP URI, file path, or webcam index (default: from settings)",
    )
    parser.add_argument(
        "--host", type=str, default=None, help="API server bind address"
    )
    parser.add_argument("--port", type=int, default=None, help="API server port")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cuda", "cpu"],
        help="Inference device for Stage 1",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the API and dashboard (no video pipeline)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip the Stage 2 VLM worker (Stage 1 only mode)",
    )

    args = parser.parse_args()
    _apply_cli_overrides(args)

    # Imported after the overrides land so the cached settings pick them up.
    from app.config.settings import get_settings

    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(
        f"Starting Eye-Q on http://{settings.api_host}:{settings.api_port} "
        f"(pipeline={'on' if settings.enable_pipeline else 'off'}, "
        f"vlm={'on' if settings.enable_vlm else 'off'})"
    )

    # uvicorn owns SIGINT/SIGTERM and drives the lifespan's shutdown path,
    # so no separate signal handling is needed here.
    uvicorn.run(
        "app.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
