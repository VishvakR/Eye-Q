"""Centralized application configuration for Eye-Q.

All settings are loaded from environment variables (prefixed with EYEQ_),
a .env file in the project root, or their default values — in that priority order.

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.inference_device)  # "mps", "cuda", or "cpu"
"""

from __future__ import annotations

import functools
import logging
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def detect_inference_device() -> str:
    """Determine the best available inference device.

    Priority: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.

    Returns:
        Device string compatible with both PyTorch and Ultralytics.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed — defaulting to CPU inference device.")
        return "cpu"

    # Apple Silicon (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Verify MPS is actually functional, not just reported as available
            _test = torch.zeros(1, device="mps")
            del _test
            logger.info("Inference device: MPS (Apple Silicon)")
            return "mps"
        except Exception as exc:
            logger.warning(f"MPS reported available but non-functional: {exc}")

    # NVIDIA CUDA
    if torch.cuda.is_available():
        logger.info(f"Inference device: CUDA ({torch.cuda.get_device_name(0)})")
        return "cuda"

    logger.info("Inference device: CPU (no GPU acceleration available)")
    return "cpu"


class Settings(BaseSettings):
    """Centralized application configuration.

    Values are loaded in this priority (highest first):
        1. Environment variables (prefixed with ``EYEQ_``)
        2. ``.env`` file in the project root
        3. Default values defined below

    Example ``.env`` file::

        EYEQ_DEFAULT_SOURCE=rtsp://192.168.1.100:554/stream1
        EYEQ_BUFFER_CAPACITY=150
        EYEQ_DEVICE=mps
    """

    model_config = SettingsConfigDict(
        env_prefix="EYEQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Video ────────────────────────────────────────────────────
    default_source: str = Field(
        default="0",
        description="Default camera source: RTSP URI, file path, or device index (as string).",
    )
    frame_width: int = Field(default=1920, ge=160, le=7680, description="Capture frame width.")
    frame_height: int = Field(default=1080, ge=120, le=4320, description="Capture frame height.")
    target_fps: int = Field(default=30, ge=1, le=120, description="Target capture frame rate.")

    # ── Buffer ───────────────────────────────────────────────────
    buffer_capacity: int = Field(
        default=300,
        ge=10,
        le=3000,
        description="Number of frames in the shared-memory ring buffer.",
    )

    # ── ZMQ ──────────────────────────────────────────────────────
    zmq_endpoint: str = Field(
        default="tcp://127.0.0.1:5557",
        description="ZeroMQ PUB/SUB endpoint for inter-component IPC.",
    )

    # ── YOLO (Stage 1) ───────────────────────────────────────────
    yolo_model: str = Field(default="yolov8n.pt", description="YOLO model filename.")
    yolo_confidence: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Minimum detection confidence."
    )
    yolo_imgsz: int = Field(default=640, description="YOLO input image size.")
    detection_cooldown_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum seconds between triggers for the same label.",
    )

    # ── VLM (Stage 2) ───────────────────────────────────────────
    vlm_model: str = Field(
        default="vikhyatk/moondream2", description="Hugging Face model ID for the VLM."
    )
    vlm_revision: str = Field(
        default="2025-06-21", description="Model revision/commit for reproducibility."
    )
    vlm_timeout_s: float = Field(
        default=10.0, ge=1.0, description="Maximum seconds for a single VLM inference."
    )
    vlm_staleness_s: float = Field(
        default=5.0,
        ge=0.5,
        description="Discard detection triggers older than this many seconds.",
    )
    vlm_crop_padding: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Extra padding around bounding box as fraction of bbox size.",
    )

    # ── Inference Device ─────────────────────────────────────────
    device: Literal["auto", "mps", "cuda", "cpu"] = Field(
        default="auto",
        description="Inference device. 'auto' detects MPS → CUDA → CPU.",
    )

    # ── API Server ───────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host.")
    api_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI bind port.")

    # ── Database ─────────────────────────────────────────────────
    database_path: str = Field(
        default="data/eyeq.db", description="SQLite database file path."
    )

    # ── Webhooks ─────────────────────────────────────────────────
    webhook_url: str | None = Field(
        default=None, description="Optional webhook URL for alert dispatch."
    )
    webhook_timeout_s: float = Field(
        default=5.0, ge=0.5, description="Webhook HTTP request timeout."
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Application log level."
    )

    @property
    def inference_device(self) -> str:
        """Resolve ``'auto'`` to the actual best-available device string."""
        if self.device == "auto":
            return detect_inference_device()
        return self.device

    @property
    def video_source(self) -> str | int:
        """Parse ``default_source`` into an integer device index or a string URI/path."""
        try:
            return int(self.default_source)
        except ValueError:
            return self.default_source


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    Call this from anywhere in the codebase to access configuration.
    The instance is created once and reused for the process lifetime.
    """
    return Settings()
