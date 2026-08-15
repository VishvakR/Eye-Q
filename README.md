# 🔬 Eye-Q

**Open-Vocabulary VLM Retrofit Appliance**

Intercept RTSP video streams from conventional IP cameras and define security alerts using plain English — no model training required.

## Architecture

Eye-Q uses a **Two-Stage Cascade** to run Vision-Language Models efficiently on edge hardware:

```
RTSP Camera → VideoCapture → Ring Buffer (Shared Memory)
                                  ↓
                          Stage 1: YOLOv8n (30 FPS, ~2ms/frame)
                                  ↓ (ZMQ trigger)
                          Stage 2: Moondream2 VLM (~800ms, wake-on-event)
                                  ↓
                          Binary Verdict → Webhook / Dashboard
```

- **Stage 1** (Always-On Watchdog): Lightweight YOLOv8n filters 99%+ of frames
- **Stage 2** (Wake-on-Event VLM): Moondream2 processes only triggered frames
- **Result**: VLM GPU utilization drops from 100% to <1%

## Quickstart

```bash
# Clone & install
git clone <repo-url> && cd Eye-Q
uv sync

# Run with webcam (default)
uv run eyeq

# Run with RTSP stream
EYEQ_DEFAULT_SOURCE="rtsp://192.168.1.100:554/stream1" uv run eyeq

# Run with video file
EYEQ_DEFAULT_SOURCE="/path/to/video.mp4" uv run eyeq
```

The dashboard will be available at `http://localhost:8000`.

## Configuration

All settings can be overridden via environment variables (prefix: `EYEQ_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `EYEQ_DEFAULT_SOURCE` | `0` | Camera source (RTSP URI, file path, or device index) |
| `EYEQ_BUFFER_CAPACITY` | `300` | Frames in ring buffer (~10s at 30 FPS) |
| `EYEQ_YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| `EYEQ_DEVICE` | `auto` | Inference device: `auto`, `mps`, `cuda`, `cpu` |
| `EYEQ_API_PORT` | `8000` | Dashboard & API port |
| `EYEQ_WEBHOOK_URL` | `None` | Webhook endpoint for alerts |

## Tech Stack

- **Runtime**: Python 3.13+ / `uv`
- **Detection**: YOLOv8n (Ultralytics) on Apple MPS
- **VLM**: Moondream2 (1.86B params) via Transformers
- **IPC**: ZeroMQ (PUB/SUB)
- **Buffer**: POSIX Shared Memory (zero-copy)
- **API**: FastAPI + SQLite (aiosqlite)
- **Frontend**: Tailwind CSS v4 + Vanilla JS

## License

MIT
