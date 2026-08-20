# 🔬 Eye-Q

**Open-Vocabulary VLM Retrofit Appliance**

Intercept RTSP video streams from conventional IP cameras and define security alerts using plain English — no model training required.

## Architecture

Eye-Q uses a **Two-Stage Cascade** to run Vision-Language Models efficiently on edge hardware:

```
RTSP Camera → VideoCapture → Ring Buffer (Shared Memory)
                                  ↓
                          Stage 1: YOLOv8n (~16ms/frame on M2 MPS)
                                  ↓ (ZMQ trigger)
                          Stage 2: Moondream2 VLM (~4s, wake-on-event)
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

The dashboard will be available at `http://localhost:8000`, with a live
MJPEG preview, the rule editor, and the alert feed.

Video files play at their recorded frame rate and loop seamlessly, so a
recorded clip drives the appliance exactly like a live camera.

## Try the Demo

`cctv_demo.mp4` is 17 seconds of warehouse CCTV — staff in hi-vis vests packing
cartons. `demo_cctv.py` points the appliance at it and seeds a rule set tuned
for the footage. The clip itself is not committed (`*.mp4` is gitignored, and
sample footage is large); drop it in the project root before running:

```bash
uv run python demo_cctv.py            # Full cascade + dashboard on :8000
uv run python demo_cctv.py --stage1   # Stage 1 only — skips the 3.5 GB VLM load
uv run python demo_cctv.py --window   # Native OpenCV window instead of the browser
uv run python demo_cctv.py --reset    # Start from an empty alert log
```

YOLOv8n can only say *"person"* about these workers. The demo's rules are
everything Stage 2 gets asked on top of that label, in plain English:

| Rule | Prompt | Expect |
|---|---|---|
| Manual handling — bending over | *Is this person bending over?* | 🔴 on ~⅓ of workers |
| PPE — missing hi-vis vest | *Is this person wearing plain dark clothing, with no high-visibility orange or yellow safety vest?* | 🟢 — everyone here is vested |
| Manual handling — carrying a box | *Is this person carrying a box?* | disabled; enable it from the dashboard |

The demo writes to `data/demo.db` and `data/demo_snapshots/`, so it never
touches the alert history of a real deployment.

### Writing your own rules

Two things decide whether a prompt works, both learned from tuning the above:

- **Phrase it so "Yes" means trouble.** An affirmative answer is recorded as a
  violation and shown in red. *"Is this person wearing a vest?"* flags every
  compliant worker.
- **Ask one thing per rule.** Given *"Is this person bending over or leaning
  across the table?"*, Moondream replies `"Bending over"` — it picks a branch
  instead of answering Yes or No, and that non-answer is read conservatively as
  "no violation". Split compound conditions into separate rules.

Match the rule to what the camera can actually resolve, too. In this clip a
worker's head is ~20 px tall, and hard-hat questions come back "No" regardless
of what is on their head — which is why the demo ships no such rule.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the architecture, component
reference, and how to swap the VLM model.

## Configuration

All settings can be overridden via environment variables (prefix: `EYEQ_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `EYEQ_DEFAULT_SOURCE` | `0` | Camera source (RTSP URI, file path, or device index) |
| `EYEQ_BUFFER_CAPACITY` | `150` | Frames in ring buffer (~5s at 30 FPS, ~405 MB at 720p) |
| `EYEQ_YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| `EYEQ_DEVICE` | `auto` | Stage 1 device: `auto`, `mps`, `cuda`, `cpu` |
| `EYEQ_VLM_BACKEND` | `moondream` | `moondream`, `transformers`, or `openai_compat` |
| `EYEQ_VLM_MODEL` | `vikhyatk/moondream2` | Swap in Qwen2-VL-2B, SmolVLM, etc. |
| `EYEQ_API_PORT` | `8000` | Dashboard & API port |
| `EYEQ_WEBHOOK_URL` | `None` | Webhook endpoint for alerts |
| `EYEQ_TRACK_ENABLED` | `true` | Judge every subject in a frame, not one per label |
| `EYEQ_TRACK_IOU_THRESHOLD` | `0.2` | Minimum overlap to treat two boxes as the same subject |
| `EYEQ_VLM_VERDICT_TTL_S` | `300` | Seconds a subject's verdict is reused before re-judging |
| `EYEQ_VLM_MIN_REJUDGE_S` | `20` | Floor between re-judgements of one subject |
| `EYEQ_VLM_MAX_INSTANCES_PER_BATCH` | `8` | Subjects judged per drain cycle |
| `EYEQ_VLM_FOCUS_DIM` | `0.4` | Dimming outside the subject's box, so neighbours can't flip a verdict |
| `EYEQ_VLM_MAX_CROP_EDGE` | `511` | Longest side of the crop shown to the VLM. Keeps Moondream on its single-pass encoder |

### Sizing the verdict TTL

Per-subject judging is affordable because a verdict is reused until the subject
moves or the TTL expires. The arithmetic that matters:

```
duty cycle = subjects_in_view × seconds_per_inference ÷ EYEQ_VLM_VERDICT_TTL_S
```

At ~6 s per query on CPU, five subjects and a 300 s TTL is a 10% duty cycle.
The same five at a 30 s TTL would saturate the VLM completely. If alerts start
lagging in a busy scene, raise the TTL before anything else.

Write rules about **one** subject — *"Is this person wearing a hard hat?"* —
because each crop shows a single person. Scene-level phrasing like *"Is anyone
missing a hard hat?"* no longer matches what the model is shown.

## Tech Stack

- **Runtime**: Python 3.13+ / `uv`
- **Detection**: YOLOv8n (Ultralytics) on Apple MPS
- **VLM**: Moondream2 (1.86B) by default; swappable for Qwen2-VL-2B, SmolVLM,
  or any model behind a local Ollama/llama.cpp server
- **IPC**: ZeroMQ (PUB/SUB)
- **Buffer**: POSIX Shared Memory (zero-copy)
- **API**: FastAPI + SQLite (aiosqlite)
- **Frontend**: Tailwind CSS v4 + Vanilla JS

## License

MIT
