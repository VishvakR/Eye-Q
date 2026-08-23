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
                          Stage 2: Moondream 2B 4-bit VLM (~1.9s, wake-on-event)
                                  ↓
                          Binary Verdict → Webhook / Dashboard
```

- **Stage 1** (Always-On Watchdog): Lightweight YOLOv8n filters 99%+ of frames
- **Stage 2** (Wake-on-Event VLM): Moondream 2B processes only triggered frames
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
| `EYEQ_VLM_BACKEND` | `moondream4bit` | `moondream4bit`, `moondream`, `transformers`, or `openai_compat` |
| `EYEQ_VLM_MODEL` | `moondream/moondream-2b-2025-04-14-4bit` | Swap in Qwen2-VL-2B, SmolVLM, etc. |
| `EYEQ_VLM_DEVICE` | `auto` | MPS → CUDA → CPU. The 4-bit model is vision-bound and wants the accelerator |
| `EYEQ_API_PORT` | `8000` | Dashboard & API port |
| `EYEQ_WEBHOOK_URL` | `None` | Webhook endpoint for alerts |
| `EYEQ_TRACK_ENABLED` | `true` | Judge every subject in a frame, not one per label |
| `EYEQ_TRACK_IOU_THRESHOLD` | `0.2` | Minimum overlap to treat two boxes as the same subject |
| `EYEQ_VLM_VERDICT_TTL_S` | `300` | Seconds a subject's verdict is reused before re-judging |
| `EYEQ_VLM_MIN_REJUDGE_S` | `20` | Floor between re-judgements of one subject |
| `EYEQ_VLM_MAX_INSTANCES_PER_BATCH` | `8` | Subjects judged per drain cycle |
| `EYEQ_VLM_FOCUS_DIM` | `1.0` | Dimming outside the subject's box. Off by default; `0.4` restores it |
| `EYEQ_VLM_SQUARE_EDGE` | `378` | Letterbox crops into this square so a 1:3 person isn't stretched |
| `EYEQ_VLM_PROBE` | `auto` | Read the verdict from token probabilities instead of parsing text |
| `EYEQ_VLM_MAX_CROP_EDGE` | `511` | Longest side of the crop shown to the VLM. Keeps Moondream on its single-pass encoder |
| `EYEQ_VLM_MIN_SUBJECT_PX` | `96` | Skip a subject smaller than this on its longer side — too few pixels to answer about |
| `EYEQ_VLM_BATCH_SIZE` | `1` | Crops answered per forward pass. Raising it is measured, not assumed — see below |

### Why batching is off by default

`EYEQ_VLM_BATCH_SIZE` exists so the question can be *measured*, and it ships at
`1` because on the hardware this appliance targets the measurement does not
favour raising it.

Stage 2's shape is unusual. A crop enters the model as **729 image tokens** and
leaves as a **one-word answer** — so a query is roughly 99% prefill and 1%
decode. That ratio is what decides whether batching helps:

- **Decode** is memory-bound. It reads the entire weight matrix to produce one
  token per sequence, so a batch of four reads those weights once instead of
  four times. This is where batching normally pays, and Stage 2 barely does any.
- **Prefill** is compute-bound — hundreds of FLOPs per byte of weights fetched,
  well past the roofline knee on any of these boards. The weight read is already
  amortized across 729 tokens. Batching amortizes it further, over work that was
  never waiting on it.

So batching buys little here, and it is not free:

- **KV cache scales linearly with batch size** — roughly 295 MB per sequence for
  Moondream2 at fp32. On an 8 GB SBC already holding YOLO, the ring buffer and a
  3.5 GB model, `4` is not obviously affordable. The default `moondream4bit`
  backend cannot batch at all — its revision's `batch_answer()` is a sequential
  loop upstream — so there this only groups queries, it does not fuse them.
- **A timeout costs the whole group.** The worker abandons the remaining batch
  when an inference overruns, so one wedged dispatch takes N subjects with it
  instead of one.
- **Latency is per batch, not per subject.** No verdict in a group exists until
  the whole dispatch returns, so every subject waits for the slowest — which the
  dashboard reports honestly rather than amortizing away.

Measure before changing it, on the board you intend to deploy:

```bash
uv run python bench/make_fixtures.py --video cctv_demo.mp4 --count 8
uv run python bench/benchmark_vlm.py --backend torch --batch-sizes 1,2,4
```

The sweep runs identical (crop, prompt) pairs at every size and reports wall
time, peak RSS, and **verdict agreement against size 1**. Raise the setting only
when a speedup above 1.0 arrives with full agreement and an RSS delta the board
can absorb. Anything else is a slower, hungrier appliance that also answers
differently.

### Why verdicts are probed, not parsed

A Stage 2 query needs exactly **one bit** back. Generating that bit as text runs
the model autoregressively and then guesses the bit back out of the words —
and the guess has a failure mode measured on this repo's own fixtures.

Asked *"is this person bending over or crouching?"*, Moondream answers
`"Crouching"`. That is plainly yes. `BinaryVerdictParser` does not recognise the
word, falls through to its conservative default, and files **no alert**. It
happened on 1 of every 3 queries in the fixture set.

So where the backend allows it, the verdict is read straight off the
distribution the prefill already computed: compare the probability mass on the
`Yes` tokens against the `No` tokens at the first answer position. One forward
pass, no decode, no parsing, and a calibrated `confidence` on every result.

Be clear that the win is correctness, not speed. Measured over the real
fixtures on the default 4-bit model at MPS, removing the decode is worth
**11 ms — 0.6%**, because the query is dominated by something else entirely:

| | | |
|---|---:|---:|
| `encode_image` (vision) | 1,664 ms | **88.8%** |
| decoder prefill | 207 ms | 11.1% |
| decode beyond the first token | ~11 ms | 0.6% |

On the older fp32 model the same saving was ~9%. Either way it is not a step
change, and on this model it is nearly nothing. The reason to probe is the
`"Crouching"` case: it cannot produce that class of error, because it never
has to recognise the wording. The calibrated `confidence` — and the
`EYEQ_VLM_PROBE_THRESHOLD` dial that acts on it — comes free with it.

Stage 2 being 89% vision encoder is the standing performance problem, and
nothing in the cascade addresses it yet. See
[EVALUATION-4bit.md](EVALUATION-4bit.md) §3.

Not every backend can. `EYEQ_VLM_PROBE=auto` uses it where available and falls
back permanently on the first backend that refuses:

| Backend | Probing | Why |
|---|---|---|
| `transformers` | Yes | Logits are directly available |
| `openai_compat` | Server-dependent | **llama.cpp implements `logprobs`; Ollama does not** |
| `moondream4bit` (default) | Yes | The 2025-04-14 revision returns logits from its prefill |
| `moondream` | No | The pinned 2024-08-26 remote-code wrapper exposes no logits |

The fallback is latched after a single refusal — re-checking would buy a doomed
request before every generate — and the query that triggered it is still
answered the old way, so the transition costs nothing.

This is the reason to prefer llama.cpp over Ollama for the same model on the
same hardware: Ollama is one command to get running, but it cannot be probed,
so it pays a decode for every verdict. `.env.pi5` points at llama.cpp on
`:8080`; `.env.ollama` is the convenience profile and says so in its header.

### When the server is not there

`openai_compat` puts inference behind a socket, which adds a failure mode the
in-process backends do not have: the endpoint can simply be gone. A refused
connection is reported as `VLMUnavailableError` and is treated as a statement
about the *endpoint*, not the query — the worker abandons the rest of the batch
rather than paying a doomed round-trip for every remaining subject. Measured
against a dead port, a batch of eight costs one connection attempt and ~10 ms
instead of eight.

The outage logs once and clears itself when verdicts resume; `/api/status`
carries `backend_unavailable` and `batches_dropped_unavailable`, because a
backend that is down looks exactly like a quiet scene in every other counter.
A 4xx is deliberately *not* treated this way — that is a malformed request,
which is a bug to fix rather than a reason to shed traffic.

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
- **VLM**: Moondream 2B 4-bit (`2025-04-14`) by default; swappable for
  Moondream2 fp32, Qwen2-VL-2B, SmolVLM, or any model behind a local
  llama.cpp/Ollama server (`.env.pi5`, `.env.ollama`)
- **IPC**: ZeroMQ (PUB/SUB)
- **Buffer**: POSIX Shared Memory (zero-copy)
- **API**: FastAPI + SQLite (aiosqlite)
- **Frontend**: Tailwind CSS v4 + Vanilla JS

## License

MIT
