# Eye-Q

A two-stage edge AI video pipeline: a cheap YOLOv8n watchdog (~16 ms/frame) decides when to wake
Moondream 2B (~1,900 ms/verdict). Three orders of magnitude apart — everything here is machinery for
deciding what "almost nothing" means.

**[ARCHITECTURE.md](ARCHITECTURE.md) is the engineering reference.** `IMPLEMENTATION.md` and
`PLAN.md` are older; where they disagree with ARCHITECTURE.md or the source, they are wrong.

## Commands

```bash
uv run pytest -m "not slow"          # fast suite; `slow` marks tests that load real models
uv run ruff check app tests bench    # line length 100
uv run python main.py                # the appliance; --help for flags
uv run python demo_cctv.py           # seeded warehouse walkthrough, full pipeline
```

## Where things live

| | |
|---|---|
| Stage 1 (YOLO watchdog) | `app/pipelines/detector.py` |
| Stage 2 (VLM worker) | `app/pipelines/vlm_worker.py` |
| Settings | `app/config/settings.py` — pydantic-settings, env prefix `EYEQ_` |
| Wire contracts | `app/schemas/events.py` |
| Control plane | `app/api/server.py` (FastAPI lifespan owns every component) |

**ZMQ carries JSON metadata only — no pixels.** Stage 2 re-reads frames from the shared-memory ring
buffer by timestamp. This is deliberate; see ARCHITECTURE.md before changing it.

## Conventions

- **No `conftest.py`.** Fixtures are local per test file, by choice. Copy the local pattern rather
  than hoisting a shared one.
- **A new filter needs three things**: an `int` counter on the component, a field on its stats
  dataclass, and a key under `/api/status` — with a comment saying what a *rising* value means
  operationally. A drop counter nobody can interpret is not observability.
- **`Field(description=...)` is the documentation.** Settings descriptions are prose explaining the
  trade, not labels. Read them before changing a default; several encode measured results.
- **Never `time.sleep()` in a loop.** Always `self._shutdown_event.wait(timeout=...)`, so a stop
  lands immediately instead of at the end of the nap.
- **ZMQ sockets belong to the thread that created them.** Don't share them across threads.
- Logging is stdlib, f-strings, `logger = logging.getLogger(__name__)`. `debug` for per-item
  decisions, `info` for lifecycle and verdicts, `warning` for degradation.

## Traps

These cost real debugging time. The full list is ARCHITECTURE.md §12.

- **`get_settings()` is `lru_cache`d** and freezes config on first call. Tests must
  `get_settings.cache_clear()` — before *and* after, if they set env vars.
- **`frame_index` is a ring-buffer slot, not a frame number.** It is reused when the buffer wraps.
  Always validate the timestamp before trusting it.
- **Timestamps are `time.monotonic()`** — frame identity, not wall clock.
- **"Yes" means violation.** Inverting a prompt inverts the whole UI.
- **A high `triggers_stale` is correct**, not a fault. It means load shedding is working. Likewise
  `vlm.instances_too_small` reading zero — Stage 1 applies the same floor first.
- **If the VLM fires too often, read `memo_miss_new_track` before touching any knob.** Every gate is
  keyed on `track_id`, and it churns — measured 80 ids for ~6 people. When that counter dominates,
  raising `vlm_verdict_ttl_s` does nothing, because nothing lives long enough to expire. See
  ARCHITECTURE.md §"Identity churn is the dominant load driver", and measure with
  `uv run python bench/replay_load.py --duration 300 --loop`.
- **Do not pass `classes=` to `model.predict()`.** Both the live overlay and the co-occurrence gate
  need every class YOLO scored; narrowing there breaks them with no counter moving.
