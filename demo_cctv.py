"""Eye-Q CCTV demo — the full two-stage cascade against ``cctv_demo.mp4``.

``cctv_demo.mp4`` is 17 seconds of warehouse CCTV: staff in high-visibility
vests packing cartons around a bench, a pallet jack, stacked pallets. YOLOv8n
can only ever say *"person"* about them. The point of the demo is everything
Stage 2 can be asked on top of that label — vest compliance, manual handling —
in plain English, with no retraining.

Hard-hat questions are deliberately absent: at this camera distance a head is
about 20 px tall, and Moondream answers "No" to every phrasing of the question
regardless of what is actually on the worker's head. The clip cannot support
that rule, so the demo does not pretend otherwise.

The clip is played at its recorded 6 FPS and looped seamlessly, so the
appliance sees it exactly as it would see a live camera.

Usage:
    # Full cascade + dashboard on http://localhost:8000
    uv run python demo_cctv.py

    # Stage 1 only — YOLO boxes and the live feed, no 2 GB model load
    uv run python demo_cctv.py --stage1

    # Native OpenCV window instead of the dashboard (Stage 1 visual runner)
    uv run python demo_cctv.py --window

    # Start from an empty alert log
    uv run python demo_cctv.py --reset

The demo writes to ``data/demo.db`` and ``data/demo_snapshots/`` so it never
disturbs the alert history of a real deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("eyeq.demo")

VIDEO_PATH = "cctv_demo.mp4"
DEMO_DB = "data/demo.db"
DEMO_SNAPSHOTS = "data/demo_snapshots"
DEMO_CAMERA_ID = "default"


# ── Demo Rules ───────────────────────────────────────────────────
#
# These three were chosen by measurement, not guesswork: each candidate prompt
# was run against person crops sampled across the whole clip, and only the ones
# that answered reliably survived. Two things decided the shortlist.
#
# **Polarity.** The verdict parser reads an affirmative answer as a violation
# and the dashboard paints those red, so every prompt is phrased such that
# "Yes" means *something an operator should look at*. "Is this person wearing a
# vest?" would invert that and flag the compliant workers.
#
# **One predicate per question.** Asked "Is this person bending over or leaning
# across the table?", Moondream answers "Bending over" — it picks a branch
# instead of answering Yes or No, and the parser conservatively reads that
# non-answer as "no violation". Split compound conditions into separate rules.
#
# All three watch ``person``, the only COCO class this camera angle yields
# reliably; the cartons and pallet stacks come back as "suitcase" and "truck".
# That is the retrofit story in miniature — the detector is a motion filter,
# and the actual question is asked by the VLM.

DEMO_RULES: list[dict] = [
    {
        # Fires on roughly a third of sampled workers — this is the rule that
        # puts red on the dashboard.
        "name": "Manual handling — bending over",
        "prompt": "Is this person bending over?",
        "target_labels": ["person"],
        "confidence": 0.30,
        "cooldown_s": 12.0,
        "enabled": True,
    },
    {
        # The counterweight: every worker in this clip *is* wearing a vest, and
        # the VLM says so every time. A demo where nothing ever comes back
        # clear teaches you nothing about the false-alarm rate.
        "name": "PPE — missing hi-vis vest",
        "prompt": (
            "Is this person wearing plain dark clothing, "
            "with no high-visibility orange or yellow safety vest?"
        ),
        "target_labels": ["person"],
        "confidence": 0.30,
        "cooldown_s": 12.0,
        "enabled": True,
    },
    {
        # Off by default: a third rule means a third sequential inference per
        # trigger, and on the CPU backend that starts outrunning the cooldown.
        # Enable it from the dashboard to watch the queue absorb the load.
        "name": "Manual handling — carrying a box",
        "prompt": "Is this person carrying a box?",
        "target_labels": ["person"],
        "confidence": 0.30,
        "cooldown_s": 12.0,
        "enabled": False,
    },
]


# ── Environment ──────────────────────────────────────────────────

def _demo_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build the ``EYEQ_*`` overrides that tune the appliance for the clip."""
    env = {
        "EYEQ_DEFAULT_SOURCE": VIDEO_PATH,
        "EYEQ_DATABASE_PATH": DEMO_DB,
        "EYEQ_SNAPSHOT_DIR": DEMO_SNAPSHOTS,
        # The clip is 1080p; 720p is what the buffer and YOLO want anyway.
        "EYEQ_FRAME_WIDTH": "1280",
        "EYEQ_FRAME_HEIGHT": "720",
        # A ceiling, not a target: the file is paced at its recorded 6 FPS,
        # so nothing is skipped and 30 leaves headroom for a faster clip.
        "EYEQ_TARGET_FPS": "30",
        # 60 frames = 10s of history at 6 FPS, ~166 MB. The default 150 would
        # hold more than the whole clip for no benefit.
        "EYEQ_BUFFER_CAPACITY": "60",
        # Wide CCTV shot: the figures are small. 0.25 starts inventing boats
        # and chairs out of the pallet stacks; 0.35 loses real workers.
        "EYEQ_YOLO_CONFIDENCE": "0.30",
        # A trigger can wait longer than the 5s default before Stage 2
        # reaches it, and discarding it then would leave the demo silent.
        # Sized against the old fp32-on-CPU backend at ~4s per query; the
        # 4-bit default answers in ~1.9s on an accelerator, so this is now
        # slack rather than a fit — deliberately, since the demo also has to
        # survive a CPU-only machine where the same query takes ~11.7s.
        "EYEQ_VLM_STALENESS_S": "25.0",
        "EYEQ_DETECTION_COOLDOWN_S": "12.0",
    }
    if args.port is not None:
        env["EYEQ_API_PORT"] = str(args.port)
    if args.device is not None:
        env["EYEQ_DEVICE"] = args.device
    if args.stage1:
        env["EYEQ_ENABLE_VLM"] = "false"
    return env


def _apply_environment(env: dict[str, str]) -> None:
    """Export the overrides, leaving anything the user already set alone.

    Must run **before** the first ``get_settings()`` call — settings are
    cached with ``lru_cache``, so a later write would be ignored.
    """
    for key, value in env.items():
        os.environ.setdefault(key, value)


# ── Rule Seeding ─────────────────────────────────────────────────

def _clear_demo_history() -> None:
    """Delete the demo database and its evidence snapshots.

    The demo owns both outright — that is the whole reason it does not write
    to ``data/eyeq.db`` — so a reset removes the files rather than issuing a
    DELETE. WAL and shared-memory sidecars go too; leaving them behind would
    resurrect the rows they still hold.
    """
    removed = 0
    for sidecar in ("", "-wal", "-shm"):
        path = Path(f"{DEMO_DB}{sidecar}")
        if path.exists():
            path.unlink()
            removed += 1

    snapshots = Path(DEMO_SNAPSHOTS)
    if snapshots.is_dir():
        for jpg in snapshots.glob("*.jpg"):
            jpg.unlink()
            removed += 1

    logger.info(f"Reset: removed {removed} demo database/evidence file(s).")


async def _seed(reset_alerts: bool) -> None:
    """Put the demo rules into the demo database, replacing any prior run's.

    Seeding is destructive for *rules* on purpose: a demo that accumulates a
    duplicate rule set on every launch would multiply the VLM queue depth by
    the number of times it had been run.
    """
    from app.core.database import Database

    Path(DEMO_DB).parent.mkdir(parents=True, exist_ok=True)

    if reset_alerts:
        _clear_demo_history()

    Path(DEMO_SNAPSHOTS).mkdir(parents=True, exist_ok=True)

    db = await Database.create(DEMO_DB)
    try:
        await db.initialize_tables()
        await db.delete_all_rules()
        for rule in DEMO_RULES:
            await db.create_rule(camera_id=DEMO_CAMERA_ID, **rule)

        active = sum(1 for r in DEMO_RULES if r["enabled"])
        logger.info(
            f"Seeded {len(DEMO_RULES)} demo rule(s), {active} enabled: "
            + ", ".join(f"'{r['name']}'" for r in DEMO_RULES if r["enabled"])
        )
    finally:
        await db.close()


# ── Entry Points ─────────────────────────────────────────────────

def _run_window(args: argparse.Namespace) -> None:
    """Hand off to the Stage 1 visual runner with the clip preloaded."""
    from demo import run_demo

    logger.info("Opening the Stage 1 window — Q:quit  R:reset cooldowns  SPACE:pause")
    run_demo(
        argparse.Namespace(
            source=VIDEO_PATH,
            device=args.device or "mps",
            model="models/yolov8n.pt",
            labels="person",
            confidence=0.30,
            cooldown=12.0,
            roi=None,
            yolo_fps=6.0,
            width=1280,
            height=720,
            zmq_port=5558,
        )
    )


def _run_dashboard(args: argparse.Namespace) -> None:
    """Start the full appliance, seeded and pointed at the clip."""
    import uvicorn

    from app.config.settings import get_settings

    settings = get_settings()

    logger.info("=" * 64)
    logger.info("  Eye-Q CCTV Demo — warehouse floor, 17s clip on loop")
    logger.info("=" * 64)
    logger.info(f"  Video:      {VIDEO_PATH} (1920×1080 @ 6 FPS, seamless loop)")
    logger.info(f"  Stage 1:    YOLOv8n on {settings.inference_device}")
    if settings.enable_vlm:
        logger.info(
            f"  Stage 2:    {settings.vlm_backend} / {settings.vlm_model} "
            f"on {settings.vlm_inference_device}"
        )
        logger.info("              First run downloads ~2 GB — verdicts start after that.")
    else:
        logger.info("  Stage 2:    disabled (--stage1)")
    logger.info(f"  Dashboard:  http://localhost:{settings.api_port}")
    logger.info("=" * 64)

    uvicorn.run(
        "app.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Parse arguments, seed the demo database, and launch."""
    parser = argparse.ArgumentParser(
        description="Eye-Q demo driven by the bundled warehouse CCTV clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python demo_cctv.py              # Full cascade + dashboard
    uv run python demo_cctv.py --stage1     # No VLM load — boxes and feed only
    uv run python demo_cctv.py --window     # OpenCV window instead of browser
    uv run python demo_cctv.py --reset      # Empty the alert log first
        """,
    )
    parser.add_argument(
        "--stage1", action="store_true",
        help="Run Stage 1 only — skips the VLM model load entirely",
    )
    parser.add_argument(
        "--window", action="store_true",
        help="Show the native OpenCV Stage 1 window instead of the dashboard",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the demo alert history before starting",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Dashboard port (default: 8000)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "mps", "cuda", "cpu"],
        help="Stage 1 inference device (default: auto)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(VIDEO_PATH).is_file():
        logger.error(
            f"'{VIDEO_PATH}' not found in {Path.cwd()}. "
            "Put the clip in the project root and run this again."
        )
        sys.exit(1)

    if args.window:
        _run_window(args)
        return

    _apply_environment(_demo_environment(args))
    asyncio.run(_seed(reset_alerts=args.reset))
    _run_dashboard(args)


if __name__ == "__main__":
    main()
