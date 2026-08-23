"""Generate every figure in the Eye-Q technical documentation.

Each ``fig_*`` function returns a finished :class:`~roughen.Canvas`. Run this
module to write all seven SVGs into this directory.

    uv run python docs/diagrams/build.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from roughen import ACCENT, HAND, INK, MUTED, PAPER, WASH, Canvas  # noqa: E402

OUT = pathlib.Path(__file__).parent


# ── Figure 1 — System architecture ───────────────────────────────


def fig1_system() -> Canvas:
    """The appliance as one process: camera → ring → Stage 1 → gate → Stage 2."""
    c = Canvas(940, 600, seed=8121)

    # Standing note, in the grammar of the reference sketch.
    c.callout(
        150, 6, 372,
        "Frames keep arriving while the VLM thinks. The ring holds ~5 s of\n"
        "history, so Stage 2 can lag seconds behind live without the\n"
        "pipeline ever dropping a frame.",
        point_to=(300, 208),
    )

    # ── The process boundary ─────────────────────────────────────
    c.rect(26, 78, 650, 400, radius=14, fill=WASH)
    c.text(44, 100, "Eye-Q appliance — one Python process", size=13,
           anchor="start", fill=MUTED)

    # Source
    c.rect(42, 200, 92, 100, radius=8, fill=PAPER)
    c.camera_icon(64, 220, scale=1.15)
    c.text(88, 282, "Camera", size=13)
    c.text(88, 294, "RTSP · file · USB", size=10, fill=MUTED)

    # Ring buffer
    c.text(276, 198, "Ring buffer — POSIX shared memory", size=12.5, fill=MUTED)
    c.slot_strip(152, 210, 248, 48, ["F1", "F2", "F3", "F4", "⋯", "Fn"])
    c.text(276, 278, "150 slots · 414.7 MB @ 720p", size=10.5, fill=MUTED)
    c.text(276, 292, "one writer, lock-free readers", size=10.5, fill=MUTED)

    # The vertical chain
    c.node(428, 106, 228, 54, "YOLOv8n watchdog",
           sub="always on · ~16 ms/frame", size=14)
    c.node(428, 190, 228, 50, "Trigger gate",
           sub="scope → ROI → per-subject cooldown", size=13.5)
    c.node(428, 272, 228, 58, "VLM worker",
           sub="wake-on-event · Moondream 2 B 4-bit", size=14)
    c.node(428, 368, 228, 50, "AlertSink",
           sub="persist · broadcast · webhook", size=13.5)

    # Flow
    c.arrow(136, 248, 148, 248)
    c.elbow(390, 210, 428, 133, vertical_first=True)
    c.arrow(542, 160, 542, 188)
    c.arrow(542, 240, 542, 270, stroke=ACCENT, width=1.7)
    c.arrow(542, 330, 542, 366, stroke=ACCENT, width=1.7)
    c.text(534, 262, 'ZMQ :5557 · "detection"', size=10.5, anchor="end", fill=ACCENT)
    c.text(534, 352, 'ZMQ :5558 · "vlm_result"', size=10.5, anchor="end", fill=ACCENT)

    # Stage 2 fetches pixels back out of the ring, by timestamp.
    c.elbow(400, 258, 424, 301, vertical_first=True, arrow=True)
    c.text(396, 316, "read_by_timestamp()", size=10.5, anchor="end", fill=MUTED)

    # ── Sinks, outside the process ───────────────────────────────
    c.line(542, 418, 542, 494)
    c.line(412, 494, 696, 494)
    for x, w, label, sub in (
        (356, 112, "SQLite", "alerts · rules"),
        (490, 128, "Dashboard", "WebSocket + MJPEG"),
        (640, 112, "Webhook", "violations only"),
    ):
        c.arrow(x + w / 2, 494, x + w / 2, 516)
        c.node(x, 516, w, 48, label, sub=sub, size=13)

    # ── Margin annotations ───────────────────────────────────────
    for y, text, target in (
        (92, "Stage 1 — the cheap, always-on filter.\n"
             "Paced independently of capture, so it\n"
             "cannot starve Stage 2 for CPU.", (656, 133)),
        (178, "The sketch's single \"event engine\" is\n"
              "really two: what to WAKE lives here,\n"
              "what to ASK lives in Stage 2.", (656, 215)),
        (264, "Stage 2 — three orders of magnitude\n"
              "slower (~5.6 s/query on CPU), so it\n"
              "never sits in the frame path.", (656, 301)),
        (356, "Both verdicts are stored, not just\n"
              "violations — history is only\n"
              "trustworthy if clears are recorded.", (656, 393)),
    ):
        c.callout(694, y, 236, text, point_to=target, size=11.5)

    # ── Legend ───────────────────────────────────────────────────
    c.rect(26, 494, 306, 92, radius=6, fill=PAPER)
    for i, (term, gloss) in enumerate((
        ("Ring buffer", "remembers the last ~5 s"),
        ("YOLOv8n", "watches every frame, cheaply"),
        ("Trigger gate", "decides when reasoning is worth it"),
        ("VLM worker", "wakes, reads the buffered frame"),
    )):
        y = 514 + i * 17
        c.text(40, y, term, size=10.5, anchor="start")
        c.text(122, y, "= " + gloss, size=10.5, anchor="start", fill=MUTED)
    c.text(40, 578, "orange = the trigger / verdict path", size=10,
           anchor="start", fill=ACCENT)

    return c


# ── Figure 2 — End-to-end workflow and data flow ─────────────────


def fig2_workflow() -> Canvas:
    """One frame becoming one alert, and where the alert lands."""
    c = Canvas(940, 408, seed=4402)

    steps = [
        ("1 · Capture", "grab / retrieve split.\nFile sources paced at\ntheir recorded FPS."),
        ("2 · Ring write", "Copy into slot, stamp\nwith time.monotonic().\nWriter lock only."),
        ("3 · Detect", "Three passes: decode +\nconfidence, whole-frame\ntrack ids, then gate."),
        ("4 · Coalesce", "Drain the backlog, keep\nthe newest trigger per\n(label, track_id)."),
        ("5 · Judge", "Memo, then crop every\nsubject, VLM under a\nhard timeout, parse."),
        ("6 · Emit", "Persist, broadcast the\nstored row, POST the\nwebhook on violations."),
    ]

    box_w, gap, top = 140.0, 14.0, 64.0
    x0 = 15.0
    for i, (title, body) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        c.rect(x, top, box_w, 54, radius=8, fill=PAPER)
        c.text(x + box_w / 2, top + 33, title, size=14)
        c.text(x + box_w / 2, top + 76, body, size=10.5, fill=MUTED, leading=14)
        if i:
            c.arrow(x - gap - 2, top + 27, x - 3, top + 27, head=7)

    # Stage 2 goes back to the ring for pixels, which is why the crop has to
    # happen before the ring wraps. Routed above the band so it crosses no text.
    c.line(700, top, 700, 34)
    c.line(700, 34, 239, 34)
    c.arrow(239, 34, 239, top - 2)
    c.text(470, 26, "Stage 2 fetches the frame back out of the ring by timestamp",
           size=10.5, fill=MUTED)

    # ── Where the data lands ─────────────────────────────────────
    c.line(15, 226, 925, 226, stroke=MUTED, width=0.9, passes=1)
    c.text(15, 248, "Where it lands", size=13, anchor="start")

    lands = [
        (15, 288, "alerts row (SQLite)",
         "rule_id · camera_id · timestamp\nverdict 0/1 · raw_response\ninference_ms · bbox_json\nframe_snapshot · track_id · label"),
        (330, 288, "WebSocket /ws/events",
         "The stored DB row, not the\nVLMResult — so live cards and\npaginated history render through\none code path."),
        (645, 288, "Webhook POST",
         "Violations only, fire-and-forget.\nA dead endpoint can never build\na retry backlog that outlives the\nevent it describes."),
    ]
    for x, w, title, body in lands:
        c.rect(x, 260, w, 110, radius=7, fill=WASH)
        c.text(x + 14, 280, title, size=12.5, anchor="start")
        c.text(x + 14, 300, body, size=10.5, anchor="start", fill=MUTED, leading=13.5)

    c.text(15, 392, "The evidence JPEG is written once per subject and shared by every rule "
                    "asked about them, then served read-only from /snapshots.",
           size=10.5, anchor="start", fill=MUTED)
    return c


# ── Figure 3 — The Stage 2 decision path ─────────────────────────


def fig3_agent() -> Canvas:
    """What Stage 2 decides, per subject, before it spends an inference."""
    c = Canvas(940, 428, seed=7731)

    c.text(20, 22, "Per subject, per rule — decided before any pixels are touched",
           size=12.5, anchor="start", fill=MUTED)

    # ── Left: is this question worth asking at all? ──────────────
    c.node(20, 38, 400, 42, "Detection arrives:  person#7", size=14)
    c.arrow(220, 80, 220, 96, head=7)

    c.node(20, 96, 400, 44, "Match rules for this label",
           sub="label-specific first, then \"*\" wildcard", size=13.5)
    c.arrow(220, 140, 220, 156, head=7)

    c.node(20, 156, 400, 44, "Per-rule ROI test",
           sub="bbox centre inside the rule's polygon", size=13.5)
    c.arrow(220, 200, 220, 216, head=7)

    c.rect(20, 216, 400, 134, radius=8, fill=WASH)
    c.text(220, 238, "Verdict memo:  (track_id, rule_id)", size=13.5)
    for i, (cond, verdict) in enumerate((
        ("no entry for this subject", "judge"),
        ("age ≥ verdict_ttl_s (300 s)", "judge"),
        ("IoU < 0.5 and age ≥ 20 s", "judge"),
        ("otherwise", "reuse — no inference"),
    )):
        y = 262 + i * 22
        c.text(38, y, cond, size=11, anchor="start", fill=MUTED)
        c.text(402, y, "→ " + verdict, size=11, anchor="end",
               fill=ACCENT if verdict == "judge" else MUTED)

    c.text(220, 374, "This is the load control: reusing a verdict until the subject moves",
           size=10.5, fill=MUTED)
    c.text(220, 388, "or the TTL expires is what makes judging a whole crowd affordable.",
           size=10.5, fill=MUTED)
    c.text(220, 406, "duty = subjects × inference_s ÷ ttl", size=11, fill=ACCENT)

    # ── Right: how the surviving questions get answered ──────────
    c.arrow(424, 283, 498, 283, stroke=ACCENT, width=1.7)

    for y, title, body in (
        (38, "Priority sort", "never-judged subjects first, so a crowd\ndelays re-checks and never first looks"),
        (106, "Truncate the batch", "keep max_instances_per_batch; the rest\nis counted as instances_shed"),
        (174, "Focused crop", "pad for context, then dim everything\noutside the subject's own box"),
        (242, "Query the backend", "single-worker executor, hard timeout,\ncleanup() after every inference"),
    ):
        c.rect(500, y, 420, 60, radius=8, fill=PAPER)
        c.text(514, y + 21, title, size=13.5, anchor="start")
        c.text(514, y + 38, body, size=10.5, anchor="start", fill=MUTED, leading=12.5)
        if y > 38:
            c.arrow(710, y - 8, 710, y - 1, head=6)

    c.rect(500, 312, 420, 86, radius=8, fill=WASH)
    c.arrow(710, 302, 710, 310, head=6)
    c.text(514, 334, "BinaryVerdictParser", size=13.5, anchor="start")
    for i, strategy in enumerate((
        "1.  leading word is yes / no  →  taken as decisive",
        "2.  affirmative vs negative keyword count, negation-aware",
        "3.  ambiguous  →  False.  Alert fatigue is the worse failure.",
    )):
        c.text(514, 353 + i * 15, strategy, size=10.5, anchor="start", fill=MUTED)

    c.text(710, 418, "verdict written back to the memo, then published on :5558",
           size=10.5, fill=MUTED)
    return c


# ── Figure 4 — Innovation map ────────────────────────────────────


def fig4_innovation() -> Canvas:
    """The twelve strongest ideas, grouped by the kind of win they buy."""
    c = Canvas(940, 322, seed=1919)

    columns = [
        ("Latency", [
            ("Two-stage cascade", "a ~16 ms detector gates a\n~1.9 s reasoner", "detector.py · vlm_worker.py"),
            ("Crop before inference", "inference outlives buffer\nslots; crops do not", "vlm_worker.py:791"),
            ("Prompt-side budgeting", "24-token cap + forced Yes/No —\nevery token is a forward pass", "settings.py:209"),
        ]),
        ("Correctness", [
            ("Focused crops", "dim outside the subject box so a\nneighbour cannot flip the verdict", "vlm_worker.py:154"),
            ("Per-subject identity", "greedy IoU ids, so five workers\nyield five judgements", "tracker.py:51"),
            ("Conservative parsing", "three strategies, defaulting to\n\"no alert\" when ambiguous", "vlm_judge.py:73"),
        ]),
        ("Cost control", [
            ("Four-layer shedding", "coalesce → memoise → prioritise\n→ truncate", "vlm_worker.py:663"),
            ("Verdict memoisation", "keyed (track_id, rule_id) under\nTTL, motion and a re-judge floor", "vlm_worker.py:451"),
            ("Zero-copy ring", "lock-free stale-OK readers; a\nreader can never stall the writer", "buffer.py:56"),
        ]),
        ("Operability", [
            ("Prompt linting", "warns when a rule asks about\npeople but watches all 80 classes", "labels.py:123"),
            ("Display ≠ trigger", "a narrow rule never blinds the\noperator to the rest of the scene", "detector.py:610"),
            ("Three backends, one Protocol", "laptop to Raspberry Pi on a\nsingle environment variable", "vlm_backends.py:40"),
        ]),
    ]

    col_w, gap = 218.0, 12.0
    for ci, (heading, items) in enumerate(columns):
        x = 16 + ci * (col_w + gap)
        c.rect(x, 14, col_w, 296, radius=9, fill=WASH)
        c.text(x + col_w / 2, 38, heading, size=14.5)
        c.line(x + 16, 48, x + col_w - 16, 48, stroke=MUTED, width=0.9, passes=1)
        for ii, (title, body, ref) in enumerate(items):
            y = 60 + ii * 84
            c.rect(x + 12, y, col_w - 24, 74, radius=6, fill=PAPER)
            c.text(x + 22, y + 18, title, size=12, anchor="start")
            c.text(x + 22, y + 34, body, size=10, anchor="start", fill=MUTED, leading=12)
            c.text(x + 22, y + 65, ref, size=9, anchor="start", fill=ACCENT)
    return c

FIGURES = {
    "01-system-architecture": fig1_system,
    "02-workflow-dataflow": fig2_workflow,
    "03-agent-decision-path": fig3_agent,
    "04-innovation-map": fig4_innovation,
}


def main() -> None:
    """Write every figure to disk."""
    for name, builder in FIGURES.items():
        path = OUT / f"{name}.svg"
        path.write_text(builder().render())
        print(f"wrote {path.relative_to(pathlib.Path.cwd())}")


if __name__ == "__main__":
    main()
