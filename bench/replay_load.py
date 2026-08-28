"""Measure Stage 2 inference demand over a clip, without running the VLM.

The numbers that matter for load control are all decided before any inference
happens: how many track ids the tracker mints, how many triggers survive the
Stage 1 ladder, and how often the verdict memo hits. None of that needs a
model, so this runs in seconds on a laptop and can be put in a loop.

It exists because those numbers were previously unobtainable. The claim that
the tracker produces "roughly 2 ids per real person" appears in
ARCHITECTURE.md, eyeq.md and settings.py, and nothing in the repo produced it;
diagnosing why the VLM was saturated meant patching the worker by hand. The
measurement that matters most is the memo *miss* split — an appliance whose
misses are almost all `new_track` cannot be fixed by raising the TTL, however
tempting that knob looks.

Example:
    # Baseline, spatial reuse off:
    uv run python bench/replay_load.py --duration 300 --loop --crossid-reuse-s 0
    # With the fallback:
    uv run python bench/replay_load.py --duration 300 --loop
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipelines.detector import CooldownGate  # noqa: E402
from app.pipelines.tracker import GreedyIoUTracker  # noqa: E402
from app.schemas.events import BoundingBox  # noqa: E402


class _Memo:
    """The Stage 2 verdict memo, reimplemented over plain data.

    Deliberately a reimplementation rather than a `VLMWorker`: constructing the
    real worker drags in ZMQ sockets, a ring buffer and a backend, none of
    which affect the arithmetic. `tests/test_vlm.py` pins the real logic; this
    exists to count it in bulk. Keep the two in step — the branch order here
    mirrors `_rules_to_judge` exactly.
    """

    def __init__(self, ttl_s, rejudge_iou, min_rejudge_s, crossid_reuse_s, crossid_iou):
        self.ttl_s = ttl_s
        self.rejudge_iou = rejudge_iou
        self.min_rejudge_s = min_rejudge_s
        self.crossid_reuse_s = crossid_reuse_s
        self.crossid_iou = crossid_iou
        self.entries: dict[tuple[int, int], tuple[float, BoundingBox, str]] = {}
        self.hits_exact = 0
        self.hits_spatial = 0
        self.miss_new_track = 0
        self.miss_ttl = 0
        self.miss_moved = 0

    def wants_inference(self, track_id, rule_id, bbox, label, now) -> bool:
        entry = self.entries.get((track_id, rule_id))

        if entry is None:
            if self.crossid_reuse_s > 0:
                best_iou, best = self.crossid_iou, None
                for (_tid, rid), (judged_at, judged_bbox, lbl) in self.entries.items():
                    if rid != rule_id or lbl != label:
                        continue
                    if now - judged_at >= self.crossid_reuse_s:
                        continue
                    iou = bbox.iou(judged_bbox)
                    if iou >= best_iou:
                        best_iou, best = iou, (judged_at, bbox, lbl)
                if best is not None:
                    self.entries[(track_id, rule_id)] = best
                    self.hits_spatial += 1
                    return False
            self.miss_new_track += 1
            return True

        judged_at, judged_bbox, _lbl = entry
        age = now - judged_at
        if age >= self.ttl_s:
            self.miss_ttl += 1
            return True
        if bbox.iou(judged_bbox) < self.rejudge_iou and age >= self.min_rejudge_s:
            self.miss_moved += 1
            return True

        self.hits_exact += 1
        return False

    def remember(self, track_id, rule_id, bbox, label, now) -> None:
        self.entries[(track_id, rule_id)] = (now, bbox, label)


def replay(args) -> dict:
    from ultralytics import YOLO

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / src_fps

    tracker = GreedyIoUTracker(args.track_iou, args.track_max_age)
    cooldown = CooldownGate(args.cooldown_s)
    memo = _Memo(
        args.verdict_ttl_s, args.rejudge_iou, args.min_rejudge_s,
        args.crossid_reuse_s, args.crossid_iou,
    )

    now = 0.0
    frames = published = dets_total = filtered_size = filtered_cooldown = 0
    demands = 0
    ids_seen: set[int] = set()
    ids_per_pass: list[int] = []
    per_frame_counts: list[int] = []
    loops = 0
    pass_ids: set[int] = set()

    # CooldownGate reads time.monotonic(); drive it from the replay clock so a
    # 300s replay does not take 300s of wall time.
    import app.pipelines.detector as detector_mod
    real_monotonic = detector_mod.time.monotonic
    detector_mod.time.monotonic = lambda: now  # type: ignore[assignment]

    try:
        while now < args.duration:
            ok, frame = cap.read()
            if not ok:
                if not args.loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loops += 1
                ids_per_pass.append(len(pass_ids))
                pass_ids = set()
                continue

            if args.width:
                frame = cv2.resize(frame, (args.width, args.height))
            frames += 1
            h, w = frame.shape[:2]

            result = model.predict(
                frame, imgsz=640, conf=args.confidence, classes=[0],
                verbose=False, device=args.device,
            )[0]

            candidates = []
            if result.boxes is not None and len(result.boxes):
                data = result.boxes.data.cpu().numpy()
                dets_total += len(data)
                scale = np.array([w, h, w, h], dtype=np.float64)
                for row in data:
                    xy = np.clip(row[:4] / scale, 0.0, 1.0)
                    candidates.append((
                        "person",
                        BoundingBox(x1=float(xy[0]), y1=float(xy[1]),
                                    x2=float(xy[2]), y2=float(xy[3])),
                    ))

            per_frame_counts.append(len(candidates))
            track_ids = tracker.update(candidates)
            for tid in track_ids:
                ids_seen.add(tid)
                pass_ids.add(tid)

            for (label, bbox), tid in zip(candidates, track_ids):
                bw, bh = bbox.pixel_size(w, h)
                if max(bw, bh) < args.min_subject_px:
                    filtered_size += 1
                    continue
                if not cooldown.should_trigger(f"{label}#{tid}"):
                    filtered_cooldown += 1
                    continue
                published += 1
                for rule_id in range(args.rules):
                    if memo.wants_inference(tid, rule_id, bbox, label, now):
                        demands += 1
                        memo.remember(tid, rule_id, bbox, label, now)

            now += dt
    finally:
        detector_mod.time.monotonic = real_monotonic  # type: ignore[assignment]
        cap.release()

    ids_per_pass.append(len(pass_ids))
    looked_up = memo.hits_exact + memo.hits_spatial + demands
    return {
        "video": str(args.video),
        "duration_s": round(now, 1),
        "frames": frames,
        "clip_frames": n_frames,
        "loops": loops,
        "detections_total": dets_total,
        "max_simultaneous": max(per_frame_counts) if per_frame_counts else 0,
        "median_simultaneous": (
            statistics.median(per_frame_counts) if per_frame_counts else 0
        ),
        "distinct_track_ids": len(ids_seen),
        "new_ids_per_pass": ids_per_pass,
        "stage1": {
            "filtered_size": filtered_size,
            "filtered_cooldown": filtered_cooldown,
            "published": published,
        },
        "stage2": {
            "lookups": looked_up,
            "hits_exact": memo.hits_exact,
            "hits_spatial": memo.hits_spatial,
            "hit_rate": round((looked_up - demands) / looked_up, 3) if looked_up else 0.0,
            "miss_new_track": memo.miss_new_track,
            "miss_ttl": memo.miss_ttl,
            "miss_moved": memo.miss_moved,
            "inferences_demanded": demands,
            "inferences_per_s": round(demands / now, 3) if now else 0.0,
        },
        "duty_cycle_at": {
            f"{s}s": round(demands * s / now, 2) for s in (1.9, 11.7)
        },
        "settings": {
            "crossid_reuse_s": args.crossid_reuse_s,
            "crossid_iou": args.crossid_iou,
            "verdict_ttl_s": args.verdict_ttl_s,
            "cooldown_s": args.cooldown_s,
            "min_subject_px": args.min_subject_px,
            "rules": args.rules,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=Path("cctv_demo.mp4"))
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--loop", action="store_true", help="replay like demo_cctv.py does")
    ap.add_argument("--model", default="models/yolov8n.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--confidence", type=float, default=0.30)
    ap.add_argument("--cooldown-s", type=float, default=12.0)
    ap.add_argument("--min-subject-px", type=int, default=96)
    ap.add_argument("--rules", type=int, default=2, help="enabled rules per subject")
    ap.add_argument("--track-iou", type=float, default=0.2)
    ap.add_argument("--track-max-age", type=int, default=60)
    ap.add_argument("--verdict-ttl-s", type=float, default=300.0)
    ap.add_argument("--rejudge-iou", type=float, default=0.5)
    ap.add_argument("--min-rejudge-s", type=float, default=20.0)
    ap.add_argument("--crossid-reuse-s", type=float, default=45.0)
    ap.add_argument("--crossid-iou", type=float, default=0.7)
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args()

    report = replay(args)
    s2, s1 = report["stage2"], report["stage1"]

    print(f"\n{report['video']}  {report['duration_s']}s  "
          f"{report['frames']} frames  {report['loops']} loops")
    print(f"  subjects in frame     max {report['max_simultaneous']}  "
          f"median {report['median_simultaneous']}")
    print(f"  distinct track ids    {report['distinct_track_ids']}")
    print(f"  new ids per pass      {report['new_ids_per_pass']}")
    print(f"\n  Stage 1: size={s1['filtered_size']}  "
          f"cooldown={s1['filtered_cooldown']}  published={s1['published']}")
    print(f"  Stage 2: lookups={s2['lookups']}  hit rate {s2['hit_rate']:.1%}"
          f"  (exact {s2['hits_exact']}, spatial {s2['hits_spatial']})")
    print(f"    misses  new_track={s2['miss_new_track']}  "
          f"ttl={s2['miss_ttl']}  moved={s2['miss_moved']}")
    print(f"\n  INFERENCES DEMANDED   {s2['inferences_demanded']}"
          f"  ({s2['inferences_per_s']}/s)")
    for k, v in report["duty_cycle_at"].items():
        flag = "  SATURATED" if v >= 1.0 else ""
        print(f"    duty cycle @ {k}/query : {v}{flag}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
