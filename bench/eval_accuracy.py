"""Measure Stage 2's verdict accuracy against hand labels.

The gap this fills: nothing in this project has ever measured whether a verdict
was *right*. `bench/benchmark_vlm.py` measures speed, and its only verdict
comparison is model-against-model agreement, which cannot tell you which model
was correct. Every accuracy-shaped statement in the repo is either that
agreement figure or an anecdote.

What it reports, and why each part is there:

**Precision, recall and F1 per rule.** Accuracy alone is misleading when the
classes are unbalanced, and here they are — most workers are not bending over.
A model that answers "no" to everything would score well on accuracy and be
worthless.

**Wilson intervals on every rate.** With fifteen subjects a proportion carries
roughly +/-25 points, and a number without that attached invites conclusions it
cannot support. Wilson rather than normal-approximation because the sample is
small and the rates are near 0 or 1, which is exactly where the normal
approximation breaks.

**Independent subjects, not crops.** Several crops of one tracked person are
near-duplicates. Intervals are computed on the subject count so they do not
narrow just because the same person was sampled twice.

**A precision-recall curve over p(yes).** The probe path already returns a
calibrated confidence, so the operating point is a free parameter that has
never been chosen. This is the evidence for setting `vlm_probe_threshold`.

**A p(yes) histogram per rule.** The cheapest diagnostic here, and it needs no
labels at all: a rule whose confidence never leaves the middle of the range is
one the model cannot answer, whatever the threshold. A live run of the rule
"alert me if person have phone" put every verdict between 0.40 and 0.55, while
the demo's rules span 0.088 to 0.957.

Usage:
    uv run python bench/eval_accuracy.py --backend moondream4bit --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.pipelines.vlm_backends import (  # noqa: E402
    BinaryProbeBackend,
    ProbeUnsupportedError,
    build_backend,
)
from app.pipelines.vlm_judge import BinaryVerdictParser  # noqa: E402
from bench.benchmark_vlm import PROMPTS, VERDICT_SUFFIX  # noqa: E402
from bench.label_fixtures import LABELS_NAME, load_labels  # noqa: E402
from bench.make_fixtures import MANIFEST_NAME  # noqa: E402
from bench.probe import system_info  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for a proportion.

    Wilson rather than the normal approximation because this project's samples
    are small and its rates sit near 0 and 1 — precisely where the normal
    interval produces bounds outside [0, 1] and understates the uncertainty.

    Returns:
        ``(point, low, high)``; all zero when ``total`` is 0.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass
class Judgement:
    """One (crop, rule) pair: what the human said, what the model said."""

    crop: str
    rule: str
    track_id: int
    subject_px: int
    truth: bool
    verdict: bool
    p_yes: float | None
    answer: str
    latency_ms: float


@dataclass
class RuleMetrics:
    """Confusion matrix and rates for one rule, with intervals."""

    rule: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unsure_excluded: int = 0
    subjects: int = 0
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    f1: float = 0.0
    p_yes_histogram: dict[str, int] = field(default_factory=dict)
    p_yes_min: float | None = None
    p_yes_max: float | None = None

    def finalise(self) -> None:
        """Compute rates once the counts are in.

        Intervals use the subject count, not the crop count: two crops of one
        person are not two independent observations, and pretending otherwise
        would narrow every interval by roughly the square root of the
        duplication factor.
        """
        scale = self.subjects / max(1, self.tp + self.fp + self.tn + self.fn)
        self.precision = list(
            wilson(round(self.tp * scale), round((self.tp + self.fp) * scale))
        )
        self.recall = list(
            wilson(round(self.tp * scale), round((self.tp + self.fn) * scale))
        )
        self.accuracy = list(wilson(round((self.tp + self.tn) * scale), self.subjects))
        p, r = self.precision[0], self.recall[0]
        self.f1 = 2 * p * r / (p + r) if (p + r) else 0.0


def pr_curve(judgements: list[Judgement], steps: int = 21) -> list[dict]:
    """Precision and recall across candidate ``vlm_probe_threshold`` values.

    Only meaningful on the probe path; a judgement without ``p_yes`` cannot be
    re-thresholded after the fact, which is the entire reason the probe is
    worth having for tuning.
    """
    scored = [j for j in judgements if j.p_yes is not None]
    if not scored:
        return []

    rows = []
    for i in range(steps):
        threshold = i / (steps - 1)
        tp = sum(1 for j in scored if j.p_yes >= threshold and j.truth)
        fp = sum(1 for j in scored if j.p_yes >= threshold and not j.truth)
        fn = sum(1 for j in scored if j.p_yes < threshold and j.truth)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "threshold": round(threshold, 3),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0,
                    4,
                ),
                "alerts": tp + fp,
            }
        )
    return rows


def evaluate(backend, fixtures: Path, rules: list[str] | None) -> list[Judgement]:
    """Ask every labelled pair and record both answers."""
    manifest = json.loads((fixtures / MANIFEST_NAME).read_text())
    labels = load_labels(fixtures / LABELS_NAME)
    by_name = {f["name"]: f for f in manifest["fixtures"]}

    probeable = isinstance(backend, BinaryProbeBackend)
    prompts = [(k, p) for k, p in PROMPTS if not rules or k in rules]

    judgements: list[Judgement] = []
    for name, meta in by_name.items():
        path = fixtures / name
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        handle = backend.encode(image)

        for rule, prompt in prompts:
            label = labels.get(f"{name}|{rule}")
            if label is None or label == "unsure":
                continue

            question = prompt + VERDICT_SUFFIX
            started = time.perf_counter()
            p_yes = None
            if probeable:
                try:
                    result = backend.probe_binary(handle, question)
                    verdict, answer, p_yes = result.verdict, result.raw, result.p_yes
                except ProbeUnsupportedError:
                    probeable = False
            if p_yes is None:
                answer = backend.answer(handle, question)
                verdict = BinaryVerdictParser.parse(answer)
            latency_ms = (time.perf_counter() - started) * 1000.0

            judgements.append(
                Judgement(
                    crop=name,
                    rule=rule,
                    track_id=meta["track_id"],
                    subject_px=meta["subject_px"],
                    truth=label == "yes",
                    verdict=verdict,
                    p_yes=None if p_yes is None else round(p_yes, 4),
                    answer=answer,
                    latency_ms=round(latency_ms, 1),
                )
            )
            mark = "ok " if verdict == (label == "yes") else "MISS"
            conf = "" if p_yes is None else f"  p(yes)={p_yes:.3f}"
            print(
                f"    {name} [{rule:9s}] truth={label:3s} said="
                f"{'yes' if verdict else 'no ':3s} {mark}{conf}"
            )
    return judgements


def summarise(judgements: list[Judgement]) -> dict[str, RuleMetrics]:
    """Per-rule confusion matrices, plus an ``overall`` row."""
    metrics: dict[str, RuleMetrics] = {}
    for rule in {j.rule for j in judgements} | {"overall"}:
        rows = judgements if rule == "overall" else [j for j in judgements if j.rule == rule]
        if not rows:
            continue
        m = RuleMetrics(rule=rule, subjects=len({j.track_id for j in rows}))
        for j in rows:
            if j.truth and j.verdict:
                m.tp += 1
            elif not j.truth and j.verdict:
                m.fp += 1
            elif not j.truth and not j.verdict:
                m.tn += 1
            else:
                m.fn += 1

        scored = [j.p_yes for j in rows if j.p_yes is not None]
        if scored:
            m.p_yes_min, m.p_yes_max = round(min(scored), 4), round(max(scored), 4)
            for value in scored:
                bucket = f"{min(9, int(value * 10)) / 10:.1f}"
                m.p_yes_histogram[bucket] = m.p_yes_histogram.get(bucket, 0) + 1
        m.finalise()
        metrics[rule] = m
    return metrics


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=Path("bench/fixtures"))
    ap.add_argument("--backend", default=None, help="Overrides EYEQ_VLM_BACKEND.")
    ap.add_argument("--device", default=None, help="Overrides the resolved device.")
    ap.add_argument("--rule", action="append", help="Limit to these rule keys.")
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    labels_path = args.fixtures / LABELS_NAME
    if not labels_path.exists():
        raise SystemExit(
            f"No {labels_path}. Run bench/label_fixtures.py first — accuracy "
            "cannot be measured without ground truth, and this harness will "
            "not invent any."
        )

    settings = get_settings()
    if args.backend:
        settings = settings.model_copy(update={"vlm_backend": args.backend})
    if args.device:
        settings = settings.model_copy(update={"vlm_device": args.device})

    backend = build_backend(settings)
    print(f"\n{'=' * 70}\n{backend!r}\n{'=' * 70}")
    backend.load()

    judgements = evaluate(backend, args.fixtures, args.rule)
    backend.unload()

    if not judgements:
        raise SystemExit("No labelled pairs to evaluate.")

    metrics = summarise(judgements)
    curve = pr_curve(judgements)

    print(f"\n  {'rule':10s} {'prec':>18s} {'recall':>18s} {'F1':>6s}  TP/FP/TN/FN")
    for rule in sorted(metrics, key=lambda r: (r == "overall", r)):
        m = metrics[rule]
        print(
            f"  {m.rule:10s} "
            f"{m.precision[0]:.2f} [{m.precision[1]:.2f}-{m.precision[2]:.2f}] "
            f"{m.recall[0]:.2f} [{m.recall[1]:.2f}-{m.recall[2]:.2f}] "
            f"{m.f1:>6.2f}  {m.tp}/{m.fp}/{m.tn}/{m.fn}"
        )

    for rule, m in metrics.items():
        if rule == "overall" or m.p_yes_min is None:
            continue
        spread = m.p_yes_max - m.p_yes_min
        if spread < 0.3:
            print(
                f"\n  WARNING: rule '{rule}' p(yes) spans only "
                f"{m.p_yes_min:.2f}-{m.p_yes_max:.2f}. The model is not "
                "separating these cases, so no threshold will fix it — the "
                "prompt or the input resolution is the limit."
            )

    if curve:
        best = max(curve, key=lambda r: r["f1"])
        print(
            f"\n  Best F1 on the curve: threshold={best['threshold']} "
            f"precision={best['precision']:.2f} recall={best['recall']:.2f}"
        )
        print(
            f"  Currently shipping vlm_probe_threshold="
            f"{settings.vlm_probe_threshold}"
        )

    subjects = len({j.track_id for j in judgements})
    print(
        f"\n  {len(judgements)} judgement(s) over {subjects} independent "
        f"subject(s). Intervals are computed on subjects, not crops."
    )
    if subjects < 100:
        print(
            "  This sample cannot separate two close options. Add more varied "
            "footage before acting on a small difference."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.backend or settings.vlm_backend
    path = args.out / f"accuracy-{tag}-{platform.machine()}.json"
    path.write_text(
        json.dumps(
            {
                "backend": tag,
                "device": backend.device,
                "system": system_info(),
                "probe_threshold": settings.vlm_probe_threshold,
                "subjects": subjects,
                "metrics": {k: asdict(v) for k, v in metrics.items()},
                "pr_curve": curve,
                "judgements": [asdict(j) for j in judgements],
            },
            indent=2,
        )
    )
    print(f"  written to {path}")


if __name__ == "__main__":
    main()
