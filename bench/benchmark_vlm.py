"""Benchmark Stage 2 VLM backends on real Eye-Q workloads.

Measures the three things that decide whether a model can be deployed on an
edge box:

    speed      cold load, then per-query latency split into vision / prefill
               / decode, over the demo's real person crops and real prompts
    memory     peak RSS during load and during steady-state inference —
               the number that decides whether it fits alongside Stage 1
    efficiency CPU-seconds burned per query, which on a fanless SBC is the
               difference between sustainable and thermally throttled

Every backend answers the same prompts about the same crops, and the answers
are recorded alongside the timings: a model that is fast because it produces
a degraded answer has not actually won.

Usage:
    uv run python bench/benchmark_vlm.py --backend onnx-q4
    uv run python bench/benchmark_vlm.py --backend torch --repeats 1
    uv run python bench/benchmark_vlm.py --backend onnx-q4 --threads 4
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipelines.vlm_judge import BinaryVerdictParser  # noqa: E402
from bench.probe import ResourceProbe, system_info  # noqa: E402

# The demo's real rules, verbatim from demo_cctv.py, plus the suffix the
# worker appends. Benchmarking on invented prompts would measure a prompt
# length this appliance never actually sends.
VERDICT_SUFFIX = " Answer with only Yes or No."

PROMPTS: list[tuple[str, str]] = [
    ("bending", "Is this person bending over?"),
    (
        "ppe",
        "Is this person wearing plain dark clothing, with no high-visibility "
        "orange or yellow safety vest?",
    ),
    ("carrying", "Is this person carrying a box?"),
]


@dataclass
class QueryRecord:
    """One prompt asked about one crop."""

    crop: str
    rule: str
    answer: str
    verdict: bool
    latency_ms: float
    vision_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    n_prompt_tokens: int = 0
    n_generated: int = 0
    decode_tok_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Everything one backend produced, ready to serialize."""

    backend: str
    label: str
    system: dict = field(default_factory=dict)
    load_wall_s: float = 0.0
    load_cpu_s: float = 0.0
    load_rss_peak_mb: float = 0.0
    load_rss_delta_mb: float = 0.0
    baseline_rss_mb: float = 0.0
    steady_rss_peak_mb: float = 0.0
    infer_cpu_s: float = 0.0
    infer_wall_s: float = 0.0
    disk_mb: float = 0.0
    records: list[QueryRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def latencies(self) -> list[float]:
        """Per-query wall latencies in milliseconds."""
        return [r.latency_ms for r in self.records]

    def summary(self) -> dict:
        """Aggregate the per-query records into headline numbers."""
        lat = self.latencies
        if not lat:
            return {}
        ordered = sorted(lat)
        return {
            "queries": len(lat),
            "median_ms": round(statistics.median(lat), 1),
            "mean_ms": round(statistics.fmean(lat), 1),
            "min_ms": round(ordered[0], 1),
            "max_ms": round(ordered[-1], 1),
            "p90_ms": round(ordered[int(0.9 * (len(ordered) - 1))], 1),
            "median_vision_ms": round(
                statistics.median([r.vision_ms for r in self.records]), 1
            ),
            "median_prefill_ms": round(
                statistics.median([r.prefill_ms for r in self.records]), 1
            ),
            "median_decode_ms": round(
                statistics.median([r.decode_ms for r in self.records]), 1
            ),
            "median_decode_tok_s": round(
                statistics.median([r.decode_tok_s for r in self.records]), 2
            ),
            "mean_generated_tokens": round(
                statistics.fmean([r.n_generated for r in self.records]), 1
            ),
            "cpu_s_per_query": round(self.infer_cpu_s / len(lat), 2),
            "cpu_utilization": round(
                self.infer_cpu_s / self.infer_wall_s if self.infer_wall_s else 0.0, 2
            ),
        }


def dir_size_mb(path: Path, patterns: tuple[str, ...]) -> float:
    """Total on-disk size of the files a backend actually loads."""
    total = 0
    for pattern in patterns:
        for f in path.rglob(pattern):
            total += f.stat().st_size
    return total / 1e6


def build(name: str, threads: int, max_tokens: int):
    """Construct a backend plus the disk footprint it will occupy.

    Args:
        name: Backend key from the CLI.
        threads: ORT intra-op threads (ONNX backends only).
        max_tokens: Generation cap, applied to every backend equally.

    Returns:
        Tuple of (backend, label, disk_mb).
    """
    if name in ("onnx-q4", "onnx-q4f16"):
        from app.pipelines.onnx_moondream import OnnxMoondreamBackend

        variant = "q4" if name == "onnx-q4" else "q4f16"
        root = Path("models/moondream2-q4-onnx")
        disk = (
            (root / "onnx/vision_encoder_q4.onnx").stat().st_size
            + (root / "onnx/embed_tokens_q4.onnx").stat().st_size
            + (root / f"onnx/decoder_model_merged_{variant}.onnx").stat().st_size
        ) / 1e6
        backend = OnnxMoondreamBackend(
            decoder_variant=variant,
            max_new_tokens=max_tokens,
            intra_op_threads=threads,
        )
        label = f"Nagafi/moondream2-q4-onnx ({variant}, ORT CPU, threads={threads or 'auto'})"
        return backend, label, disk

    if name == "torch":
        from app.pipelines.vlm_backends import MoondreamBackend

        backend = MoondreamBackend(device="cpu")
        disk = dir_size_mb(Path("models/moondream2"), ("*.safetensors", "*.bin"))
        return backend, "vikhyatk/moondream2 @2024-08-26 (HF Transformers, CPU)", disk

    raise SystemExit(f"Unknown backend: {name}")


def run(
    name: str, crops: list[Path], repeats: int, threads: int, max_tokens: int
) -> BenchmarkReport:
    """Load one backend, run every prompt over every crop, and measure.

    A warmup query runs before the measured loop. The first inference pays
    for ORT arena allocation and lazy kernel selection, and folding that
    one-off cost into the steady-state median would overstate the latency an
    operator actually sees.
    """
    import psutil

    proc = psutil.Process()
    report = BenchmarkReport(backend=name, label="", system=system_info())
    report.baseline_rss_mb = proc.memory_info().rss / 1e6

    backend, label, disk = build(name, threads, max_tokens)
    report.label = label
    report.disk_mb = round(disk, 1)

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  on-disk weights: {report.disk_mb:,.0f} MB")
    print(f"  process RSS before load: {report.baseline_rss_mb:,.0f} MB")

    try:
        with ResourceProbe("load") as probe:
            backend.load()
    except Exception as exc:  # noqa: BLE001 — a failed load is a result
        report.error = f"{type(exc).__name__}: {exc}"
        print(f"  LOAD FAILED: {report.error}")
        return report

    report.load_wall_s = round(probe.phase.wall_s, 2)
    report.load_cpu_s = round(probe.phase.cpu_s, 2)
    report.load_rss_peak_mb = round(probe.phase.rss_peak_mb, 1)
    report.load_rss_delta_mb = round(probe.phase.rss_delta_mb, 1)
    print(
        f"  loaded in {report.load_wall_s}s  "
        f"(RSS {report.baseline_rss_mb:,.0f} -> {probe.phase.rss_end_mb:,.0f} MB, "
        f"peak {report.load_rss_peak_mb:,.0f} MB)"
    )

    images = {p.name: Image.open(p).convert("RGB") for p in crops}

    # Warmup — excluded from the statistics on purpose.
    first = next(iter(images.values()))
    t0 = time.perf_counter()
    warm = backend.query(first, PROMPTS[0][1] + VERDICT_SUFFIX)
    print(f"  warmup: {(time.perf_counter() - t0) * 1000:,.0f} ms -> {warm[:60]!r}")

    trace_of = getattr(backend, "_runtime", None)

    with ResourceProbe("inference") as infer_probe:
        for _ in range(repeats):
            for crop_name, image in images.items():
                for rule, prompt in PROMPTS:
                    t0 = time.perf_counter()
                    answer = backend.query(image, prompt + VERDICT_SUFFIX)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    rec = QueryRecord(
                        crop=crop_name,
                        rule=rule,
                        answer=answer,
                        verdict=BinaryVerdictParser.parse(answer),
                        latency_ms=round(latency_ms, 1),
                    )
                    trace = getattr(trace_of, "last_trace", None)
                    if trace is not None:
                        rec.vision_ms = round(trace.vision_ms, 1)
                        rec.prefill_ms = round(trace.prefill_ms, 1)
                        rec.decode_ms = round(trace.decode_ms, 1)
                        rec.n_prompt_tokens = trace.n_prompt_tokens
                        rec.n_generated = trace.n_generated
                        rec.decode_tok_s = round(trace.decode_tokens_per_s, 2)

                    report.records.append(rec)
                    print(
                        f"    {crop_name} [{rule:9s}] {latency_ms:8,.0f} ms  "
                        f"{'RED ' if rec.verdict else 'green'}  {answer[:44]!r}"
                    )

    report.infer_wall_s = round(infer_probe.phase.wall_s, 2)
    report.infer_cpu_s = round(infer_probe.phase.cpu_s, 2)
    report.steady_rss_peak_mb = round(infer_probe.phase.rss_peak_mb, 1)

    backend.unload()
    return report


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        default="onnx-q4",
        choices=["onnx-q4", "onnx-q4f16", "torch"],
        help="Which Stage 2 backend to measure.",
    )
    ap.add_argument("--fixtures", type=Path, default=Path("bench/fixtures"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads.")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--limit-crops", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    crops = sorted(args.fixtures.glob("crop_*.png"))[: args.limit_crops]
    if not crops:
        raise SystemExit(
            f"No fixtures in {args.fixtures}. Run bench/make_fixtures.py first."
        )

    report = run(args.backend, crops, args.repeats, args.threads, args.max_tokens)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backend}-t{args.threads}" if args.threads else args.backend
    path = args.out / f"{tag}-{platform.machine()}.json"
    payload = asdict(report) | {"summary": report.summary()}
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n  SUMMARY  {report.label}")
    for k, v in report.summary().items():
        print(f"    {k:24s} {v}")
    print(f"  written to {path}")


if __name__ == "__main__":
    main()
