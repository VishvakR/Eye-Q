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
    p_yes: float | None = None


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


def build(
    name: str,
    threads: int,
    max_tokens: int,
    compute_dtype: str = "",
    device: str = "cpu",
):
    """Construct a backend plus the disk footprint it will occupy.

    Args:
        name: Backend key from the CLI.
        threads: ORT intra-op threads (ONNX backends only).
        max_tokens: Generation cap, applied to every backend equally.
        compute_dtype: Weight dtype override (moondream4bit only).
        device: Torch device for the in-process backends.

    Returns:
        Tuple of (backend, label, disk_mb).
    """
    if name in ("onnx-q4", "onnx-q4f16"):
        try:
            from app.pipelines.onnx_moondream import OnnxMoondreamBackend
        except ImportError as exc:  # pragma: no cover - depends on the branch
            raise SystemExit(
                "The ONNX backend lives on eval/moondream2-q4-onnx and was "
                "deliberately not carried onto this branch; see EVALUATION.md "
                f"for why it is not deployed. ({exc})"
            ) from exc

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

        backend = MoondreamBackend(device=device)
        disk = dir_size_mb(Path("models/moondream2"), ("*.safetensors", "*.bin"))
        label = f"vikhyatk/moondream2 @2024-08-26 (HF Transformers, {device})"
        return backend, label, disk

    if name == "moondream4bit":
        from app.pipelines.vlm_backends import Moondream4BitBackend

        backend = Moondream4BitBackend(
            device=device,
            max_new_tokens=max_tokens,
            compute_dtype=compute_dtype or None,
        )
        # Only the safetensors: the repo also ships two GGUFs for llama.cpp
        # that this backend never opens, and counting them would inflate the
        # download by more than the quantization saves.
        disk = dir_size_mb(Path("models/moondream-4bit"), ("*.safetensors",))
        dtype = compute_dtype or "auto"
        return (
            backend,
            f"moondream/moondream-2b-2025-04-14-4bit (HF Transformers, "
            f"{device}, dequantized to {dtype})",
            disk,
        )

    if name == "openai_compat":
        # The profile .env.pi5 actually deploys: a quantized model behind
        # llama.cpp. Without this arm the edge path could not be measured at
        # all, which meant its latency was the only one taken on faith.
        from app.config.settings import get_settings
        from app.pipelines.vlm_backends import OpenAICompatBackend

        settings = get_settings()
        backend = OpenAICompatBackend(
            api_base=settings.vlm_api_base,
            model=settings.vlm_api_model,
            api_key=settings.vlm_api_key,
            max_tokens=max_tokens,
            timeout_s=settings.vlm_timeout_s,
            probe_threshold=settings.vlm_probe_threshold,
        )
        # Weights live in the server's process, not this one. Reporting 0 is
        # honest: it is not that the model is free, it is that this benchmark
        # cannot see it.
        return backend, f"{settings.vlm_api_model} @ {settings.vlm_api_base}", 0.0

    raise SystemExit(f"Unknown backend: {name}")


def run(
    name: str,
    crops: list[Path],
    repeats: int,
    threads: int,
    max_tokens: int,
    compute_dtype: str = "",
    use_probe: bool = False,
    device: str = "cpu",
) -> BenchmarkReport:
    """Load one backend, run every prompt over every crop, and measure.

    A warmup query runs before the measured loop. The first inference pays
    for ORT arena allocation and lazy kernel selection, and folding that
    one-off cost into the steady-state median would overstate the latency an
    operator actually sees.

    With ``use_probe=True`` the verdict is read off the prefill logits instead of
    generated as text. That is the same work minus the decode loop, so the two
    arms are directly comparable on the same crops and prompts — which is the
    only way to see what the decode was actually costing.
    """
    import psutil

    proc = psutil.Process()
    report = BenchmarkReport(backend=name, label="", system=system_info())
    report.baseline_rss_mb = proc.memory_info().rss / 1e6

    backend, label, disk = build(name, threads, max_tokens, compute_dtype, device)
    if use_probe:
        from app.pipelines.vlm_backends import BinaryProbeBackend

        if not isinstance(backend, BinaryProbeBackend):
            raise SystemExit(f"--probe: {name} cannot report token probabilities.")
        label = f"{label} [probe]"
    report.label = label
    report.disk_mb = round(disk, 1)

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  on-disk weights: {report.disk_mb:,.0f} MB")
    print(f"  process RSS before load: {report.baseline_rss_mb:,.0f} MB")

    try:
        with ResourceProbe("load") as load_probe:
            backend.load()
    except Exception as exc:  # noqa: BLE001 — a failed load is a result
        report.error = f"{type(exc).__name__}: {exc}"
        print(f"  LOAD FAILED: {report.error}")
        return report

    report.load_wall_s = round(load_probe.phase.wall_s, 2)
    report.load_cpu_s = round(load_probe.phase.cpu_s, 2)
    report.load_rss_peak_mb = round(load_probe.phase.rss_peak_mb, 1)
    report.load_rss_delta_mb = round(load_probe.phase.rss_delta_mb, 1)
    print(
        f"  loaded in {report.load_wall_s}s  "
        f"(RSS {report.baseline_rss_mb:,.0f} -> {load_probe.phase.rss_end_mb:,.0f} MB, "
        f"peak {report.load_rss_peak_mb:,.0f} MB)"
    )

    images = {p.name: Image.open(p).convert("RGB") for p in crops}

    def ask(image, prompt: str) -> tuple[str, bool, float | None]:
        """One measured verdict, by whichever path is under test."""
        if not use_probe:
            answer = backend.query(image, prompt)
            return answer, BinaryVerdictParser.parse(answer), None
        result = backend.probe_binary(backend.encode(image), prompt)
        return result.raw, result.verdict, result.p_yes

    # Warmup — excluded from the statistics on purpose.
    first = next(iter(images.values()))
    t0 = time.perf_counter()
    warm, _, _ = ask(first, PROMPTS[0][1] + VERDICT_SUFFIX)
    print(f"  warmup: {(time.perf_counter() - t0) * 1000:,.0f} ms -> {warm[:60]!r}")

    trace_of = getattr(backend, "_runtime", None)

    with ResourceProbe("inference") as infer_probe:
        for _ in range(repeats):
            for crop_name, image in images.items():
                for rule, prompt in PROMPTS:
                    t0 = time.perf_counter()
                    answer, verdict, p_yes = ask(image, prompt + VERDICT_SUFFIX)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    rec = QueryRecord(
                        crop=crop_name,
                        rule=rule,
                        answer=answer,
                        verdict=verdict,
                        latency_ms=round(latency_ms, 1),
                        p_yes=None if p_yes is None else round(p_yes, 4),
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
                    conf = "" if p_yes is None else f"  p(yes)={p_yes:.3f}"
                    print(
                        f"    {crop_name} [{rule:9s}] {latency_ms:8,.0f} ms  "
                        f"{'RED ' if rec.verdict else 'green'}  {answer[:44]!r}{conf}"
                    )

    report.infer_wall_s = round(infer_probe.phase.wall_s, 2)
    report.infer_cpu_s = round(infer_probe.phase.cpu_s, 2)
    report.steady_rss_peak_mb = round(infer_probe.phase.rss_peak_mb, 1)

    backend.unload()
    return report


@dataclass
class BatchResult:
    """One batch size, measured over the same work as every other."""

    batch_size: int
    wall_s: float
    per_query_ms: float
    rss_peak_mb: float
    rss_delta_vs_1_mb: float
    speedup_vs_1: float
    verdicts: list[bool] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    agreement_vs_1: str = ""


def run_batch_sweep(
    name: str, crops: list[Path], sizes: list[int], threads: int, max_tokens: int
) -> dict:
    """Answer the same question set at several batch sizes and compare.

    The comparison that decides whether ``vlm_batch_size`` should ever be
    raised above 1. Every size sees the identical (crop, prompt) pairs in the
    identical order, so the only variable is how many of them are handed to
    the model at once.

    Two things are recorded because two things can go wrong. Latency answers
    "did batching pay"; verdict agreement against size 1 answers "did it stay
    the same model" — batching left-pads mixed-length prompts, and padding is
    the one place a batched answer could legitimately drift.
    """
    import psutil

    proc = psutil.Process()
    backend, label, _disk = build(name, threads, max_tokens)

    print(f"\n{'=' * 70}\n{label}\n  batch sweep: {sizes}\n{'=' * 70}")

    with ResourceProbe("load"):
        backend.load()

    images = [Image.open(p).convert("RGB") for p in crops]
    # Every (crop, prompt) pair, in a fixed order shared by all sizes.
    pairs = [(im, prompt + VERDICT_SUFFIX) for im in images for _rule, prompt in PROMPTS]

    # Warmup, excluded — the first call pays for lazy allocation.
    backend.query(pairs[0][0], pairs[0][1])

    results: list[BatchResult] = []
    baseline: BatchResult | None = None

    for size in sizes:
        answers: list[str] = []
        with ResourceProbe(f"batch-{size}") as probe:
            for start in range(0, len(pairs), size):
                chunk = pairs[start : start + size]
                if size == 1:
                    answers.append(backend.query(chunk[0][0], chunk[0][1]))
                else:
                    answers.extend(
                        backend.query_batch(
                            [im for im, _q in chunk], [q for _im, q in chunk]
                        )
                    )
            backend.cleanup()

        verdicts = [BinaryVerdictParser.parse(a) for a in answers]
        res = BatchResult(
            batch_size=size,
            wall_s=round(probe.phase.wall_s, 2),
            per_query_ms=round(probe.phase.wall_s * 1000.0 / len(pairs), 1),
            rss_peak_mb=round(probe.phase.rss_peak_mb, 1),
            rss_delta_vs_1_mb=0.0,
            speedup_vs_1=1.0,
            verdicts=verdicts,
            answers=answers,
        )

        if baseline is None:
            baseline = res
            res.agreement_vs_1 = f"{len(verdicts)}/{len(verdicts)} (baseline)"
        else:
            agree = sum(a == b for a, b in zip(res.verdicts, baseline.verdicts))
            res.agreement_vs_1 = f"{agree}/{len(verdicts)}"
            res.rss_delta_vs_1_mb = round(res.rss_peak_mb - baseline.rss_peak_mb, 1)
            res.speedup_vs_1 = round(baseline.wall_s / res.wall_s, 2) if res.wall_s else 0.0

        results.append(res)
        print(
            f"  n={size:<2d}  {res.wall_s:7,.2f}s total  "
            f"{res.per_query_ms:8,.0f} ms/query  "
            f"peak RSS {res.rss_peak_mb:8,.0f} MB  "
            f"speedup {res.speedup_vs_1:4.2f}x  "
            f"verdicts {res.agreement_vs_1}"
        )

    backend.unload()

    print(
        "\n  Batching is worth enabling only if a speedup above 1 arrives "
        "with\n  full verdict agreement and an RSS delta the target board can "
        "absorb."
    )
    return {
        "backend": name,
        "label": label,
        "system": system_info(),
        "queries_per_size": len(pairs),
        "baseline_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        default="moondream4bit",
        choices=[
            "moondream4bit",
            "onnx-q4",
            "onnx-q4f16",
            "torch",
            "openai_compat",
        ],
        help="Which Stage 2 backend to measure.",
    )
    ap.add_argument("--fixtures", type=Path, default=Path("bench/fixtures"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads.")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--limit-crops", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    ap.add_argument(
        "--compute-dtype",
        default="",
        choices=["", "float32", "float16", "bfloat16"],
        help=(
            "Weight dtype for the moondream4bit backend; empty picks per "
            "device. The CPU choice is the one worth sweeping — upstream "
            "dequantizes to bf16 on its way to a CUDA kernel that is not "
            "reachable here."
        ),
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help=(
            "Torch device for the in-process backends. The default is 'cpu' "
            "because that is what the committed baseline was measured on; the "
            "moondream4bit arm is vision-bound and wants 'mps' on Apple."
        ),
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Read the verdict off the prefill logits instead of generating "
            "text. Measures what the decode loop was costing."
        ),
    )
    ap.add_argument(
        "--batch-sizes",
        default="",
        help=(
            "Comma-separated batch sizes to sweep, e.g. '1,2,4'. The first is "
            "the baseline every other size is compared against. Decides "
            "whether vlm_batch_size should ever be raised above 1."
        ),
    )
    args = ap.parse_args()

    crops = sorted(args.fixtures.glob("crop_*.png"))[: args.limit_crops]
    if not crops:
        raise SystemExit(
            f"No fixtures in {args.fixtures}. Run bench/make_fixtures.py first."
        )

    if args.batch_sizes:
        sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
        payload = run_batch_sweep(
            args.backend, crops, sizes, args.threads, args.max_tokens
        )
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"batch-sweep-{args.backend}-{platform.machine()}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"  written to {path}")
        return

    report = run(
        args.backend,
        crops,
        args.repeats,
        args.threads,
        args.max_tokens,
        args.compute_dtype,
        args.probe,
        args.device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backend}-t{args.threads}" if args.threads else args.backend
    if args.device != "cpu":
        tag = f"{tag}-{args.device}"
    if args.compute_dtype:
        tag = f"{tag}-{args.compute_dtype}"
    if args.probe:
        tag = f"{tag}-probe"
    path = args.out / f"{tag}-{platform.machine()}.json"
    payload = asdict(report) | {"summary": report.summary()}
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n  SUMMARY  {report.label}")
    for k, v in report.summary().items():
        print(f"    {k:24s} {v}")
    print(f"  written to {path}")


if __name__ == "__main__":
    main()
