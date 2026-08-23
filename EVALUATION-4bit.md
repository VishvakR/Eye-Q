# Evaluating `moondream/moondream-2b-2025-04-14-4bit` for Stage 2

**Verdict: adopt, and move `vlm_device` off `cpu` with it.** On Eye-Q's own
workload the 4-bit `2025-04-14` Moondream answers in **1,873 ms** against the
**4,757 ms** fp32 `2024-08-26` baseline it replaces — **2.5× faster**, on
**45% less disk**, burning **98× less CPU** per query.

Two caveats stated up front, because both change how the headline should be
read:

1. **Part of that win is the device, not the model.** Run the old fp32 model
   on MPS too and it lands at 3,327 ms, so the like-for-like model gain is
   **1.8×**, not 2.5×. Both numbers are below.
2. **This is a cross-revision comparison**, so the verdict disagreements in §4
   are not attributable to quantization.

The result is the opposite of what [EVALUATION.md](EVALUATION.md) found for
`Nagafi/moondream2-q4-onnx`, and for a reason worth keeping: that export lost
because 4-bit weights had to be unpacked on every GEMM of a compute-bound
prefill. This one **never runs a 4-bit GEMM at all** on this hardware. It is
4-bit *on disk*, dequantized once at load. The download shrinks; the arithmetic
does not change.

---

## 1. What this model actually is

| | |
|---|---|
| `model_type` | `moondream1`, remote code declaring `transformers` 4.44 |
| Format | Packed uint8 safetensors — two 4-bit values per byte, group size 128, plus per-group scale and zero-point |
| Lineage | Quantization-aware-trained release of Moondream 2B, `2025-04-14` |
| Pinned at | `a89c59223ef8b5bb7826780728eeec172727ca84` (repo `main` @ 2025-05-22) |

The model card is written for one deployment: `device_map={"": "cuda"}` with
`torchao` installed. Three things had to be handled before it ran here, and all
three are in [`vlm_backends.py`](app/pipelines/vlm_backends.py) rather than in
a fork of the weights.

### 1.1 The 4-bit path is CUDA-only

`QuantizedLinear.unpack()` dequantizes the packed weights to bf16 and then
immediately *re*-quantizes them:

```python
quantize_(self, int4_weight_only(group_size=128))
```

`int4_weight_only` defaults to `TensorCoreTiledLayout`, which dispatches to
`torch._weight_int4pack_mm` — a kernel that exists on CUDA and nowhere else.
There is no MPS path and the CPU path needs a different layout the remote code
never asks for.

Worse, the name is a moving target: **torchao 0.14 removed `int4_weight_only`**
in favour of `Int4WeightOnlyConfig`, and `layers.py` imports it at module scope.
Taking a hard dependency on torchao would mean pinning `<0.14` forever to
satisfy an import whose function is unreachable on most of our targets.

So `_ensure_torchao_importable()` installs a no-op stub in `sys.modules` when
torchao is genuinely absent, and `_patch_quantized_linear()` replaces `unpack`
with one that dequantizes to a chosen dtype and calls `quantize_` **only on
CUDA**. torchao stays an optional `[cuda]` extra. Unpacking is also forced at
load rather than left lazy in `forward()`, so the cost is load cost and not one
arbitrarily slow first verdict.

### 1.2 The pooling shim targets MPS by machine, not by model

`vision.py` swaps in a CPU round-trip for `adaptive_avg_pool2d` at *import*
time, routing around a missing MPS kernel for non-divisible sizes:

```python
if torch.backends.mps.is_available():
    def adaptive_avg_pool2d(input, output_size):
        return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to("mps")
```

The test asks whether the *machine* has MPS; the conclusion assumes the *model*
is on it. On an Apple box running the VLM on CPU — Eye-Q's default until this
change — half the vision projection is shipped to the GPU and the next
`torch.cat` dies on mismatched devices:

```
RuntimeError: Passed CPU tensor to MPS op
```

`_align_remote_code()` reinstates the same workaround returning to
`input.device`.

### 1.3 bf16 is assumed everywhere, and it is the wrong dtype on CPU

Only the packed layers go through `QuantizedLinear`. `MoondreamModel` defaults
to bf16, `prepare_crops` casts its pixels to bf16 outright, and the vision
tower's plain Linears load bf16 from the checkpoint — so choosing a dtype in
one place leaves a model whose halves disagree.

That matters because bf16 is the worst available choice on this hardware. One
prefill-shaped 2048×8192 GEMM, measured on the M2 CPU:

| dtype | per GEMM |
|---|---:|
| **fp32** | **24.5 ms** |
| fp16 | 110.9 ms |
| bf16 | 171.9 ms |

ARM has no Accelerate-backed reduced-precision GEMM, so upstream's bf16 is
**7× slower** on CPU. `_align_remote_code()` casts the whole model and patches
`prepare_crops`, and the default is fp32 on CPU, fp16 on MPS, bf16 on CUDA.
`EYEQ_VLM_COMPUTE_DTYPE` overrides it; `bench/benchmark_vlm.py --compute-dtype`
sweeps it.

### 1.4 The API moved on, in two ways that matter

`answer_question()` and `batch_answer()` survive as compatibility shims but
**ignore `max_new_tokens`**, and `batch_answer()` is now a plain sequential
loop with none of the padding and stacking the 2024 revision did. The backend
therefore calls `query()` directly and delegates `query_batch` to
`sequential_query_batch` rather than claiming a batching win that no longer
exists.

More importantly, `query()` **samples by default** — temperature 0.5, top_p 0.3.
A sampled binary verdict alerts on one frame and not the next for the same crop
and rule, which is indistinguishable from a flickering detection and impossible
to tune against. The backend passes `temperature=0`, which routes
`_prefill_prompt` to `argmax`.

### 1.5 It can be probed

The upside. `MoondreamModel._prefill_prompt` returns its logits, so unlike the
pinned 2024 revision this model satisfies `BinaryProbeBackend`: the verdict is
read off the distribution the prefill already computed instead of generated as
text and parsed back. §5 measures what that is and is not worth.

---

## 2. Method

Same fixtures, prompts, and harness as [EVALUATION.md](EVALUATION.md), so the
numbers are comparable to the ones already committed there. Stage 2 never sees
a full frame — it sees a padded, context-dimmed crop of one person produced by
`crop_detection_focused()` after YOLO fires — so fixtures come from running the
real Stage 1 over the real clip, and the prompts are the demo's own three rules
verbatim including the ` Answer with only Yes or No.` suffix.

- 3 crops × 3 prompts × 6 repeats = **54 measured queries per arm**
- One warmup query first, excluded
- Peak RSS sampled on a background thread at 20 ms
- Every arm capped at 16 new tokens and run greedy

Machine: Apple M2, 8 cores, 17.2 GB, macOS 26.5.1, Python 3.13.11, torch 2.13.

**Five arms, because two controls were needed.** Comparing the new model on MPS
against the old one on CPU would have confounded the model with the device, and
comparing on CPU alone would have measured the new model at its worst.

```bash
uv run python bench/benchmark_vlm.py --backend torch                      # committed baseline
uv run python bench/benchmark_vlm.py --backend torch --device mps         # device control
uv run python bench/benchmark_vlm.py --backend moondream4bit              # CPU
uv run python bench/benchmark_vlm.py --backend moondream4bit --device mps
uv run python bench/benchmark_vlm.py --backend moondream4bit --device mps --probe
```

---

## 3. Results

| | 4-bit @ MPS | 4-bit @ MPS, probe | 4-bit @ CPU | fp32 @ MPS *(control)* | **fp32 @ CPU *(baseline)*** |
|---|---:|---:|---:|---:|---:|
| **Median latency** | **1,873 ms** | 1,862 ms | 11,712 ms | 3,327 ms | **4,757 ms** |
| Mean | 1,884 ms | 1,904 ms | 12,252 ms | 3,627 ms | 4,864 ms |
| p90 | 1,899 ms | 2,071 ms | 13,941 ms | 4,259 ms | 5,981 ms |
| Min / max | 1,866 / 2,061 | 1,823 / 2,201 | 10,580 / 17,634 | 2,384 / 15,377 | 3,755 / 8,052 |
| **CPU per query** | **0.08 CPU-s** | 0.08 CPU-s | 17.28 CPU-s | 0.38 CPU-s | 7.86 CPU-s |
| CPU utilization | 0.04× | 0.04× | 1.41× | 0.11× | 1.62× |
| Steady peak RSS | 451 MB | 509 MB | 7,547 MB | 358 MB | 8,574 MB |
| On-disk weights | 2,032 MB | 2,032 MB | 2,032 MB | 3,736 MB | 3,736 MB |
| Load time | 7.0 s | 7.0 s | 10.1 s | 18.4 s | 20.5 s |

**Read the RSS row with care.** MPS allocations do not land in process RSS, so
451 MB does not mean the model fits in 451 MB — it means the weights moved
somewhere this probe cannot see. Only the two CPU columns are comparable to
each other, and there the 4-bit model holds **7,547 MB against 8,574 MB**: a
12% saving, not the 4× the name suggests, because off CUDA the weights are
dequantized to fp32 at load. The **disk** saving is real and unconditional:
2,032 MB against 3,736 MB, **1.8× smaller**.

### Where the time goes

This is the whole story, and it is a different story from last time:

| Stage | 4-bit @ MPS | Share |
|---|---:|---:|
| **Vision encoder** | **1,664 ms** | **88.8%** |
| Decoder prefill | 207 ms | 11.1% |
| Decode | ~0 ms | <1% |

The previous evaluation found Stage 2 to be **74% prefill** and concluded that
reducing image tokens was the lever. On this model and this device the
imbalance is sharper and has moved: Stage 2 is now **89% vision encoder**. The
language half costs 207 ms.

That is not a quantization effect. Measured per-stage on one crop at CPU fp32,
the 2025 model's vision encoder takes **7,313 ms against the 2024 model's
1,289 ms**, while its text prefill is *faster* — 2,313 ms against 3,156 ms for
the older model's whole answer. The new vision tower is simply much heavier,
and it is the half that benefits most from an accelerator.

Two things follow. First, **the CPU arm is not a viable deployment** and is
reported only as a control: 11,712 ms and 17.3 CPU-seconds per query is worse
than the model it replaces on both axes. Second, `EYEQ_VLM_MAX_CROP_EDGE=511`
was tuned against the 2024 encoder's single-pass threshold; this one always
emits a global view plus at least one tile — `overlap_crop_image` returns 2
crops for every fixture here — so that knob no longer does what its
documentation says. Untangling it is the obvious next piece of work and is not
attempted here.

### The CPU cost is the quiet headline

0.08 CPU-seconds per query against 7.86 is a **98× reduction**, and for an
appliance it may matter more than the latency. Stage 1 runs on those same cores
at ~16 ms/frame. Under the old default the VLM saturated 1.6 cores for five
seconds per verdict and competed directly with the detector; now it occupies
the GPU and leaves the CPU essentially free. On a fanless board this is also
the difference between sustainable and thermally throttled.

---

## 4. Verdicts agree 5 times out of 9

| Crop | Rule | fp32 `2024-08-26` | 4-bit `2025-04-14` | p(yes) |
|---|---|---|---|---:|
| crop_00 | bending | Yes | Yes | 0.761 |
| crop_00 | ppe | No | No | 0.108 |
| crop_00 | carrying | **Yes** | **No** | 0.254 |
| crop_01 | bending | **No** | **Yes** | 0.938 |
| crop_01 | ppe | No | No | 0.088 |
| crop_01 | carrying | No | No | 0.407 |
| crop_02 | bending | **No** | **Yes** | 0.957 |
| crop_02 | ppe | No | No | 0.097 |
| crop_02 | carrying | **Yes** | **No** | 0.239 |

Both models produce clean, parseable Yes/No on every one of the 54 queries —
`BinaryVerdictParser` never had to fall through to keyword scoring, and the
integration is sound in both directions.

But they disagree on four of nine, and **nine queries cannot tell you which one
is right.** The disagreement is confounded between quantization loss, the
revision gap, and the dtype and device change; separating them needs a labelled
set, which this evaluation does not have. Treat 5/9 as a flag for further work,
not as an accuracy measurement.

Two observations that are not measurements but are worth recording. The
disagreements cluster on `bending` and `carrying` and never on `ppe`, which is
the rule with the least ambiguous wording. And the 4-bit model's confidences on
the disputed `bending` calls are high — 0.938 and 0.957 — so it is not
hesitating; it simply reads those crops differently.

---

## 5. What the probe is and is not worth

The probe path is **9/9 identical** to generate-and-parse on the same crops, so
it is a correct shortcut. Its latency saving here is **11 ms — 0.6%**.

That is much less than the 9.1% the README records for the fp32 model, and the
reason is §3: decode was always a small share, and now that vision dominates it
is a small share of a small share. **Do not adopt probing for its latency on
this model.**

Adopt it for the other two properties, which are unchanged:

- **It cannot produce a parse failure.** The failure mode the README documents
  — the model answering `"Crouching"` to *"is this person bending over or
  crouching?"*, which parses as no violation and files no alert — is
  structurally impossible when the verdict is read off two token-id sets. It
  did not occur in this fixture set for either model, but it is not a matter of
  luck when probing.
- **Every verdict gets a calibrated confidence.** The `p(yes)` column in §4 is
  free, and it is the dial `EYEQ_VLM_PROBE_THRESHOLD` acts on. The text path
  never offered one, because a parsed "Yes" carries no strength.

---

## 6. Only one stage can have the accelerator

Moving the VLM onto MPS is not free, and the cost is not the one the old
`vlm_device=cpu` comment anticipated. **Stage 1 and an in-process VLM cannot
both drive MPS.** With both on it, the process segfaults:

```
EXC_BAD_ACCESS (SIGSEGV) ... faulting thread:
  at::native::_mps_convolution_impl(...)  -> +[NSDictionary dictionaryWithObjects:forKeys:count:]
concurrently, another thread:
  -[MPSGraph .cxx_destruct] -> mlir::MLIRContext::~MLIRContext()
```

One thread building an `MPSGraph` while another tears one down. PyTorch's MPS
backend is not safe to drive from two threads at once. It is a hard crash — no
traceback, no shutdown hook, and the ring buffer's shared memory is orphaned —
and it lands early, usually before the VLM has finished loading. Reproduced on
every run of a 75 s trial:

| `EYEQ_DEVICE` | `EYEQ_VLM_DEVICE` | Outcome over 75 s |
|---|---|---|
| `mps` | `mps` | **SIGSEGV**, model never finished loading |
| `mps` | `cpu` | survives, 9 verdicts |
| `cpu` | `mps` | survives, **29 verdicts** |

So the accelerator has to be given to one stage, and the choice is not close.
Per call on the M2:

| | MPS | CPU | cost of losing it |
|---|---:|---:|---:|
| Stage 1 (YOLOv8n @720p) | 16.9 ms | 26.8 ms | **9.9 ms** |
| Stage 2 (Moondream 2B 4-bit) | 1,873 ms | 11,712 ms | **9,839 ms** |

Stage 1 gives up ten milliseconds a frame; Stage 2 would give up ten seconds a
verdict — a thousandfold asymmetry. `Settings.inference_device` therefore
arbitrates the two `auto`s in one place instead of resolving them
independently, and Stage 1 takes CPU. Naming either device explicitly opts out;
naming both as the same accelerator is honoured with a warning, because an
operator who named both meant both.

This applies only to the in-process backends. `openai_compat` keeps its weights
in another process, so Stage 1 keeps the accelerator under that profile.

Verified end to end: 90 s on the demo clip with defaults, no crash, 34 verdicts,
inference steady at ~1,870 ms — matching §3.

---

## 7. Recommendation

**Adopt, with the device defaults arbitrated.**

1. `EYEQ_VLM_BACKEND` defaults to `moondream4bit`.
2. `EYEQ_VLM_DEVICE` moves from `cpu` to `auto`, and Stage 1 yields the
   accelerator per §6. Note the control run also **retires the old finding on
   its own terms**: the fp32 `2024-08-26` model now measures 3,327 ms on MPS
   against 4,757 ms on CPU, reversing the 6.1 s vs 5.6 s recorded when
   `vlm_device=cpu` was chosen. That default was correct when it was measured
   and has been overtaken by the toolchain, not by this change.
3. `EYEQ_VLM_BACKEND=moondream` remains the better choice on a CPU-only box,
   and is kept for exactly that.

What this change does **not** do:

- It does not make Stage 2 cheap on CPU. It makes it worse there.
- It does not deliver a 4× memory saving. Off CUDA it delivers 12% of RSS and
  1.8× of disk.
- It does not address the thing §3 identifies as dominant. 89% of the query is
  the vision encoder, and nothing here touches it.
- It does not let both stages use the accelerator. §6 gives it to Stage 2 and
  puts Stage 1 on CPU, at ~10 ms/frame.

The next piece of work follows from §3, not from quantization:

1. **Attack the vision encoder.** It is 89% of the query. `vlm_max_crop_edge`
   and the global-plus-tile behaviour of `overlap_crop_image` are where the
   time is, and that knob is currently documented against a threshold that no
   longer applies.
2. **Settle the accuracy question** with a labelled set. 5/9 agreement across a
   revision gap is a flag, and it is the one number here that could still
   reverse the recommendation.
3. **Re-measure `openai_compat` against llama.cpp on the target board.**
   `.env.pi5` still deploys that path, and nothing in this evaluation displaces
   it — a Pi 5 has no MPS, and this model on CPU is 11.7 s.

---

## 8. Reproducing

```bash
# Fixtures — real Stage 2 person crops from the demo clip
uv run python bench/make_fixtures.py --count 8

# The five arms in §3
uv run python bench/benchmark_vlm.py --backend torch --repeats 6
uv run python bench/benchmark_vlm.py --backend torch --device mps --repeats 6
uv run python bench/benchmark_vlm.py --backend moondream4bit --repeats 6
uv run python bench/benchmark_vlm.py --backend moondream4bit --device mps --repeats 6
uv run python bench/benchmark_vlm.py --backend moondream4bit --device mps --probe --repeats 6

# The dtype sweep behind §1.3
uv run python bench/benchmark_vlm.py --backend moondream4bit --compute-dtype bfloat16

# Driver tests (no weights needed)
uv run python -m pytest tests/test_vlm.py -m "not slow" -q

# Live tests, including the premise that pins §1.1's patch
uv run python -m pytest tests/test_vlm.py -m slow -q
```

Weights are not committed (`/models/` is gitignored); the backend fetches them
on first `load()` into `models/moondream-4bit`. Each run writes JSON to
`bench/results/`.

---

## 9. Limitations

- **n=9 distinct queries per arm, one clip, one machine.** Enough to settle a
  2.5× latency gap; not enough to say anything about accuracy.
- **Verdict disagreement is confounded** between quantization, the revision
  gap, and the dtype/device change. Untangling it needs labelled ground truth.
- **Measured on an M2, not on target hardware.** Every MPS number here is
  irrelevant to a Pi 5, which is why §7 leaves `.env.pi5` alone.
- **RSS is not comparable across devices** — see §3. The CPU-to-CPU comparison
  is the only honest memory read in the table.
- **CUDA was not measured at all.** The int4 path in §1.1, the bf16 default,
  and the ~2.4 GB figure the model card quotes are all untested here; this
  machine has no CUDA device. The code path exists and is unexercised.
- **The vision encoder was profiled on one crop**, not across the fixture set,
  when attributing the 2024/2025 difference in §3.
- **The MPS concurrency crash was not bisected against PyTorch versions.** It
  is reproducible here on torch 2.13 and the arbitration in §6 avoids it
  rather than fixing it. A serialising lock around both stages' accelerator
  work would be the alternative, and was not measured — Stage 1 would stall
  for the ~1.9 s of each verdict, against the flat ~10 ms/frame the chosen fix
  costs.
