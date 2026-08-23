# Eye-Q — Interview Guide

An open-vocabulary video alerting appliance. This document explains it from
first principles, in plain language, with every number traced to a measurement.

**How to use this.** Section 1 is what you say unprompted. Sections 2–6 are what
you say when they dig. Section 7 is practice. Section 8 is what you memorise.
Section 9 is where you will get pushed, and what to say when you are.

**One rule for this whole document.** Every number carries its model and its
device. The same model is 1,873 ms on one device and 11,712 ms on another. A
bare number is not an answer.

---

## 1. The two-minute pitch

> Security cameras are mostly useless. Not because the video is bad — because
> nobody watches it. The classic fix is to train a detector for each thing you
> care about, which means a labelling project per rule, so in practice you get
> three rules and never a fourth.
>
> Eye-Q lets you type the rule in English. "Is this person bending over?" A
> vision-language model looks at the frame and answers yes or no. No training,
> no labelling, any rule you can phrase.
>
> The obvious problem is that a VLM is about two seconds per query and video is
> thirty frames a second. You are off by three orders of magnitude. So Eye-Q is
> a cascade: a small YOLO detector watches every frame at about seventeen
> milliseconds, and it does exactly one job — decide whether anything happened
> that is worth asking a question about. Almost always the answer is no. When it
> is yes, it wakes the VLM for one crop of one person.
>
> That is the whole idea. Everything else is the consequences of it, and the
> consequences are where the engineering is. Inference takes seconds and the
> frame buffer only holds seconds, so if you fetch the frame right before you
> infer, the second person in a crowded frame finds their frame already gone —
> invisible with one person in shot, total with five. So you crop everybody up
> front and let the buffer expire. Five people times three rules is fifteen
> questions per trigger, so verdicts are cached per person until they move or
> time out, and the cache TTL is derived from a duty-cycle formula rather than
> picked. And when I actually profiled it, eighty-nine percent of the query was
> the vision encoder, not the language model — which meant most of my instincts
> about where to optimise were wrong.
>
> The honest summary: it runs on my laptop at about 1.9 seconds a verdict using
> four percent of one CPU core. The edge deployment — a Pi with llama.cpp — is
> designed and abstracted for, and I have not run it. I know exactly which
> measurement would settle it.

That last paragraph is deliberate. Say it. It is worth more than an overclaim.

---

## 2. First principles: why this is hard

Start from the numbers, because the whole architecture falls out of one gap.

| Stage | What it does | Cost per call |
|---|---|---|
| YOLOv8n object detector | "Is there a person?" | **~17 ms** |
| Moondream 2B vision-language model | "Is this person bending over?" | **~1,900 ms** |

That is roughly **110×**. A 30 FPS camera gives you a new frame every 33 ms. The
detector fits. The VLM does not fit, is not close to fitting, and no amount of
tuning makes a 2-billion-parameter model fit into 33 ms on a laptop.

So you cannot run the VLM on the stream. You can only run it on **events**. That
is the cascade, and it is not a clever trick — it is the only shape that works.

Two consequences follow immediately, and both drove real design:

**The cheap stage decides what the expensive stage sees.** If YOLO wakes the VLM
too often, the VLM falls behind and every verdict describes a frame that is
already gone. Most of the code in Stage 1 is about *not* waking Stage 2.

**The expensive stage runs longer than the data lives.** The frame buffer holds
about five seconds. One VLM query takes about two. If you queue five subjects,
the last one's frame is gone before you reach it. This is a correctness problem,
not a performance problem, and it is easy to miss because it is invisible when
only one person is in shot.

---

## 3. Architecture and workflow

### The pipeline

```
RTSP / file / webcam
        │
        ▼
  VideoCapture thread ──► Ring buffer (POSIX shared memory)
                                 │
                                 ▼
                       Stage 1: YOLOv8n watchdog thread
                       every frame · ~17 ms · "did anything happen?"
                                 │  ZMQ pub/sub :5557  (topic: detection)
                                 ▼
                       Stage 2: VLM worker thread
                       on events only · ~1.9 s · "is the rule violated?"
                                 │  ZMQ pub/sub :5558  (topic: vlm_result)
                                 ▼
                       Alert sink (asyncio)
                       SQLite · WebSocket · webhook
```

One process, four threads that do very unequal amounts of work. Two ZMQ
endpoints rather than one, because ZMQ sockets are not thread-safe and this way
each publisher owns its own socket in its own thread.

### The ring buffer

Three shared-memory segments: pixels (uint8), timestamps (float64), and a
two-slot meta block holding the write head and a monotonic frame count.

The interesting decision is **the writer locks; readers do not.** A reader can
in principle catch a frame mid-overwrite. That is fine: a torn frame produces a
garbage YOLO detection that the confidence filter drops. Paying for reader locks
to prevent an error that the next stage already filters would be paying twice.

Small detail worth knowing because it shows up in interviews about shared state:
the counters live *inside* the shared memory rather than in `multiprocessing.Value`,
because each `Value` carries a POSIX semaphore that `unlink()` cannot release —
it leaked on every shutdown.

### Stage 1 — the watchdog

Four things, in this order, and the order matters.

**1. Decode all boxes in one transfer.** Every `.item()` on an accelerator tensor
is a synchronisation point. Reading eleven boxes field-by-field measured **5.4 ms**;
pulling the whole `(N, 6)` array across once measured **0.15 ms**. That is a third
of Stage 1's entire frame budget, spent fetching sixty-six numbers.

**2. Assign identity before gating.** YOLO tells you *what* is in the frame, never
*who*. Two frames of the same five workers give you ten unrelated boxes. A greedy
IoU tracker — match each box to the previous frame's box with the highest overlap
— assigns stable integer ids.

Why not ByteTrack? Ultralytics' ByteTrack calls `check_requirements()` on the `lap`
package at *import* time, which attempts a pip install at runtime. On an offline
appliance that is a hard failure. Greedy IoU is a few thousand float operations,
deterministic, and unit-testable without loading a model.

The IoU threshold is 0.2, which looks far too loose until you see why: at 0.45 the
clip produced **4.2 ids per real person**; at 0.2 it produces **2.0**. A subject
that fails to match gets a fresh id, looks like someone nobody has judged yet, and
costs a full VLM inference. A strict threshold does not show up as tracking errors.
It shows up as wasted inference.

**3. Gate on label, region, and cooldown — but only for waking the VLM.** A person
standing in view generates thirty detections per second. The cooldown is keyed
`person#7`, not `person`, which is what makes a frame of five workers produce five
triggers instead of one.

The gate deliberately does **not** decide what the dashboard draws. A rule watching
`person` must not blind the operator to the forklift beside them. YOLO scores every
class anyway, so showing them all is free.

**4. Crop.** Covered in §5.6 — it is a correctness argument, not a preprocessing step.

### Stage 2 — the judge

Four layers of load control, each removing work the next one would otherwise do.

**Coalesce.** One blocking receive, then drain up to 512 queued triggers. Keep only
the newest per **(label, track_id)**. Keying on label alone was a real bug: a frame
of five workers collapsed to one judgement, and the other four were discarded as
"superseded" when they were different people.

**Shed stale.** Group by frame timestamp; drop any group older than five seconds
*before* fetching a single frame. A rising stale count is the system working, not
failing — it means Stage 1 is outrunning Stage 2 and Stage 2 is declining to answer
questions about the past.

**Memoise verdicts.** Skip a `(track_id, rule_id)` if the subject is tracked, the
verdict is younger than the TTL, and the box has not moved (IoU against the box we
judged). This is the actual load control, and its sizing is a formula rather than a
guess:

```
duty_cycle = subjects × inference_seconds ÷ verdict_ttl_seconds
```

Five subjects at 8 s with a 300 s TTL is **0.13** — comfortable. Drop the TTL to
30 s and it is **1.33**, and the worker is permanently behind. Giving an operator a
formula instead of a magic number is the part worth saying out loud.

**Prioritise and truncate.** Never-judged subjects sort ahead of re-checks, and the
batch re-polls between subjects so someone who walks in mid-batch can overtake the
queued re-checks. Overflow past the cap is counted, not silently dropped.

---

## 4. The tech stack

Split into two tables on purpose. The split is the credible part.

### Running, and measured

| Layer | Choice | Note |
|---|---|---|
| Stage 1 | YOLOv8n via Ultralytics | 16.9 ms MPS / 26.8 ms CPU @720p |
| Tracking | Hand-written greedy IoU | No runtime dependency install |
| Stage 2 | Moondream 2B, `2025-04-14`, 4-bit | HF Transformers, `trust_remote_code` |
| Accelerator | PyTorch MPS (Apple Silicon) | CUDA path written, never executed |
| Transport | ZMQ PUB/SUB, two endpoints | ~30 µs localhost, `NOBLOCK` sends |
| Frames | POSIX shared memory ring | Writer-locked, lock-free readers |
| Control plane | FastAPI + uvicorn, SQLite, WebSocket | Single process |
| Verdicts | Logit probing, with text parsing as fallback | See §5.4 |

### Designed and abstracted for — not yet run

Say this plainly. It is not a weakness if you know why it matters.

| Layer | Status |
|---|---|
| llama.cpp / `llama-server` with 4-bit GGUF | **Never compiled, never benchmarked.** No `openai_compat` result has ever existed in this repo. |
| Raspberry Pi 5 deployment | **Never run.** The profile says so: "nobody has yet run Eye-Q on a Pi." |
| Batch-size sweep | Harness written and complete; **no result committed.** |

**What *is* real** is the abstraction. Stage 2 sits behind a `VLMBackend` Protocol
with four implementations, and the HTTP one is properly built: a single choke point
for every request, so "the server is gone" is distinguishable from "this query
failed"; transport errors and 5xx raise a distinct exception that drops the whole
batch, while 4xx deliberately does not, because a malformed request is your bug to
fix and shedding traffic on it hides the defect; and a connect timeout of 2 s split
out from the 60 s read timeout, because a read must allow a full prefill but a
connect that has not landed in two seconds never will.

It is unit-tested against a mocked HTTP client with a hand-written llama.cpp
response body. That is protocol conformance. It is not evidence that a server ever
replied, and you should not describe it as if it were.

---

## 5. Six engineering decisions

Each in the same shape: what I expected, what I measured, what I did.

### 5.1 Batching is off by default

**Expected:** batch four crops, get most of a 4× speedup. Standard practice.

**Measured — actually, reasoned, and this is the honest version:** Stage 2's shape
is unusual. A crop enters as **729 image tokens** and leaves as a **one-word
answer**. So a query is ~99% *prefill* — the one-shot pass over the whole prompt —
and ~1% *decode*, the token-by-token generation after it.

That ratio decides everything:

- **Decode is memory-bound.** It reads the entire weight matrix to produce one
  token per sequence. Batch four and you read those weights once instead of four
  times. This is where batching pays — and Stage 2 barely does any decode.
- **Prefill is compute-bound.** The weight read is already spread across 729
  tokens. Batching spreads it further, over work that was never waiting on it.

**Did:** shipped `batch_size = 1`, and shipped the sweep harness so the question can
be answered rather than argued.

**Be precise about two things.** First, this is about the *language model's*
prefill, not the vision encoder's — they are different passes and an interviewer
may ask which. Second, the supporting figures (KV cache ~295 MB per sequence, ~280
FLOP per byte) are **back-of-envelope, not measured**, and no sweep result is
committed. The reasoning is sound; the artifact is missing. Say so.

### 5.2 Quantisation made it slower — the first time

**Expected:** 4-bit weights are a quarter the size, so they should be faster.

**Measured:** a 4-bit ONNX export ran **10,363 ms** against the fp32 baseline
measured in that same run, **3,900 ms** — 2.7× *slower*, and 6× more CPU. (That
baseline figure is from a 9-query paired run; the 4,757 ms elsewhere in this
document is the later 54-query committed baseline. Quote whichever run you are
describing, not a mix.) Memory dropped 2.5×, which is the
one axis Stage 2 was not constrained on.

**Why:** profiling put 74% of the time in decoder prefill. Prefill is compute-bound,
and 4-bit weights must be unpacked before they can be multiplied — so every GEMM
pays an unpacking tax. Quantisation optimises the 1% and taxes the 99%. A model
that emits a single token is close to the worst possible case for weight
quantisation — the exact opposite of a chatbot, where long generations make it pay.

**Did:** rejected it, and wrote down why, because the reason generalises.

**The second time is a different case, and this is the nuance to lead with.** The
4-bit model now in use is 4-bit *on disk only*. Its runtime kernel is CUDA-only, so
on Apple Silicon the weights are unpacked once at load into fp16 and every matmul
after that is ordinary fp16. The download drops from 3,736 MB to 2,032 MB. There is
**no 4-bit arithmetic anywhere**, and none of the speedup comes from quantisation —
it comes from a newer model and from the GPU. If you claim a quantisation win here,
the follow-up question ends badly.

Related, and measurable: on ARM CPU the dtype you dequantise *to* matters enormously.
One prefill-shaped GEMM measured **fp32 24.5 ms, fp16 110.9 ms, bf16 171.9 ms**. The
upstream code hardcodes bf16 — sensible on its way to a CUDA kernel, 7× slower here.
ARM has no Accelerate-backed reduced-precision GEMM path.

### 5.3 The bottleneck moved

This is the most important item, because it is about method rather than a result.

**Expected:** after the earlier finding of "74% prefill", keep attacking prefill.

**Measured:** after changing model and device, I re-profiled instead of assuming:

| Stage | Time | Share |
|---|---|---|
| **Vision encoder** | **1,664 ms** | **88.8%** |
| Decoder prefill | 207 ms | 11.1% |
| Decode | ~0 ms | <1% |

The bottleneck had moved from the language model to the vision tower. Per-stage on
CPU, the new model's vision encoder is **7,313 ms** against the old model's
**1,289 ms**, while its text prefill is actually *faster*.

**Did:** two things. Documented that the standing performance problem is now the
vision encoder and that nothing in the current design addresses it. And flagged a
setting that had gone stale: the crop is capped at 511 px because the *old* model
resized anything under 512 px once, and at 512+ switched to tiling — 3 to 5 forward
passes instead of 1, measured at **1.42 s versus 3.16 s** for the same one-word
answer. The new model always emits a global view plus a tile, so that cap no longer
does what its documentation says.

**The transferable point:** a profile is valid for one model on one device. Change
either and it expires. I have the same system profiled two ways, and they disagree,
and I know why.

### 5.4 Reading the answer instead of parsing it

**Expected:** removing the decode loop would be a meaningful latency win.

**Measured:** **11 ms. About 0.6%.** Prefill genuinely halved, 207 ms → 104 ms, but
the total is within run-to-run noise because the vision encoder dominates. Do not
claim a latency win here.

**What it actually bought — and this is the better story anyway.** The old path
generates text and then guesses the answer back out of the words with a string
parser. That parser has a failure mode I can demonstrate in three lines:

```python
>>> BinaryVerdictParser.parse("Crouching")
False
>>> BinaryVerdictParser.parse("Bending over")
False
```

Asked *"is this person bending over or crouching?"*, the model answers `"Crouching"`.
The model is **right**. But `"crouching"` is in none of the parser's four word sets,
so it falls through to a conservative default and files **no alert**. A correct
answer became a silent missed alert. Note this is not a hallucination — inventing
that framing is easy to catch and weaker than the truth.

Probing removes the class of error entirely. Instead of generating text, take the
logits at the first answer position, sum the probability mass on the yes tokens
against the no tokens, and normalise over the pair. There is no wording to
recognise, so there is nothing to fail to recognise. It also yields a calibrated
confidence for free — `p(yes) = 0.94` — which is the dial the alert threshold acts
on, and which a parsed "Yes" never carried.

**Did:** made it a *separate optional* Protocol rather than part of the base
interface, because one backend genuinely cannot do it. The worker checks with
`isinstance`, and latches the capability off permanently on the first refusal —
re-checking would buy a doomed request before every single query. The query that
triggered the fallback is still answered the old way, so the transition costs
nothing.

Renormalising over the yes/no pair matters and is worth explaining: the absolute
mass on "Yes" depends on how much the model wanted to say anything else at all
("The", "I", "Sure"), which is a property of its verbosity, not of its answer. The
*ratio* carries the verdict.

### 5.5 Two threads, one GPU, one segfault

**Expected:** moving the VLM to the GPU is a pure win. It measured 2.5× faster in
isolation.

**Measured:** the app started, loaded the model, published detections — and died
about twenty seconds in. Clean exit, no traceback, no Python exception. Just gone.

I had twice told my collaborator it was environmental. It was not. The macOS
crash reports settled it — one per run, all of them since the device default
changed:

```
EXC_BAD_ACCESS (SIGSEGV)
thread 7:  at::native::_mps_convolution_impl(...) → +[NSDictionary dictionaryWithObjects:...]
thread 8:  -[MPSGraph .cxx_destruct] → mlir::MLIRContext::~MLIRContext()
```

One thread building an `MPSGraph` while another tears one down. PyTorch's MPS
backend is not safe to drive from two threads. Before the change only YOLO used
the GPU; after it, Stage 1 and Stage 2 both did.

I isolated it with a three-cell matrix rather than guessing, 75 s per cell:

| Stage 1 | Stage 2 | Result |
|---|---|---|
| MPS | MPS | **SIGSEGV**, model never finished loading |
| MPS | CPU | survives, 9 verdicts |
| CPU | MPS | survives, **29 verdicts** |

**Did:** the GPU has to go to one stage, and the measurement makes it a
non-decision:

| | GPU | CPU | cost of losing it |
|---|---:|---:|---:|
| Stage 1 (YOLOv8n) | 16.9 ms | 26.8 ms | **9.9 ms** |
| Stage 2 (Moondream 2B) | 1,873 ms | 11,712 ms | **9,839 ms** |

A thousandfold asymmetry. So the two `auto` device settings are now arbitrated in
one place instead of resolving independently, and Stage 2 wins. Naming a device
explicitly opts out; naming both as the same accelerator is honoured with a warning,
because someone who named both meant both.

**Be honest about what this is.** It *avoids* the race, it does not fix it. The
alternative is a lock serialising both stages' GPU work, which I did not measure —
it would stall Stage 1 for the full ~1.9 s of every verdict, against a flat ~10 ms
per frame for the arbitration. That is the trade I chose and the reasoning behind it.

### 5.6 Crop before you infer

**Expected:** fetch the frame, crop, infer. The obvious order.

**Measured:** it is a correctness bug, and it hides. Inference takes seconds. The
ring buffer holds seconds. Fetch the frame immediately before each inference and
the *second* subject in a group finds their frame already overwritten and their
trigger already past the staleness threshold — dropped without ever being looked at.

**Invisible in a single-subject scene. Total in a crowded one.** Which is exactly
backwards from what you want, because a crowded scene is when alerting matters.

**Did:** read the frame once per timestamp group, take *every* crop up front, free
the frame, and only then start inference. Cropping costs about a millisecond and a
couple of megabytes per subject, and it makes the buffer irrelevant by the time the
slow part begins.

Two related crop decisions, both measured:

- **Letterbox to a fixed square** rather than resizing. Standing people crop to
  roughly 1:3; a square resize stretches a person to three times their width, which
  is not a shape anything in training looked like. The pad colour is grey 114, not
  black — black bars read as unlit scene and the model describes them. The side
  effect matters more than the fix: every crop now has identical dimensions, so the
  image-token count is constant and latency stops varying with how tall the subject
  happened to be.
- **Skip subjects below 96 px** on the longer side. At this camera distance a head
  is about 20 px, and the model answers "No" to every phrasing of a hard-hat
  question regardless of what is on the worker's head. That is the most expensive
  inference in the system spent on the subject least able to support an answer.
  This trades recall for precision and is resolution-dependent — at 1080p a floor
  of 96 skips 6.5% of detections, at 960×540 it skips 38.7%.

---

## 6. Trade-offs

| Decision | Buys | Costs | Reverse it when |
|---|---|---|---|
| Two-stage cascade | VLM utilisation near zero | Verdict lags the event by ~2 s | Never — it is the premise |
| Batch size 1 | Bounded memory, per-subject latency | Leaves any batching win unclaimed | A committed sweep shows speedup >1 with full verdict agreement |
| Verdict memo | Cost scales with *new* subjects, not crowd size | A subject re-judged only on movement or TTL | Rules where state flips faster than the TTL |
| Conservative parser default | No alert fatigue | Silent missed alerts | Probing removes the need entirely |
| Greedy IoU tracker | No runtime dependency, deterministic, free | Id swaps on occlusion | Never — a swap costs one redundant inference, not a missed alert |
| GPU to Stage 2 | 9.8 s/verdict | 9.9 ms/frame on Stage 1 | Stage 1 needs full frame rate more than Stage 2 needs speed |
| 4-bit model | 1.8× smaller download, newer weights | Nothing on GPU; 2.4× slower on CPU | CPU-only box — use the fp32 backend there |
| In-process VLM | No external service, no IPC | Weights in the app's memory; competes for the GPU | Edge boxes — that is what the HTTP backend is for |
| Probing | Removes a class of silent failure; free confidence | Not supported by every backend | Never; it degrades gracefully |

---

## 7. Four questions, with answers

### Q1. "Walk me through where the latency actually goes."

> About 1.9 seconds per verdict on an M2 with the 4-bit Moondream on the GPU. The
> split is 1,664 ms in the vision encoder, 207 ms in the decoder prefill, and
> essentially zero decode — so **89% is vision**.
>
> What I think is more interesting is that this is the second profile I took, and
> it disagrees with the first. Earlier, on a different model on CPU, 74% was
> decoder prefill and vision was 25%. The bottleneck moved when I changed the model
> and the device. If I had carried the first profile forward I would have spent my
> time on prefill, which is now eleven percent of the query.

**Follow-up: "So what would you do next?"**

> Attack the vision encoder, and specifically the number of image tiles. The prompt
> is 729 image tokens for a one-word answer. There is a crop cap in the config set
> to 511 px, which existed because the old model resized anything under 512 once and
> tiled above it — 3 to 5 forward passes instead of 1, measured at 3.16 s versus
> 1.42 s. The new model always emits a global view plus a tile, so that knob no
> longer does what it says. That is the first thing I would fix, and I would expect
> it to matter more than anything I could do to the language model.

**Follow-up: "How did you measure the split?"**

> The backend records vision time and prefill time separately per query. For the
> decode share I ran the same crops twice — once generating text, once stopping at
> the prefill logits — and took the difference, rather than instrumenting inside
> the generation loop, because that loop is upstream code and a second copy of it
> would drift.

### Q2. "Why not just batch the inferences?"

> Because of the shape of the workload. A crop goes in as 729 image tokens and comes
> out as one word. That is about 99% prefill and 1% decode.
>
> Batching pays when you are memory-bound — decode reads the whole weight matrix to
> produce one token per sequence, so batching four reads those weights once instead
> of four times. But Stage 2 barely decodes. Prefill is compute-bound and its weight
> read is already spread over 729 tokens, so batching spreads it further over work
> that was never waiting on it.
>
> It also is not free. KV cache grows linearly with batch size, a timeout takes the
> whole group instead of one subject, and latency becomes per-batch so every subject
> waits for the slowest.

**Follow-up: "Did you measure that, or is it a roofline argument?"**

Answer this one straight. It is the most likely place you get caught.

> The ratio is measured — I have the prompt token counts and the per-stage timings.
> The roofline part is reasoned, not measured: the 295 MB-per-sequence KV figure and
> the FLOP-per-byte number are back-of-envelope. I built the sweep harness so it can
> be settled properly — it runs identical crop-and-prompt pairs at every batch size
> and reports wall time, peak memory, *and* verdict agreement against size 1, because
> batching left-pads mixed-length prompts and padding is where a batched answer could
> legitimately drift. I have not published a result from it. So the setting defaults
> to 1 on an argument, not on a measurement, and that is a gap.

### Q3. "You put a 2B VLM on an edge box. What broke?"

> The most interesting failure was a segfault, and I got the diagnosis wrong twice
> before I got it right.
>
> Moving the VLM to the GPU measured 2.5× faster in isolation, so I shipped it. Then
> the app started dying about twenty seconds in — clean exit, no traceback, no Python
> exception. I assumed it was my sandbox killing long-running processes, and said so.
> It kept happening on hardware I did not control the lifecycle of, which is what
> made me actually look.
>
> The macOS crash reports had one entry per run, all of them dated after the device
> change. The faulting stack was inside PyTorch's MPS convolution, building an
> Objective-C dictionary — and a second thread was simultaneously destroying an
> MPSGraph. Two threads driving Metal at once. Before my change only YOLO used the
> GPU; after it, both stages did.
>
> I isolated it with a three-cell matrix instead of guessing: both on GPU segfaults
> before the model even finishes loading, Stage 2 on CPU survives with 9 verdicts in
> 75 seconds, Stage 1 on CPU survives with 29.
>
> So one stage gets the GPU, and the choice is not close: losing it costs Stage 1
> about 10 milliseconds a frame and would cost Stage 2 about 10 seconds a verdict.
> A thousandfold asymmetry. I made the two device settings resolve together in one
> place rather than independently, so the unsafe pairing cannot arise by default.

**Follow-up: "Why not a lock?"**

> That is the real alternative and I did not measure it. A lock keeps both stages on
> the GPU but stalls Stage 1 for the full 1.9 seconds of every verdict. The
> arbitration costs a flat 10 ms per frame and never stalls. I took the simpler,
> cheaper one — but I should be clear that I avoided the race rather than fixing it,
> and if Stage 1 ever needed the GPU badly the lock is the thing to measure.

**Follow-up: "How do you know it is fixed?"**

> Ninety seconds on the demo clip with defaults, no new crash report, 34 verdicts,
> inference steady at ~1,870 ms — matching the benchmark. Plus seven unit tests
> pinning the arbitration, including that an explicit unsafe pairing is honoured
> with a warning rather than silently overridden.

### Q4. "What would you do differently, and what don't you know?"

The strongest answer in the set. Do not soften it.

> Three things I do not know.
>
> **The edge path is unmeasured.** The whole point is a Pi-class box, and I have
> never run it on one. The backend abstraction and the llama.cpp profile exist and
> are unit-tested, but no server has ever answered a request from this code. I know
> exactly which measurement settles it — benchmark the HTTP backend against a real
> llama.cpp server on the same crops on the target board — and I know why my current
> numbers do not: a Pi has no Metal GPU, and this model on CPU is 11.7 seconds.
>
> **Accuracy is a flag, not a result.** When I swapped models, the old and new
> disagreed on 4 of 9 queries. But that comparison is confounded three ways —
> different revision, different quantisation, different device — so it tells me to
> look, not what is true. Settling it needs a labelled set, which I do not have.
> Nine queries can settle a 2.5× latency gap. They cannot settle accuracy.
>
> **The dominant cost is untouched.** 89% of the query is the vision encoder and
> nothing in the design addresses it. Everything I optimised — batching, encode
> hoisting, the probe path, crop sizing — operates on the other 11% or on the queue
> around it.
>
> What I would do differently: measure before defaulting. I changed the device
> default on a benchmark that ran the VLM alone, with no detector thread. That is
> precisely the configuration that never happens in production, and it is why the
> benchmark could not have caught the segfault. The bench should exercise the real
> thread topology.

---

## 8. Numbers cheat sheet

Every number below traces to a committed artifact or a measurement in this repo.

**Latency, per call, Apple M2**

| | GPU (MPS) | CPU |
|---|---:|---:|
| YOLOv8n @720p | 16.9 ms | 26.8 ms |
| Moondream 2B 4-bit | **1,873 ms** | 11,712 ms |
| Moondream2 fp32 (old) | 3,327 ms | 4,757 ms |

**Where a 1,873 ms query goes:** vision 1,664 (88.8%) · prefill 207 (11.1%) · decode ~0

**Cost:** 0.08 CPU-s per query, vs 7.86 for the old fp32-on-CPU default — **98×** less CPU

**Model:** 2,032 MB on disk (vs 3,736 MB) · 729 image tokens in · 1 token out

**ARM CPU GEMM, prefill-shaped:** fp32 **24.5 ms** · fp16 110.9 ms · bf16 171.9 ms

**Optimisations, before → after**

| | Before | After |
|---|---:|---:|
| Box decode, 11 boxes on GPU | 5.4 ms | 0.15 ms |
| Crop above vs below the tiling cliff | 3.16 s | 1.42 s |
| 3 rules, one subject (encode hoisted) | 11.62 s | 9.37 s |
| Tracker ids per real person (IoU 0.45 → 0.2) | 4.2 | 2.0 |
| Probe vs generate, prefill only | 207 ms | 104 ms |

**Rejected:** 4-bit ONNX export — 10,363 ms vs 3,900 ms fp32 baseline, 6× CPU

**Marked analytical, not measured:** KV cache ~295 MB/sequence · ~280 FLOP/byte

---

## 9. Where you will get pushed

Each of these is a real seam. The one-line answer is the honest one.

| They ask | Say |
|---|---|
| "What was your Pi 5 p95?" | "I never ran it on a Pi. The profile is derived from Apple Silicon measurements and labelled unverified in the repo. The measurement that settles it needs the board." |
| "How much did llama.cpp help?" | "I don't know — it was never compiled or benchmarked. The HTTP abstraction is real and unit-tested against a mocked server; that is protocol conformance, not a result." |
| "Show me the batching numbers." | "I have the harness, not the reading. The default is set on a roofline argument. That's a gap and I'd close it before recommending the setting to anyone." |
| "Isn't 4-bit why it got faster?" | "No. It's 4-bit on disk only — the kernel is CUDA-only, so on Apple Silicon it dequantises to fp16 at load. The speedup is the newer model plus the GPU. The quantisation bought download size." |
| "You said quantisation makes prefill slower — but you shipped a 4-bit model?" | "Different mechanism. That finding was about paying an unpack cost on every GEMM. Here there is no 4-bit GEMM at all." |
| "So probing made it faster?" | "By 0.6%, which is inside the noise. It's a correctness change, not a latency one. Prefill did halve, but vision dominates." |
| "How often did the parser actually fail?" | "I can prove the mechanism — `parse('Crouching')` returns False, so do `'Bending over'` — but I don't have a clean frequency measurement, and two of my own docs disagree about it. I'd trust the mechanism and re-measure the rate." |
| "Why is the VLM in-process at all on an edge box?" | "It shouldn't be, and it isn't in the edge profile — that's what the HTTP backend is for. In-process is the laptop path." |
| "Only 9 test queries?" | "Enough to settle a 2.5× latency gap, not enough for accuracy. I say so in the evaluation and I don't claim an accuracy result." |

**Two known inconsistencies in the repo itself.** Know them before an interviewer
finds them in your own document:

1. The README and ARCHITECTURE both say the parser failure happened "on 1 of every
   3 queries in the fixture set." No committed artifact contains that string, the
   quoted prompt does not exist in the repo, and the newer evaluation says it did
   *not* occur in the measured fixture set. The mechanism is provable; the frequency
   is an unsourced anecdote.
2. `EVALUATION-4bit.md` links to `EVALUATION.md`, which lives only on the
   `eval/moondream2-q4-onnx` branch — a broken link on the current branch.

Being the person who already knows the weak spots in their own write-up reads far
better than being caught by them.
