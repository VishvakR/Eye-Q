# Stage 1 Load Shedding — Size Floor and Co-occurrence Gating

Design rationale for the two Stage 1 trigger filters added in
`feat/stage1-load-shedding`. This document carries the parts that are not
recoverable from reading the diff: why each filter sits exactly where it sits,
and what silently breaks if it moves.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the system as a whole. Claims here
are pinned to symbol names rather than line numbers, because ARCHITECTURE.md's
own line citations have already gone stale relative to `vlm_worker.py`.

---

## 1. What already existed

Four load-shedding mechanisms were specified. **Two of them were already built
and already met the specification**, so they were not rewritten:

| Mechanism | Where it lives | State |
|---|---|---|
| Coalesce pending requests by `(label, track_id)` | `VLMWorker._coalesce` | **Already built.** Keeps the newest payload per subject, sorts newest-first, caps at `_BATCH_HARD_CAP = 64`. Counter: `triggers_coalesced`. |
| Drop frames older than a 5 s staleness threshold | `VLMWorker._prepare` | **Already built.** Groups events by `timestamp` (one source frame) and discards whole groups past `vlm_staleness_s = 5.0` *before touching pixels*. Counter: `triggers_stale`. |
| Verdict memoisation on a 300 s TTL + IoU movement threshold | `VLMWorker._rules_to_judge` / `_remember_verdict` | **Already built.** Keyed `(track_id, rule_id)`; re-judges iff `age >= vlm_verdict_ttl_s (300)` **or** (`bbox.iou(judged_bbox) < vlm_rejudge_iou (0.5)` **and** `age >= vlm_min_rejudge_s (20)`). Self-prunes at `_MEMO_PRUNE_AT = 1024`. Counter: `instances_memoised`. |
| Size floor: discard boxes whose longest edge is under 96 px | `VLMWorker._too_small` | Existed, but **at Stage 2** — see §2. |
| Co-occurrence gating on two intersecting classes | — | Did not exist. |

The `vlm_min_rejudge_s` floor in the memo is worth calling out because it is not
obvious from the specification: without it, a *walking* subject fails the IoU
test on nearly every frame and demands a fresh 1,900 ms inference every stride.
The TTL alone does not prevent this; the movement threshold needs its own floor.

Tests for the pre-existing three live in `tests/test_vlm.py` as
`TestTriggerCoalescing` and `TestVerdictMemo`.

---

## 2. Why the size floor moved to Stage 1

The floor was correct at Stage 2 but paid for too late. A 30 px subject that
`_too_small` was always going to reject had already cost:

- a ZMQ publish on `tcp://127.0.0.1:5557`,
- a ring-buffer read in `_read_frame` (and, on a wrapped buffer, a failed one),
- a slot in the batch that `vlm_max_instances_per_batch` was about to truncate —
  which means it could **displace a legible subject** from the priority sort.

That last point is the one that made this worth moving rather than leaving
alone. Shedding at Stage 1 is not merely cheaper; it stops an unanswerable
subject from crowding out an answerable one.

**Both floors are kept.** Stage 2's `_too_small` remains as a backstop against
anything else publishing on the detection endpoint, and against `_prepare` being
driven directly (which every test in `TestUnreadableSubjectsAreSkipped` does).
Both read the single `vlm_min_subject_px` setting and both call
`BoundingBox.pixel_size`, so the predicate is literally the same function and
the two cannot drift by a pixel.

> **Reading the counters:** once this landed, `vlm.instances_too_small` should
> read **zero**. That is the healthy steady state, not a regression — Stage 1
> now sheds first. A non-zero value means either the floors were deliberately
> set apart, or something other than this watchdog is publishing on `:5557`.
> The live counter to watch is `watchdog.detections_filtered_size`.

The setting keeps its `vlm_`-prefixed name despite now governing both stages,
because `.env.pi5` and `.env.ollama` set `EYEQ_VLM_MIN_SUBJECT_PX` (the Pi
profile uses `64`, correct for its 960×540 capture) and renaming it would
silently revert those appliances to the 96 default.

---

## 3. The gating ladder order is load-bearing

`label → ROI → size → pair → cooldown`

### Both new gates MUST sit above the cooldown

`CooldownGate.should_trigger` **mutates**: it writes `_last_trigger[key] = now`
and returns `True` in the same breath. Any gate placed *below* it therefore
spends a subject's cooldown for a detection it then discards.

Concretely: a person at 30 px would burn `person#7`'s cooldown slot and be
rejected by the size floor. When they walk closer two seconds later — now
perfectly legible — they are suppressed by a cooldown they never actually
spent. The appliance goes blind to the exact subject it exists to watch, and no
counter reveals it.

This mirrors a decision Stage 2 already made one stage later: `_prepare`
deliberately does **not** memoise a subject it skipped for being too small,
with the same reasoning in its comment ("the same person will be worth asking
about once they are closer").

Two tests exist solely to lock this in:
`test_the_floor_does_not_consume_the_cooldown` and
`test_the_pair_gate_does_not_consume_the_cooldown`.

### Why size sits after the two scope gates

Not for cost — the arithmetic is trivial either way. It is for **counter
attributability**. The ladder is an `if/elif` chain, so each detection is
attributed to exactly one drop reason. Placed first, `detections_filtered_size`
would be dominated by distant background objects of classes no rule watches, and
would stop mapping onto any decision an operator could make. Placed after
`label` and `roi`, a rising value means "your rules want these and the camera is
too far", which is actionable: lower the floor, or move the camera.

---

## 4. Both gates live in Pass 3, never Pass 1

`_process_frame` runs three passes: decode + confidence filter, then whole-frame
tracking, then per-detection gating. It is tempting to reject small boxes in
Pass 1, before they cost anything. **Do not.** Two things break, and both are
silent.

### It breaks the display-vs-trigger invariant

Gating decides whether to *wake the VLM*, not what the operator's live feed
shows. A rule watching `person` must not blind the operator to the forklift
beside them, and YOLO scores every class regardless, so drawing them all is
free. Every confident detection is appended to `self._recent` with a `fired`
flag — filtering in Pass 1 removes it from the overlay entirely.
`TestDisplayVsTriggerScope` guards this.

### It breaks the tracker, which is the expensive half

This is the non-obvious consequence. Filtering in Pass 1 also removes the box
from `GreedyIoUTracker.update()` in Pass 2. A subject approaching the camera
would then be *untracked* below the floor and *tracked* above it — so they cross
the floor carrying a **brand-new `track_id`**, the old one having aged out or,
worse, been greedily matched to someone else.

Stage 2's verdict memo is keyed `(track_id, rule_id)`. A new id on every
approach is a guaranteed cache miss. The filter would **increase** VLM load,
which is the precise opposite of its purpose. `max_age_frames = 60` cannot
rescue this, because the track was never fed in the first place.

Guarded by `test_a_subject_under_the_floor_still_reaches_the_tracker`.

---

## 5. Co-occurrence: why `overlap_ratio`, not `iou`

`BoundingBox.iou` answers *"are these the same object"* — the right question for
tracking, the wrong one for co-occurrence.

A cell phone fully inside a person's box scores **IoU ≈ 0.028**. Not because the
phone is barely inside the person, but because the person is roughly thirty-six
times larger, and IoU divides by the *union*. Any IoU threshold loose enough to
admit that pair is also loose enough to admit two boxes that merely graze at a
corner.

`BoundingBox.overlap_ratio` divides by the **smaller** box's area instead, which
asks what the operator actually means — *"is the phone in the person?"* — and
returns ~1.0 for full containment at any size ratio. It is symmetric, because
`min` is, so a rule never depends on which box the caller named first.

The default `yolo_pair_overlap = 0.5` reads as "at least half the smaller object
lies inside the larger". Bare `intersection > 0` was rejected as the opposite
failure mode: YOLO boxes are generous and axis-aligned, so a phone on a desk two
metres behind someone clips their box by a pixel, and at 30 FPS that is a
firehose.

`test_iou_would_have_missed_this_pair` asserts both halves — that IoU is under
0.03 *and* that the watchdog publishes — so a future "simplification" back to
`iou` fails loudly.

---

## 6. Co-occurrence: what the partner is exempt from

The pair search runs over the **raw** candidate list from Pass 1, which has had
only the confidence filter applied. Two exemptions follow, and both fail
silently if broken.

### Exempt from the label allowlist

A rule targeting only `person` causes `sync_rules` to narrow `target_labels` to
`{"person"}`. If the pair search ran after the label gate, the `cell phone`
candidate would already be gone and the gate could never fire.

This works today **only because `model.predict()` is given no `classes=`
argument**, so `candidates` holds every COCO class YOLO scored. Ultralytics
accepts `classes=` and restricting inference to the target labels is an obvious
future "optimisation" — it would disable the pair gate completely with no
counter moving. There is a comment at the `predict()` call saying so.

### Exempt from the size floor

A cell phone is *always* under a 96 px floor. If `_pair_blocked` iterated a
size-filtered list, the gate could never fire — and it **fails closed**, so
Stage 2 simply goes quiet while every counter looks like a peaceful scene.

`test_a_tiny_partner_is_not_dropped_by_the_size_floor` pins this: the phone
counts as a valid partner *and* increments `detections_filtered_size` for
itself, in the same frame.

---

## 7. Co-occurrence: the primary's own box publishes

When `person + cell phone` matches, the published `DetectionEvent` carries the
**person's** box and the **person's** `track_id`. Not a union, not both.

A union box has no identity of its own. Synthesising one would break, all at
once: `_coalesce`'s `(label, track_id)` key, `_rules_to_judge`'s
`(track_id, rule_id)` memo, `_is_new_subject`, and `_remember_verdict`. Worse,
a union's shape *changes every frame* as the phone moves in the hand, so even a
stable synthetic id would fail the `rejudge_iou` test constantly. That would
disable Stage 2's main load control in the name of a load-shedding feature — a
net loss.

Publishing both as separate events was rejected for doubling Stage 2 load on one
real event, and for filing a verdict against a 6×13 px phone that the size floor
discards anyway.

The consequence to know: the gate answers *"does this co-occurrence exist"*, and
the crop is still of the primary. `vlm_crop_padding = 0.15` around a person
already contains a phone that is by definition ≥50% inside them. If a future
pair needs both objects visible (`person + truck`, where the partner is not
contained), `vlm_crop_padding` is the lever — not a union box.

The cooldown key is unchanged (`f"{label}#{track_id}"` on the primary). Folding
the partner's id in would let one person re-trigger by swapping phones, and
would churn the key every time the tracker reassigns a 6×13 px box.

---

## 8. Why the pair requirement is per-rule, and when it does nothing

`pair_label` is a column on the `rules` table, not a global setting. This has a
real weakness, stated plainly because it is invisible otherwise.

`sync_rules` hands Stage 1 the **coarsest** setting across all rules — label
union, minimum confidence, minimum cooldown — because Stage 1 must never filter
out something a rule still wants. A pair requirement is a *narrowing*, and there
is no coarsest narrowing. So with:

- Rule A: watches `person`, no pair
- Rule B: watches `person`, pairs with `cell phone`

the only composition that keeps Rule A firing is **no gate at all**. Rule B's
pair requirement is dropped, and every person wakes the VLM.

That is the correct outcome, but it looks like the feature is broken. So
`sync_rules` logs a warning naming the rule responsible:

```
Rule 'Forklift watch' targets 'person' with no pair_label, so the
'person + cell phone' gate on rule 'Phone use' cannot be enforced at
Stage 1 — every person will still wake the VLM.
```

**Why per-rule was chosen anyway.** A global setting would always shed, but the
two failure directions are not symmetric:

- Per-rule fails toward **over-triggering** — costs inference, wastes money.
- Global fails toward **silently blinding a rule** the operator never meant to
  narrow — misses violations.

On a safety appliance the second is far worse. A global `EYEQ_YOLO_REQUIRE_PAIR`
would quietly stop a plain `person` rule from ever firing, with no per-rule
signal that it had happened. Paying in wasted inference is recoverable; missing
the event the appliance exists to catch is not.

---

## 9. Counter reference

Both new counters appear under `watchdog` in `GET /api/status`.

| Counter | Rising means |
|---|---|
| `detections_filtered_size` | The camera is too far for the floor. Lower `vlm_min_subject_px` rather than accept the blind spot — a skipped subject is never judged at all, so this buys precision with recall. Note it is measured in *captured* pixels, so it moves with `frame_width`/`frame_height`. |
| `detections_filtered_pair` | The gate is working. But if it tracks `detections_total` closely, the partner class is never being detected at all — check `yolo_confidence`, which small classes like `cell phone` rarely clear at the 0.25 default. |
| `vlm.instances_too_small` | Should be **zero**. See §2. |
| `triggers_coalesced`, `instances_shed` | Should both *fall* relative to the pre-change baseline, since Stage 1 now sheds ahead of Stage 2's four layers. |
