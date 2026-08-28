# Identity Churn — Why the VLM Was Called 2.6× Too Often

Design rationale for the spatial verdict memo. The short version: every load-
control layer in this appliance is keyed on `track_id`, and `track_id` is not
stable. This records the measurement, the fix, and the correctness cost.

Reproduce any number here with `bench/replay_load.py`.

---

## 1. The measurement

`cctv_demo.mp4` replayed for 300 s under `demo_cctv.py`'s settings — 1280×720,
`conf=0.30`, `cooldown_s=12`, `min_subject_px=96`, two enabled rules, TTL 300 s:

```
distinct track ids    80          (for ~6 real people — max 6 simultaneous)
new ids per pass      [12, 14, 14, 14, ... 14, 13]   over 17 loops

Stage 1: size=1777  cooldown=5391  published=190
Stage 2: lookups=380  hit rate 67.4%
   misses  new_track=122   ttl=0   moved=2

INFERENCES DEMANDED   124  (0.413/s)
```

**80 track ids for about six people — thirteen ids per person.** And the miss
split is the whole story:

| Miss reason | Count | Share |
|---|---|---|
| `entry is None` — a **new track id** | **122** | **98.4%** |
| TTL expired | 0 | 0% |
| Subject moved (`rejudge_iou`) | 2 | 1.6% |

**Zero TTL expiries in 300 seconds.** `vlm_verdict_ttl_s=300`,
`vlm_rejudge_iou=0.5` and `vlm_min_rejudge_s=20` were doing nothing at all.
Raising any of them could not have helped — nothing was living long enough to
expire. This is the finding that matters, because those are the three knobs an
operator reaches for first.

With perfect identity the same scene needs ~12 inferences per TTL window.
Measured: 124. **Churn cost roughly 10×.**

## 2. Why the gates failed together

They look independent. They are not — both are keyed on the same unstable
value, and a fresh id defeats both in the same instant:

- `CooldownGate.should_trigger` finds `last is None` and returns `True`
  (`app/pipelines/detector.py`), so Stage 1 publishes.
- `_rules_to_judge` finds `entry is None` (`app/pipelines/vlm_worker.py`), so
  Stage 2 infers.
- `_is_new_subject` returns `True`, so the subject is sorted *ahead* of genuine
  re-checks in the batch.

There was no gate anywhere in the pipeline that a new id did not walk straight
through.

## 3. Where the ids come from

**Looping playback (the demo).** The clip is 17 s; `VideoCapture` seeks back to
frame 0 on EOF. On rewind the previous pass's tracks are still alive —
`max_age_frames=60` is 10 s at 6 FPS — but the frame-1 boxes are 17 s of motion
away, so nothing matches and ~14 fresh ids mint every loop, forever. About 85%
of the 80 ids are loop-boundary artefacts.

The tracker is behaving **correctly** here. The boxes genuinely do not overlap.
Resetting the tracker on seek would not change the id count; those subjects are
new to it either way.

**Continuous footage.** Churn is smaller but real. On `demo1.mp4` (37 s, no
loop) a person visible in 105 of 555 frames still produced 8 distinct ids, from
detection flicker alone — gaps longer than `max_age_frames` retire a track that
then returns as a stranger. Add greedy id swaps when two people cross, and
occlusions longer than 10 s, and the documented "~2 ids per real person" holds.

## 4. The fix — a spatial second tier on the memo

The memo now has two tiers, tried in order:

1. **Exact**, on `(track_id, rule_id)`, with the TTL and movement rules
   unchanged.
2. **Spatial**, on position, tried *only* when the exact lookup misses because
   the id is new. Scans memo entries for the same rule and the same label,
   within `vlm_crossid_reuse_s`, and reuses the best match whose
   `bbox.iou(judged_bbox) >= vlm_crossid_iou`.

A hit is written through under the new key, so the next frame takes the O(1)
exact path instead of rescanning.

The label is carried in the memo entry rather than the key precisely because
this tier matches on geometry, and geometry alone would hand a forklift's
verdict to the person driving it.

### Measured effect

Same 300 s replay, `--crossid-reuse-s 0` against the default 45:

| | Baseline | With spatial memo |
|---|---|---|
| distinct track ids | 80 | 80 *(unchanged — this is a Stage 2 fix)* |
| Stage 1 published | 190 | 190 *(unchanged)* |
| memo hit rate | 67.4% | **87.4%** |
| — hits exact | 256 | 256 |
| — hits spatial | 0 | **76** |
| miss `new_track` | **122** | **46** |
| miss `ttl` / `moved` | 0 / 2 | 0 / 2 |
| **inferences demanded** | **124** | **48** |
| inferences per second | 0.413 | **0.16** |
| duty cycle @ 1.9 s/query | 0.78 | **0.30** |
| duty cycle @ 11.7 s/query | 4.83 | 1.87 |

**61% fewer inferences, a 2.6× reduction.** The important check is not the
headline: `new_track` fell by exactly 76 and `hits_spatial` rose by exactly 76.
The fix works for the reason diagnosed rather than by luck.

The 46 remaining `new_track` misses are subjects at genuinely new positions —
arrivals, and people who moved between loop passes. Those *should* be judged.

On an accelerator this takes the appliance from near-saturation (0.78) to
comfortable (0.30). On CPU at 11.7 s/query it is still oversubscribed; that is a
backend problem, not a load-control one.

## 5. The correctness cost, stated plainly

**This is the one place the appliance trades correctness for load.** Reusing a
verdict across identities can hand person B the answer earned by person A. A
wrongly reused "No" is a **missed alert** — the bad direction on a safety
appliance.

Three things bound it, and all three matter:

1. **`vlm_crossid_iou = 0.7`** is far stricter than the `0.2` the tracker itself
   uses to assert two boxes are the same object. The gate never claims an
   identity the tracker would have rejected — it only recovers one the tracker
   lost.
2. **`vlm_crossid_reuse_s = 45`** is a separate, much shorter window than the
   300 s TTL. Cross-identity reuse can be shortened or disabled (`0`) without
   touching same-subject reuse.
3. **`instances_memoised_spatial`** is counted separately on `/api/status`, so
   an operator can see how much of their shedding rests on this tier.

**The accuracy cost is unmeasured.** `bench/fixtures/labels.json` does not
exist, so `bench/eval_accuracy.py` will not run — see the Gaps section of
`ARCHITECTURE.md`. Everything above is a *load* measurement. Do not read it as
evidence the change is safe; read it as evidence the change is bounded.

## 6. Diagnosing this on a live appliance

`/api/status` now exposes the memo miss reasons, plus seven counters that
existed all along and were never returned:

- **`memo_miss_new_track` dominating** → identity is churning. The TTL is
  irrelevant however high you set it. This tier is the lever.
- **`memo_miss_ttl` dominating** → verdicts really are ageing out. Raising
  `vlm_verdict_ttl_s` will help.
- **`memo_miss_moved` dominating** → subjects are walking out of their own
  boxes. Raise `vlm_min_rejudge_s`.
- **`instances_memoised_spatial` ≫ `instances_memoised`** → almost all your
  shedding depends on cross-identity reuse. Worth tightening `vlm_crossid_iou`
  and checking whether the tracker is being defeated by something fixable
  (frame rate, occlusion, a camera that drops frames).
- **`detections_filtered_cooldown` dwarfing every other Stage 1 drop** → the
  cooldown is carrying the whole load and nothing else contributes.

At Stage 1 the churn signature is **`cooldown_keys` ≫ `tracks_active`**: keys
are `label#id` and prune on age, so ids minting faster than the cooldown ages
them out shows up as a growing ratio while `tracks_active` stays flat.

## 7. What was considered and not done

- **A motion model in the tracker.** Constant-velocity prediction would cut
  churn at source and help fast movers. It does nothing for detection flicker,
  for occlusion past `max_age_frames`, or for a video seek — which is most of
  the churn here. Worth revisiting; not the first lever.
- **Appearance hashing.** Most robust to movement, and the only approach that
  would recognise a person who moved *and* changed id. But it is net-new
  machinery, fragile under lighting and pose change, and no such utility exists
  in the repo.
- **Raising `vlm_verdict_ttl_s`.** Measured to be useless here: zero TTL
  expiries in 300 s.
- **Changing Stage 1's cooldown.** Stage 1 already sheds 73% (5391 of 7358).
  The leak was entirely at Stage 2.
