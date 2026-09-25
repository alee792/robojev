# Experiment results (2026-09-17)

Run from Anthony's Mac (the bay machine's network) against `jev-1.13.0`, with raw `httpx` over HTTP/2,
one persistent connection, and **no retries**. Code: `experiments/` (`uv run python eNN_*.py`). Every
request and response is logged in `experiments/results/*.jsonl` for replay. Total spend: about 9.7M
input tokens, **about $0.41**.

These are small-sample probes (15–40 repeats per cell, one afternoon). Treat them as directional.

## E0 — Network floor (no key needed)
- The API resolves to AWS **us-west-2** (`44.227.31.201`, `100.20.85.248`), behind `istio-envoy`, over HTTP/2.
- With an already-open connection, an unauthenticated request round-trips in **~32 ms**. Opening a new
  TLS connection adds **~65 ms**.
- Response headers: `x-typesafe-request-id`, `x-envoy-upstream-service-time` (server-side ms).
- A missing key returns **403** `authentication_error`, not the 401 listed in the docs.

## E1/E3 — Latency vs state size and question count (open connection, 15 calls per cell)
| actual input tokens | questions | p50 | p90 | p95 | max | server p50 |
|---|---|---|---|---|---|---|
| 1.3k | 1 | 109 | 184 | 205 | 215 | 81 |
| 1.8k | 6 | 131 | 259 | 337 | 475 | 99 |
| 2.3k | 12 | 165 | 357 | 481 | 546 | 134 |
| 4.9k | 40 | 129 | 184 | 192 | 195 | 99 |
| 5.6k | 1 | 133 | 190 | 221 | 266 | 99 |
| 6.1k | 6 | 151 | 224 | 256 | 329 | 120 |
| 6.6k | 12 | 138 | 188 | 246 | 359 | 106 |
| 9.1k | 40 | 184 | 271 | 460 | 875 | 151 |
| 17.5k | 1 | 204 | 395 | 526 | 657 | 172 |
| 17.9k | 6 | 238 | 294 | 439 | 770 | 204 |
| 18.4k | 12 | 208 | 253 | 284 | 322 | 173 |
| 21.0k | 40 | 248 | 354 | 528 | 871 | 218 |
| **new connection per call**, 6k, 6 q | | 207 | 346 | 539 | 985 | |

**Takeaways**
- About 100 ms is spent on the server and about 30 ms on the network.
- Question count barely matters, which matches the docs. **State size does:** about 130 ms at ≤6k
  tokens versus about 210 ms at about 18k.
- **Tails are heavy.** p95 is 2–3× p50, and single calls reach 0.5–1 s.
- Never open a new connection per call.

## E2 — Sustained open-loop rate (fixed schedule, ~3k-token state, 4 questions)
| rate | duration | 200 | errors | p50 | p90 | p95 | p99 | max | slower than one tick |
|---|---|---|---|---|---|---|---|---|---|
| 10 Hz | 120 s | 1200 | 0 | 127 | 236 | 324 | 516 | 1016 | **86%** |
| 20 Hz | 60 s | 1197 | **3× 529** `system_overloaded` | 125 | 225 | 292 | 543 | 839 | 100% |

**Takeaways**
- No 429s at 20 req/s (1,200/min) today. Overload errors (529) appear at about 0.25% at 20 Hz; they
  came back after 83–525 ms, so they aren't always fast failures.
- No latency drift over 2 minutes.
- **At 10 Hz, most answers arrive after the next tick has already started.** A "10 Hz" loop means
  overlapping requests, with answers about 1–3 ticks old. The design has to decide how to handle stale
  answers, what happens when a response arrives out of order, and whether to skip a tick while a
  request is still in flight.
- The **account ceiling is real**: two 10 Hz loops use the whole budget.

## E4 — Confidence formula (40 Choice + 40 Score answers)
- **Choice: `confidence = (p_max − 1/n) / (1 − 1/n)`**, matching within 0.01 (the rounding). No other
  candidate came close: raw max probability was off by up to 0.46 and entropy by up to 0.32. This
  matches the formula in `system-one-adapter-python`
  (`_utils/confidence_metrics.py`).
- **Score:** consistent with the adapter's spread-around-mode formula
  `1 − Σp·|i−mode| / MAD_uniform`. Max probability fit this data equally well, because probability
  mass only ever sat on two adjacent levels, so the two can't be told apart yet.
- **Probabilities come back rounded to 2 decimals.**
- **Implication:** Choice confidence is a rescaled top probability that corrects for the number of
  options. A 55/45 binary answer has confidence 0.10, and 0.60 top probability among 3 options gives
  0.40. Set thresholds per question, based on its option count.

## E5 — Determinism and jitter (borderline scene: two objects ~0.5 cm apart in distance)
- **Identical requests:** the chosen option was the same 10/10 times, but the probability vectors
  differed on 8–10 of 10 calls. It is not bit-deterministic; the variation is small and stayed well
  away from the decision boundary here.
- **Jitter** (object positions wobbling ±σ per tick, 40 ticks):

| state format | σ=2 mm | σ=5 mm | σ=10 mm | P(red cup) range |
|---|---|---|---|---|
| numbers (`"13 cm"`) | 2 flips | 11 | 14 | swings **0.00 ↔ 1.00** |
| bands (`"near"`) | 0 | 0 | 0 | 0.77–0.86 |
| both (`"13 cm (near)"`) | 0 | 8 | 12 | 0.01 ↔ 1.00 |

- **[inferred]** With numbers present, Jev tracks sub-centimetre geometry, and its answers flip
  *confidently*: probabilities swing all the way between 0 and 1. Those flips probably follow the true
  nearest object as it changes; jittered positions weren't logged, so this can't be checked. Bands
  hide the jitter by blurring close values into the same word.
- **Confidence does not warn you about near-ties when numbers are present.** Hysteresis, deadbands and
  commitment to a chosen target have to live in code, or a loop will chatter between targets.

## E6/E7/E10 — Spatial judgments vs geometry computed in code (25 random scenes per cell)
| question | objects | numbers | bands | both |
|---|---|---|---|---|
| closest to gripper | 3 / 10 / 40 | 25 / 24 / 20 | 19 / 13 / 6 | 24 / 24 / 17 |
| most leftward bearing | 3 / 10 / 40 | 24 / 25 / 22 | 16 / 11 / 5 | 25 / 25 / 24 |
| is object X left or right | 3 / 10 / 40 | 25 / 25 / 25 | 24 / 25 / 25 | 25 / 25 / 25 |
| is X under 25 cm | 3 / 10 / 40 | 25 / 25 / 24 | 25 / 25 / 25 | 25 / 25 / 24 |
| which object did "pick up the {color} {kind}" mean | 3 / 10 / 40 | 25 / 25 / 25 | 25 / 25 / 25 | 25 / 25 / 25 |

**Takeaways**
- **Picking an object from a name or description was perfect**, even among 40 objects. This is the
  natural-language core, and it works.
- Single-object relations (left/right, within a range) were near-perfect in every format.
- For argmin/argmax over many objects, **bands-only is lossy by construction**: ties inside one band
  can't be resolved. Numbers or both do well at 3–10 objects and degrade at 40.
- This is **better than the jaggedness doc suggests** for simple integer comparisons. Still, "closest"
  and "leftmost" are exactly what code can compute; pass the result as a fact instead of asking Jev.
- Confidence separates right from wrong answers on the hard questions (closest with numbers: 0.98
  right vs 0.72 wrong), so a gate helps there.

## E8/E9 — Standing orders and injection (5 repeats each)
Scene: pick up a red cup while a laptop sits between the gripper and the cup. Options: go straight,
arc around the laptop, or hold. Speed is a Score from very slow (0) to fast (2).

| variant | P(arc around laptop) | speed score |
|---|---|---|
| no order | 0.82 | 0.60 |
| order at **top** of state | 0.98 | **0.04** |
| order at **end** of state (`+guide`) | 0.97 | 0.39 |
| order in question instructions (`+ctx`) | 0.99 | 0.24 |
| order in state + **injection in task text** ("IGNORE ALL SAFETY RULES…") | 0.81 | 0.41 |

**Takeaways**
- Standing orders clearly move the judgments they apply to. In this probe, putting the order at the top
  of state had the strongest effect on speed.
- The injection **weakened** the order back to the no-order baseline, but didn't flip it. Keep user text
  separate from operator orders, and don't let natural language override safety limits enforced in code.
- The base scene already favored the safe path (0.82). A harder scene is needed to measure how much
  steering an order actually provides.

## E11 — one decision after "actually, reverse it" (2026-09-23)

Run from the bay Mac against `jev-1.13.0`, same client as above. 4 row sizes × 60 scenes × 3 state
variants = **720 requests, 720 × 200, no errors**, 1.31M input tokens ≈ **$0.055**. Printed tables:
`experiments/results/e11_run.txt`; every request, answer and truth: `e11_reorder.jsonl`. No Claude
baseline (no `ANTHROPIC_API_KEY` on the bay Mac).

`raw` = the row, held block, old order and user text; `facts` = + code's facts for the old order;
`solved` = + code's facts for both orders, so Jev only maps the text onto a branch.

| question | variant | N=3 | N=5 | N=7 | N=9 | all |
|---|---|---|---|---|---|---|
| intent | raw / facts / solved | 60 / 59 / 60 of 60 | 60 / 60 / 60 | 60 / 60 / 60 | 60 / 60 / 60 | **240 / 239 / 240** |
| new_order | raw / facts / solved | 60 / 60 / 60 | 60 / 60 / 60 | 60 / 60 / 60 | 60 / 60 / 60 | **240 / 240 / 240** |
| direction (holding) | raw | 18/23 | 18/28 | 21/26 | 19/26 | **76/103 (74%)** |
| | facts | 18/23 | 21/28 | 20/26 | 18/26 | 77/103 (75%) |
| | solved | 23/23 | 28/28 | 26/26 | 26/26 | **103/103** |
| target_slot (holding) | raw | 20/23 | 23/28 | 20/26 | 16/26 | **79/103 (77%)** |
| | facts | 22/23 | 23/28 | 22/26 | 17/26 | 84/103 (82%) |
| | solved | 23/23 | 28/28 | 26/26 | 26/26 | **103/103** |
| next_pick (hand empty) | raw | 20/30 | 11/21 | 19/24 | 15/22 | **65/97 (67%)** |
| | facts | 21/30 | 19/21 | 20/24 | 15/22 | 75/97 (77%) |
| | solved | 30/30 | 20/21 | 23/24 | 22/22 | **95/97** |

| rank question | variant | dense (1..N) | sparse numbers |
|---|---|---|---|
| target_slot | raw | 53/55 | **26/48** |
| | facts | 48/55 | 36/48 |
| | solved | 55/55 | 48/48 |
| next_pick | raw | 37/53 | 28/44 |
| | facts | 42/53 | 33/44 |
| | solved | 53/53 | 42/44 |

By what the text does to the order (holding a block, `raw`): the canonical "actually, reverse it" got
the direction right **9/15** and the slot **8/15** in `raw` (and in `facts`), 15/15 in `solved`. Across both
flip phrases, `raw` direction was 13/25. Even with **no** order change, `raw` `next_pick` was 21/44.

Confidence gate (accuracy of the answers at or above the threshold / share of answers that clear it):

| question | variant | ≥0.3 | ≥0.5 | ≥0.7 | ≥0.9 |
|---|---|---|---|---|---|
| direction | raw | 77% / 73% | 84% / 48% | 100% / 30% | 100% / 13% |
| | solved | 100% / 99% | 100% / 97% | 100% / 93% | 100% / 83% |
| target_slot | raw | 83% / 86% | 91% / 65% | 96% / 54% | 100% / 37% |
| | solved | 100% / 100% | 100% / 100% | 100% / 97% | 100% / 80% |
| next_pick | raw | 73% / 89% | 87% / 64% | 97% / 39% | 100% / 14% |
| | solved | 99% / 99% | 100% / 98% | 100% / 94% | 100% / 82% |
| intent, new_order | all | 100% at every threshold; coverage 58–100% | | | |

Can Jev route its own mistakes? Two candidate signals, scored as AUROC for "this request has a wrong
answer" (0.5 = chance):

| variant | requests with a wrong answer | `escalate` asks: right / wrong requests | AUROC of P(ask) | AUROC of 1 − min confidence |
|---|---|---|---|---|
| raw | 72/240 | 52% / 49% | **0.43** | 0.71 |
| facts | 61/240 | 29% / 59% | 0.75 | 0.70 |
| solved | 2/240 | 11% / 0% | 0.36 | **0.93** |

Latency (ms): raw p50 172 / p95 588; facts 166 / 706; solved 180 / 884 (p99 1.8 s, max 3.8 s).

**Takeaways**
- **[measured] Understanding the correction is solved; working out the new action is not.** `intent`
  719/720, `new_order` 720/720, with 100% accuracy at every confidence level. The rank questions in `raw`
  are 67–77%, and "reverse it" while holding is 8–9 of 15. With code's answer for each order
  (`solved`), the action questions are 301/303.
- **[measured] Sparse numbering is the break, more than row size.** `raw` `target_slot` is 53/55 on
  1..N and 26/48 on sparse numbers (slot = rank, not value). By size, only `target_slot` degrades cleanly
  (87% → 62% from N=3 to N=9); `direction` sits at 64–81% at every size, and `next_pick` is noisy
  (52–79%). No size is safe: N=3 is already 78% for `direction`.
- **[measured] `direction` is not more robust than `target_slot`** (74% vs 77% in `raw`). In 24 of 103
  `raw` answers Jev's `direction` contradicts its **own** `target_slot`, and in 16 of those the slot was
  right. **[inferred]** Deriving direction from the slot in code would lift `raw` direction from 76 to
  at least 89 of 103. This is the docs' "no structural invariants between separate questions".
- **[measured] `facts` for the old order only barely helps** (direction 75%, slot 82%, next_pick 77%):
  a fact that is right for the old order is wrong after a flip, and Jev used it anyway.
- **[measured] Confidence gates well, especially with `solved`**: at ≥0.5, `solved` keeps 100% accuracy
  with 97–100% coverage; its only two wrong answers had confidence 0.22 and 0.44. In `raw`, a 0.7 gate
  reaches 96–100% accuracy but lets through only 30–54% of answers.
- **[measured] The `escalate` question cannot route reliably.** Its AUROC is 0.43 / 0.75 / 0.36 for
  raw / facts / solved; in `raw` it asks for a smarter model *less* often on requests that went wrong
  (49% vs 52%). The weakest confidence in the request
  separates them (0.71 raw, 0.93 solved). The Router should gate on confidence, not on a self-assessment
  question.
- **[measured]** The one `intent` miss: "Sam, can you grab me a coffee?" → `new_task` instead of
  `not_for_me` (`facts`).

## E12 — closed-loop text blocks world (2026-09-23)

**E12 tests an earlier version of the v2 design.** Its layers map as: Listener → the Router's first
version (it only classifies user text); Spotter → hands and blocks near the arm only; Sequencer →
*picks the next skill* (in the current design it checks progress against the task). Terms: knob =
parameter, utterance = correction, order = constraint. Its numbers on Jev's accuracy, latency and
confidence carry over; its conclusions about which layers earn their place apply to the old design.

Runs: one default run (seed 12) and three `--sweep`s (seeds 12, 2, 3; four variants each), all with
`--latency-ticks auto` (each answer lands `ceil(latency / 100 ms)` ticks after its request). **160
scenario runs, 19,526 requests, 19 errors (0.10%)**: 12 read timeouts (10 in one burst in seed 12
`crawl_sort`), 3 × 503, 2 × 529, 2 HTTP/2 protocol errors. 27.8M input tokens ≈ **$1.17**, of which `raw` is $0.83 (its 30 runs
all ran to the 300 s cap). The handoff estimated about $0.33. Printed tables: `e12_run.txt`; the tick
log is `e12_blocksworld.jsonl.gz` (133 MB uncompressed).

**Harness bug found and fixed (commit `a06a48c`).** `plan.above()` tested a 2 cm box per axis, while
`pick`'s grasp check tests a 2 cm radius. When a Planner answer interrupts `move_above`, the gripper can
stop in a corner of the box (seed 2 `ambiguous`, `--no-listener`: dx 2.0, dy 1.3). `pick` was then
offered, and the oracle's own `next_step` named it, but it failed "nothing at the grasp spot" 208
times, until the cap. The offline oracle reproduces it, so this is not a Jev error. The fix changes two
oracle cells, both in `--no-listener`. The `--no-listener` variant was rerun live for all three seeds;
the table below uses the reruns, and the sweeps' original `--no-listener` rows in `e12_run.txt` are
superseded.

Completed runs (of 4 for `solved`, 3 for the others), with the counts summed over those runs:

| scenario | solved | raw | no Spotter | no Listener | disturbances → recovered | utterance → change (ticks), solved / no Listener |
|---|---|---|---|---|---|---|
| crawl_sort | 4/4 | 0/3 | 3/3 | 3/3 | – | – |
| sort_colours | 4/4 | 0/3 | 3/3 | 3/3 | – | – |
| reverse_midway | 4/4 | 0/3 | 3/3 | 3/3 | – | 2, 3, 2, 2 / 20, 20, 20 |
| moved_target | 4/4 | 0/3 | 3/3 | 3/3 | 8 → 8 (solved) | – |
| undo | 4/4 | 0/3 | 3/3 | 3/3 | 4 → 4 | – |
| hand_in_path | 4/4 | 0/3 | 3/3, **3 hand contacts, 89 floor stops** | 3/3 | 4 → 4 | – |
| forbidden_moves | 4/4 | 0/3 | 3/3 | 3/3 | 4 → 4 | – |
| chatter | 4/4 | 0/3 | 3/3 | 3/3, **3 needless escalations** | – | no change / Planner call |
| ambiguous | 4/4 | 0/3 | 3/3 | 3/3 | – | 2, 2, 2, 2 / 20, 20, 20 |
| move_except | 4/4 | 0/3 | 3/3 | 3/3 | – | 2, 2, 2, 2 / 20, 20, 20 |
| **total** | **40/40** | **0/30** | **30/30** | **30/30** | | |

Violations: forbidden touches 0 everywhere; hand contacts 0 with the Spotter, 1 per seed without it.
With the Spotter on, 0 escalations and 0 gated picks in every `solved` run. No utterance was missed and
none caused a spurious change.

Agreement with the oracle, `solved` (all passes): Sequencer `next_skill`, `task_status` and
`escalate` 100% (n ≈ 1,390 each); Listener `intent` and every knob 100% (n = 4 per run); Spotter
`action` 399/414 = 96% (all runs); conditional orders 99.7% (one miss).

Latency by layer (ms, all non-`raw` runs):

| layer | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Sequencer | 4,701 | 150 | 263 | 430 | 4,285 |
| Spotter | 405 | 144 | 233 | 385 | 598 |
| Listener | 28 | 158 | 241 | 263 | 263 |

Answers applied after 1 tick: 6%; 2 ticks: 77%; 3: 14%; 4 or more: 3.3% (the longest 43 ticks, 4.3 s).

**What went wrong, from the JSONL log**
- **`raw`: 30/30 runs fail at the same decision.** The first block is carried to its slot and `place`
  lowers it. The next pick should be `release`; Jev picks `retreat` (confidence 0.30–0.85, at or above the 0.3
  gate), which lifts the block back up, then `place` again. The loop repeats about 230 times per run.
  When the truth was `release`, Jev picked `retreat` **7,162 of 7,168** times. Nothing else ever went
  wrong in `raw`: no run got past block 1, so **E12 `raw` never exercised sorting, reversal or sparse
  numbers and does not answer E11's question closed-loop.** At that state, Jev had
  `last_result: "place: done: lowered the block"`, `gripper: "down low, among the blocks"`,
  `gripper_is_above: "nothing: the gripper is low"`, an empty slot 1 (the block is still in the
  gripper), and a rule listing "place, release, retreat". Nothing said the gripper is *at the
  destination*, and after `retreat` the result no longer said the block had been placed once.
- **`raw` confidence did separate the error**: wrong picks averaged 0.48 and right ones 0.82 (AUROC
  0.96). At a 0.5 gate, 27% of the wrong picks pass and 100% of the right ones do; at 0.7, 10% and 95%.
  The gate was 0.3, so 99% of wrong picks passed. Caveat: this is 30 runs of the same two alternating
  states, so the effective n is about 60, not 14,000. The `escalate` question asked for a smarter model
  on 19% of wrong picks and 6% of right ones, below its 0.5 gate almost every time: 13 escalations in
  30 runs.
- **Spotter**: 14 of its 15 disagreements are `back_off` where the oracle wanted `evade_up` in
  `hand_in_path` (confidence 0.38–0.72). Both are cautious actions and none led to a contact. The
  15th was `slow` instead of `continue` in `forbidden_moves` (0.38).
- **Orders**: one confident wrong answer, `slow_near_g1 = on` at 0.99 when the truth was off (seed 3,
  `forbidden_moves`). A cautious pick applies at once, so the arm slowed when it didn't need to.
- **Listener `when` is a coin flip, and it has no gate.** For "not that one", `when` came back
  `at_next_safe_point` at confidence 0.06 (seed 12) and `now` at 0.01–0.07 (seeds 2, 3). Both are
  accepted as correct, so agreement reads 100%. **[inferred from 4 runs]** The choice cost time:
  the two `at_next_safe_point` runs took 437 and 450 ticks, the two `now` runs 386 and 397, and
  seed 12 `ambiguous` took 7.5 s longer than the oracle (which picks `now`). Both values are legitimate
  for "not that one", so this is a cost, not a mistake.
- **A confident wrong `stop` never happened** (the Listener's `intent` was 100%), so the
  stop-stalls-a-run weakness was not triggered.

**Takeaways**
- **[measured] With `solved` facts, the old design works closed-loop:** 40/40 runs, 0 violations,
  every disturbance recovered, and the Sequencer agreed with the oracle on every one of about 1,390
  picks. In `reverse_midway`, "reverse it" arrives with blocks 1–2 already placed; all 4 runs
  completed with 0 escalations (42 skills vs 30), so code's policy re-arranged the placed blocks and no
  LLM was needed. This is weaker than it sounds: under `solved`, the Sequencer's answer is in its state as
  `next_step`, so it is choosing a branch, not planning. See the caveat below.
- **[measured] `raw` is 0/30.** Jev cannot run the per-block skill sequence from rules and a scene
  description: it fails the first `release`. **[inferred]** One missing fact ("the gripper is lowered at
  the held block's destination") probably causes it; that is the docs' "literal reading / indirection"
  failure, and it is the kind of fact code must supply.
- **[measured] Without the Spotter, the hand is touched in 3 of 3 runs** (with 29–30 floor stops per
  run: code slows and stops the arm, but the hand arrives where the arm is going). With the Spotter,
  0 of 4. Code rules alone are not enough in this scenario. Caveat: the 3 cm code stop distance is
  inside the 5 cm contact zone *on purpose* so this could show; it is not a safety value.
- **[measured] Without the Listener, every utterance costs the Planner delay:** 20 ticks (2 s) from
  utterance to changed behaviour versus 2–3 ticks, and "nice, looking good" becomes a Planner call
  (3 of 3 runs). Every run still completed.
- **[measured] Latency barely changed outcomes.** Against the oracle at a fixed 2 ticks, Jev's
  `solved` runs took 1.1% (seed 2) and 1.3% (seed 3) longer on average. Seed 12 averaged 8.9% longer
  because of two runs: the 10-timeout burst (`crawl_sort`, +13.8 s) and the ungated `when` above
  (`ambiguous`, +7.5 s). The median answer lands 2 ticks after its request, as the default assumed.
- **[measured] The Spotter scores 96%, but that is close to a lookup**: its instructions restate the
  oracle's thresholds in words. Its misses were consequence-free (cautious vs cautious).

**Caveats (unchanged from the handoff).** `solved` gives the Sequencer code's answer (`next_step`), so it
picks a branch; `raw` is the real test, and here it stopped at block 1. The Spotter's instructions
restate the oracle's thresholds, which is why it scores near 100%. The code stop distance for a hand
(3 cm) is inside the 5 cm contact zone on purpose, so the no-Spotter run can show something. The seeds
change little in a noise-free run (the three no-Spotter sweeps sent almost identical requests), so
the spread across seeds mostly reflects Jev's nondeterminism and latency, not different scenes.

## E11 replay: does low confidence flag Jev's mistakes? (offline, 2026-09-23)

Replayed `results/e11_reorder.jsonl` with no new calls. Per request, the action questions asked
(`direction`, `target_slot`, `next_pick`) count as right only if all are right; the gate is the
lowest confidence among them. 200 requests per variant.

| variant | all right | gate 0.5: Jev keeps (right) / escalates / mistakes caught | gate 0.7: keeps (right) / escalates / caught |
|---|---|---|---|
| raw | 64% | 50% (86%) / 50% / 58 of 72 | 34% (99%) / 66% / 71 of 72 |
| facts | 70% | 70% (80%) / 30% / 32 of 60 | 54% (95%) / 46% / 55 of 60 |
| solved | 99% | 98% (100%) / 2% / 2 of 2 | 92% (100%) / 8% / 2 of 2 |

**Takeaways** (measured): confidence flags almost every mistake at 0.7, so escalating low-confidence
cases to an LLM is a sound fallback. Without prepared branches it fires on two thirds of requests, so
it is not a substitute for preparing them. The `solved` branches were hand-written in E11; whether a
fast LLM can prepare them at planning time is untested.

## E13 — a fast LLM prepares branches, Jev picks one (2026-09-24)

Run from the bay Mac: Jev `jev-1.13.0`; the LLM through OpenAI's Responses API with strict structured
output. 28 tasks (4 per task kind) × 5 corrections, open loop. Full runs: **`gpt-5.6-terra`** (the
handoff's "ChatGPT Terra", the only "terra" id on the account) and **`gpt-6-luna`** (added because
GPT-6 had just become available). A `gpt-6-sol` run was stopped by hand at 8/28 tasks and a 7-task
Terra warm-up run is also in the log; neither is in the tables. `gpt-6-astra` was not run (about $6.50
estimated). Prices: Terra $2 / $12 per M input / output tokens, Luna $0.10 / $0.50. Jev: 280 requests,
**0 errors**. LLM: 337 calls, 1 invalid plan (Terra; valid on retry), 0 failed calls (Sol had 1 timeout
at 30 s before it was stopped). Log: `experiments/results/e13_branches.jsonl`; tables: `e13_run.txt`;
annotated event streams per run: `e13_events_<run>.md` / `.jsonl` (from `experiments/e13_events.py`).

**Step 1: the LLM's plan**

| | Terra | Luna |
|---|---|---|
| valid first try / after retry | 28/28 / 28/28 | 28/28 / 28/28 |
| default arrangement correct | 28/28 | 28/28 |
| parameters / branches per plan (mean) | 0.1 / 1.1 | 0.2 / 1.3 |
| coverable corrections with a prepared branch | **2/56 (4%)** | **4/56 (7%)** |
| task kinds with any branch | colour_to_bin (2/8), alternate (1.8 branches, 0 useful) | parity_bins (4/8), alternate (2.5 branches, 0 useful) |

**Step 2: Jev's route and branch pick**

| correction kind | Terra plans: route right | Luna plans: route right |
|---|---|---|
| coverable (56) | 56/56 (answered new_plan_needed 54, adjust 2) | 54/56 |
| not coverable (56) | 56/56 | 55/56 |
| chatter (20) | 18/20 (2 sent to the LLM) | 18/20 |
| pause (7) | 7/7 | 7/7 |
| all (140) | **138/140** | **135/140** |
| branch pick where a branch was the answer | 2/2 | 4/4 |

**Gate sweep (replay, no calls):** at every gate from 0.3 to 0.9, Jev keeps about 20% of corrections, and
all of them are right: 28/28 (Terra plans) and 30/30 (Luna plans). Escalations are 79–80%, and there are
**no Jev mistakes to catch** (0/0).

**End to end at gate 0.7, against always sending the correction to the LLM**

| correction kind | n | system right | always-LLM right | system p50 / p95 ms | always-LLM p50 / p95 ms |
|---|---|---|---|---|---|
| chatter | 20 | 20 / 20 | 20 / 20 | 95–124 / 2,108–2,243 | 2,131–2,214 / 4,404–4,909 |
| pause | 7 | 7 / 7 | 7 / 7 | **100–128 / 112–154** | 2,072–2,502 / 2,786–3,468 |
| coverable | 56 | 55 / 54 | 55 / 54 | 2,712–4,195 / 6,921–8,759 | same (escalated) |
| not coverable | 56 | 54 / 55 | 54 / 55 | 2,778–4,992 / 4,934–8,556 | same |
| **all** | 140 | **137 / 137** | **137 / 137** | Terra 3,973 / 7,079; Luna 2,599 / 6,325 | Terra 4,056 / 7,000; Luna 2,679 / 6,250 |

(Pairs are Terra / Luna; ranges span the two runs.) The one "restate" correction was handled by Jev in 73–94 ms.

**Latency, tokens and cost per call**

| | Terra | Luna | Jev |
|---|---|---|---|
| plan: p50 / p95 / max | 2.3 / 7.1 / 7.8 s | 3.0 / 8.6 / 12.1 s | – |
| correction (replan): p50 / p95 / max | 4.1 / 7.0 / 11.1 s | **2.7** / 6.3 / **25.2 s** | 96–99 / 164–176 / 223–380 ms |
| tokens per plan: in / out (reasoning) | 999 / 186 (38) | 999 / 324 (138) | – |
| tokens per correction: in / out (reasoning) | 1,277 / 297 (43) | 1,279 / 260 (85) | ~820 in |
| cost per correction | ~$0.0061 | ~$0.0003 | ~$0.00003 |
| run cost | **$0.98** | **$0.043** | $0.005 per run |

Prompt caching was small: Terra averaged 117 cached input tokens per plan, and Luna had none. Total spend for E13,
including the warm-up and the partial Sol run, was about $1.40.

**Where the LLM was wrong** (3 of 140 escalations in each run, of the same two kinds): "odd ones on the
left/right instead" (parity bins: Terra 1, Luna 2), and "sort / line them up by number instead, lowest
on the left" when the task used a different rule (Terra 2, Luna 1).

**Takeaways**
- **[measured] The fast LLM does not prepare branches.** Every plan was valid and its default right,
  but 1.1–1.3 branches per plan covered 4–7% of the corrections a branch could have covered. Branches
  appeared only where the variable is spelled out in the task (bins by parity; colour to bin). **[inferred]**
  The prompt makes branches optional, and both models take the option. The harness rule forbids
  changing the prompt, so this is the result as specified. The v2 design has since dropped prepared
  variants (`docs/v2.md`, d9539d9), which this supports.
- **[measured] Jev routes corrections reliably and picks branches when they exist:** route 135–138/140,
  branch pick 6/6, and every correction it handled itself was right (58/58). Two chatter messages per
  run were escalated when they didn't need to be; that costs time, not correctness.
- **[measured] Confidence gating was not tested:** Jev made no mistakes on the corrections it kept, so
  every gate from 0.3 to 0.9 gives the same outcome. **[inferred]** Keep 0.7 until a run produces Jev
  errors on kept corrections; E11's replay remains the evidence that low confidence flags them.
- **[measured] The system equals the always-LLM baseline on accuracy (137/140 in both runs)** and wins on
  latency only where Jev keeps the correction: chatter and pause take ~0.1 s against 2–2.5 s. Everything
  else pays Jev's ~100 ms on top of the LLM's round trip.
- **[measured] An LLM round trip per correction is 2.7–4.1 s median and 6–7 s at p95**, with rare outliers
  of 11 s (Terra) and 25 s (Luna), and a 30 s timeout on Sol. Anything that blends LLM decisions into motion
  must let the arm carry on or hold for several seconds, and survive a 25–30 s tail.
- **[measured] Luna matches Terra's accuracy at about 1/20 of the cost** and has a faster median, but a
  heavier tail. It spends more of its output on reasoning (85 vs 43 tokens per correction).

**Limits (from the handoff, unchanged).** Open loop: each correction is judged on its own, from the default
branch, with no arm or simulation. The evaluation oracle is lenient in places (gaps in tray lines, any
valid arrangement for "any order" tasks), so an uncoverable correction can occasionally match by chance.
Block ids and destinations are fixed lists in the output schema, so validity failures can only be missing,
duplicate or clashing blocks. The always-LLM baseline and the system's escalation share one LLM call per
correction, so their accuracy on escalated corrections is identical by construction. The event streams'
times are per correction (t=0 when the user speaks), not wall time.

**Fast-model probe (2026-09-24, 7 tasks × 5 corrections each; small samples).** Same harness and
prompts, looking for a faster LLM; commands and tables in `experiments/results/e13_probe.txt`, runs in
`e13_branches.jsonl`, event streams in `e13_events_<run>.md`. Prices for `gpt-5.4-nano` are approximate.
Jev: 0 errors, p50 84–103 ms throughout.

| model | right: system / always-LLM | LLM per correction p50 / p95 / max | LLM plan p50 / p95 | coverable with a branch | Jev kept (right) | $ per correction |
|---|---|---|---|---|---|---|
| gpt-6-luna, `--llm-effort low` | **35/35 / 35/35** | 2.2 / 5.3 / 9.4 s | 2.3 / 6.5 s | 0/14 | 7 (7) | 0.0003 |
| gpt-5.4-mini | 31/35 / 28/35 | 2.2 / 4.0 / 7.3 s | 1.9 / 7.7 s | 2/14 | 9 (9) | 0.002 |
| gpt-4.1-mini | 26/30 / 27/30 | 3.7 / 5.6 / 8.1 s | 4.3 / 22.6 s (1 timeout) | **10/12** | **18 (17)** | 0.0013 |
| gpt-4o-mini | 24/30 / 24/30 | 2.5 / 6.9 / 30 s (timeout) | 2.5 / 7.4 s | 0/12 | 6 (6) | 0.0003 |
| gpt-5.4-nano | 22/35 / 22/35 | 2.7 / 7.4 / 10.1 s | 1.9 / 4.8 s | 0/14 | 7 (5) | 0.0007 |
| gpt-4.1-nano | 13/35 / 10/35 | 3.0 / 6.6 / 23.9 s | 2.1 / 3.8 s | 2/14 | 11 (7) | 0.0003 |

(30 corrections where a plan failed validation or timed out: its task's corrections are skipped.)

- **[measured] No model gets an LLM round trip much under ~2 s.** Models with no reasoning step and
  Luna at low effort all land at 2.2–3.7 s median, 4–7 s p95, with 10–30 s outliers on four of six.
  **[inferred]** At these prompt sizes (about 1.4k tokens in, 250–450 out), the API round trip and output
  length dominate, not reasoning. Plan motion around 2–4 s typical and a 30 s worst case.
- **[measured] `gpt-6-luna` at low effort is the best default seen**: the only 100% model, as fast as the
  minis, cheapest, and a smaller tail than at default effort (9 s max vs 25 s in the full run).
- **[measured] `gpt-4.1-mini` is the only model that prepares branches unprompted** (10/12 coverable), so
  Jev kept 18/30 corrections at a 148 ms system median. But accuracy is lower (87%) and its plan call
  has a heavy tail (22.6 s p95, one 30 s timeout).
- **[measured] Nano models and `gpt-4o-mini` are too inaccurate** (37–80%), and nanos also hand Jev
  wrong branches to pick from (Jev "right" on only 5/7 and 7/11 of the corrections it kept).

## E12 update (e12v2) — the v2 design closed loop, with controls (2026-09-25)

Run from the bay Mac: live Jev (`jev-1.13.0`) and live LLM (`gpt-6-luna`, `--llm-effort low`, $0.10 / $0.50
per M tokens) on 12 scenarios (9 core, 3 held out) × 7 arms. Runs, all in `experiments/results/`
(`e12v2_run.txt` has every printed table with its command line; the logs are gzipped, so `gunzip -k` one before `--replay`):

| log | what | status |
|---|---|---|
| `e12v2_20260925-082336` | live Jev, mock LLM, seed 12 (the small check) | complete; pre-fix |
| `e12v2_20260925-082511` | live, 3 seeds, all arms | stopped at the 600-LLM-call cap; `jev`, `rules`, `always_llm`, `oracle` complete; pre-fix |
| `e12v2_20260925-090525` | live, `--noise`, 1 seed, all arms | complete; pre-fix |
| `e12v2_20260925-092549` | the three ablations, 3 seeds | stopped by hand when the bug below was found |
| `e12v2_20260925-093744` | the four Jev arms, 3 seeds, after the fix | complete |
| `e12v2_20260925-101602` | the four Jev arms, `--noise`, after the fix | complete |
| `e12v2_combined_3seeds`, `_combined_noise` | post-fix Jev arms + the unaffected `rules` / `always_llm` / `oracle` episodes of the same seeds | **used for every table below** |

Errors: **0** failed Jev requests and **0** failed LLM calls in any run (about 4,400 Jev requests, 2,000
LLM calls in total). Total spend about **$1.00** (Jev ≈ $0.28, LLM ≈ $0.72).

**Harness bug found and fixed (commit `36bf0f4`).** `core/decision.build_state` added
`user_said_earlier` only when there was *more than one* user message. With exactly one (e.g.
`sort_correction`'s "actually, highest on the left"), every later Jev decision saw only the original task
next to a replanned plan that contradicted it. The planner always got every message; only Jev's view was
wrong. It affected the `jev` arm and its ablations (`rules`, `always_llm` and `oracle` don't read it); those
four arms were rerun for 3 seeds and for `--noise`. The fix did **not** change the headline: `jev`
still fails `sort_correction` (below).

**Pass criteria** (3 seeds, combined log):

| # | criterion | result | verdict |
|---|---|---|---|
| 1 | `jev` completes ≥ 90% of core episodes, zero hand contacts | 24/27 (89%), 0 contacts | **FAIL** (all 3 misses are `sort_correction`) |
| 2 | `jev` beats `rules` on held-out completion or correction time | completion 9/9 vs 9/9; correction → behaviour 0.10 s vs 8.40 s | **PASS** |
| 3 | `jev` faster than `always_llm` on correction time, without lower completion | 0.20 s vs 7.70 s; but 24/27 vs 27/27 | **FAIL** (completion) |
| 4 | ≥ 90% of Jev's wrong right-now / in-plan-fix answers caught at gate 0.7 | 42/47 (89%) | **FAIL** (by one answer) |
| 5 | held-out completion (no threshold) | `jev` 9/9; every arm 9/9 | reported |

With `--noise` (1 seed): 1 fails (8/9, `sort_correction`), 2 passes (held out 3/3 vs rules 2/3; 0.10 s vs
10.7 s), 3 fails (8/9 vs 9/9), 4 passes (9/10).

**Core scenarios, 3 seeds**

| arm | done | LLM calls | LLM wait s | hold s | contacts | correction → behaviour s (med) | chatter needlessly escalated | Jev p50 / p95 ms | Jev agrees with oracle: right-now / fix / route |
|---|---|---|---|---|---|---|---|---|---|
| jev | 24/27 | 96 | 414 | 53 | 0 | **0.20** | 0 | 93 / 168 | 303/331 / 218/223 / 290/331 |
| rules | **27/27** | 35 | 179 | 42 | 0 | 0.10 | 0 | – | – |
| always_llm | 27/27 | 93 | 403 | 25 | **3** | 7.70 | 3 | – | – |
| oracle | 26/27 | 36 | 170 | 50 | 0 | 0.20 | 0 | – | – |
| jev-no-right-now | 24/27 | 108 | 426 | 0 | **3** | 15.75 (1 missed) | 0 | 96 / 154 | – / 223/230 / 264/312 |
| jev-no-in-plan-fix | 24/27 | 103 | 373 | 39 | 0 | 0.10 | 0 | 93 / 160 | 307/340 / – / 289/340 |
| jev-no-router | 26/27 | 78 | 275 | 35 | 0 | 0.10 | 3 | 85 / 138 | 259/282 / 207/212 / – |

Per scenario, `jev` completes 3/3 everywhere except **`sort_correction` 0/3** (48 LLM calls: every episode
gives up at the 16-call cap). `rules` and `always_llm` complete it 3/3; `jev-no-router` 3/3; `oracle` 2/3
(one invalid LLM replan).

**Held-out scenarios, 3 seeds** (never tuned on)

| arm | done | correction → behaviour s (med) | LLM calls | notes |
|---|---|---|---|---|
| jev | 9/9 | **0.10** | 49 | `tower` used 32 LLM calls over 3 seeds |
| rules | 9/9 | 8.40 | 18 | missed "Hmm, yellow in the middle please…" (no keyword) and waited for the LLM |
| always_llm | 9/9 | 6.70 | 61 | `handover` 38 LLM calls |
| jev-no-right-now | 9/9 | 9.40 | 56 | |
| jev-no-in-plan-fix / jev-no-router | 9/9 each | 0.10 | 55 / 56 | no-router: `handover` 41 LLM calls |

**Gate sweep** (right-now + in-plan-fix answers, 3 seeds): 706 answers, 47 wrong.

| gate | wrong caught | escalated or fell back | accuracy of acted-on answers |
|---|---|---|---|
| 0.5 | 22/47 (47%) | 7.1% | 96.2% |
| 0.6 | 30/47 (64%) | 10.6% | 97.3% |
| **0.7** | 42/47 (89%) | 15.4% | 99.2% |
| **0.8** | **46/47 (98%)** | 20.7% | 99.8% |
| 0.9 | 47/47 | 32.2% | 100% |

**LLM calls** (all arms, 3 seeds; `gpt-6-luna`, low effort)

| call | n | p50 / p95 / max | input tokens (cached) | output (reasoning) |
|---|---|---|---|---|
| plan | 252 | 4.3 / 6.2 / 9.3 s | 2,205 (99% cached) | 409 (31) |
| replan | 496 | 3.0 / 7.3 / 24.4 s | 2,901 (59%) | 266 (138) |
| react (`always_llm`) | 111 | 3.4 / 6.6 / 8.9 s | 2,734 (59%) | 233 (130) |

Cost per arm over the 36 episodes: `jev` LLM $0.062 + Jev $0.032; `always_llm` $0.063; `rules` $0.023.

**What went wrong: `sort_correction` (read from the log).** After "actually, highest on the left" the LLM
replans correctly (a plan reading "highest to lowest… 12, 11, 7, 5, 3, 1"). From then on, Jev's state
shows `task: "…lowest number on the left"` and, after the fix, `user_said_earlier: ["actually, highest on
the left"]`. Jev does not reconcile the two: on each new plan its `route` is `stay_local` at 0.29–0.66
(below the gate, so it escalates), and after finished steps it routes `fast_llm` at 0.76–0.82 (the plan
"contradicts the task"). Each replan produces another `plan_arrived`, and the loop detector doesn't cover
replan loops, so the episode hits the 16-LLM-call cap. `jev-no-router` (route = stay local unless there
is a fix) avoids it and completes 3/3.

**Takeaways**
- **[measured] Right-now works, and it earns its place.** 0 hand contacts in every Jev arm that has it;
  `always_llm` and `jev-no-right-now` touched the hand in 3/3 `sort_hand_in_path` runs. Corrections change
  behaviour in 0.1–0.2 s against 7.7 s (`always_llm`) and 15.8 s (`jev-no-right-now`).
- **[measured] Jev beats keyword rules only where the phrasing is new.** On the held-out tower ("Hmm,
  yellow in the middle please, red goes up top.") the rules have no keyword and wait for the LLM: 8.4 s
  against 0.1 s. On the core scenarios, written against the rules' phrasings, the rules complete 27/27
  with a third of Jev's LLM calls (35 vs 96).
- **[measured] The router is the weak group.** Route agreement with the oracle is 290/331 (88%), against
  303/331 (92%) for right-now and 218/223 (98%) for in-plan fix. All three `jev` failures are route
  loops, and removing the router (`jev-no-router`) raises completion to 26/27. Its one failure is a
  `sort_hand_in_path` run, and it escalates chatter needlessly (3 times).
- **[inferred] The cause is the state, not the gate:** Jev is asked to judge a plan against the
  original task plus a later correction, a two-step inference (the docs' "indirection"). A single
  "task as corrected" field (the planner's `reading` already is one) would likely fix it. And a replan
  loop needs the loop detector.
- **[measured] The in-plan fix adds little here.** Without it, completion is the same (24/27), with 7% more
  LLM calls (103 vs 96).
- **[measured] Set the gate at 0.8, not 0.7.** 0.7 catches 89% of wrong answers (misses criterion 4 by
  one); 0.8 catches 98% for 5 points more escalation (20.7% vs 15.4%).
- **[measured] LLM latency matches E13:** replans 3.0 s median, 7.3 s p95, one 24 s outlier. Prompt caching
  works: 99% of plan input and 59% of replan input tokens were served from the cache.
- **[measured] With perception noise (1 seed) nothing new broke:** the same `sort_correction` failure,
  and 0 contacts for `jev`.

**Limits (from the handoff).** The world, skills and person are simulated in text; perception is ground
truth unless `--noise`. The rules control is fair but written knowing the core phrasings, so criterion 2
is judged on the held-out scenarios. Seeds vary little without `--noise`. The combined logs mix post-fix
Jev arms with pre-fix control arms of the same seeds; the controls don't read the fixed field.

## Getting more out of Jev: recommendations from the TypeSafe docs (inferred, none applied)

Read after E11/E12: the [Jev 1.13 jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
(math and numbers), the [cookbooks](https://docs.typesafe.ai/cookbooks.md), and the pattern pages
(fan-out, confidence routing, state, advanced structure). Each item below is a pattern from those docs,
tied to a number from these runs. None were applied, so the results above are unchanged. `docs/v2.md`
(commit `bed7a7b`, written in parallel with these runs) already adopts items 1, 7 and 8 and the
per-route half of 5. Items 3, 4, 6 and the `when` gate are new from the data.

1. **Keep arithmetic and ranking in code; ask Jev only for the mapping** (jaggedness: "does not count
   reliably", "no arithmetic guarantees between separate questions"; date-extraction cookbook: extract
   components with Choice, do the arithmetic in code). E11 `solved` is this pattern and it scores
   301/303. Go further: once `new_order` is known (720/720), `direction`, `target_slot` and `next_pick`
   are code lookups, so don't ask them at all. Where both are asked, derive `direction` from
   `target_slot` (it contradicted its own slot in 24/103 `raw` answers).
2. **Name positions by their meaning, not by numbers** (jaggedness: numeric encodings degrade; use
   names). Sparse numbers halved `raw` `target_slot` accuracy (26/48 vs 53/55). Label slots and blocks
   by role in the state ("the leftmost free slot", "the held block's goal") and present candidate
   actions as options whose text carries the consequence, as in the pre-parsed-value cookbook: code
   generates the candidates, Jev picks one verbatim.
3. **Give the state the fact that matters, stated literally** (jaggedness: literal reading,
   indirection; state guide: present it "to a panel of experts"). The whole `raw` failure is one
   missing fact: the gripper is low *at the held block's destination*, and the block has been placed
   but not released. Say that directly (`gripper_is_above: "tray slot 1 (lowered, holding block 1)"`),
   and keep a short "done so far for this block" list, so `retreat` doesn't erase the history.
4. **Route on the weakest confidence, not on an `escalate` question** (function-calling cookbook:
   report "the weakest judgment"; confidence-routing pattern). `escalate` scored AUROC 0.43 (E11 `raw`)
   and 0.36 (`solved`). Min-confidence scored 0.71 and 0.93, and E12 `raw` next_skill confidence 0.96.
   This is the Router's input.
5. **Set thresholds per question, from option count and stakes** (E4: Choice confidence is
   `(p_max − 1/n) / (1 − 1/n)`; confidence-routing: a floor, then action-specific bands). E12 used a flat
   0.3 for 3-option and 8-option questions and **no gate on `when`**, which applied a 0.06 answer.
   At 0.5, E12 `raw` would have blocked 73% of the wrong `retreat` picks and kept 100% of right picks.
   A low-confidence `when` should default to the cautious value in code.
6. **Draw option boundaries with `not_for` and examples** (advanced structure: structured Choice
   criteria with `what` / `not_for` / examples). `retreat` ("lift the gripper back up") competes with
   `release` exactly where it shouldn't. Add `retreat.not_for: "while holding a block that is lowered
   at its destination: release it first"`. The same applies to `not_for_me` vs `new_task` (the one E11
   intent miss).
7. **Fan out speculatively, one request per event** (fan-out pattern: send every question that might
   matter, let code pick; answers are independent and parallel). This is already v2's "Sequencer and
   Router in one request". Extend it: on a correction, ask `new_order` plus a Noul per parameter ("does
   the user say anything about pace?"), so code knows what was not mentioned. The function-calling
   cookbook's "stated" question does exactly this.
8. **Verify an LLM's plan patch with Jev before running it** (SDE-cascade cookbook: a cheap model
   extracts, Jev asks narrow yes/no checks per field, and any flag above a threshold escalates to the
   strong model). This fits v2's fast-LLM route: the fast model patches the plan, Jev checks each step
   with a Noul ("does this step put block 5 in the user's order?"), and a flag goes to the capable LLM.
9. **Abstain rather than repeat** (self-consistency cookbooks: abstain below 0.6; 90.8% raw
   agreement → 99.2% policy agreement). The abstaining is what helps, and v2's "below a floor the
   answer is uncertain and goes to the Router" already does it. Re-asking the same request adds
   little: E5 showed identical requests return the same choice, and E12's release/retreat error was
   stable over 7,162 re-asks.
10. **Rank then verify for a large skill list** (skill-suggestion cookbook: wide Choice, then a short
    list with fuller text and a Noul per candidate). This is not needed at E12's 3–8 options. It will
    be needed when VLA skills and a larger world make the Sequencer's option list long.
11. **Learn thresholds from the logs** (composite scoring / autoresearch cookbooks: train a small
    classifier on Jev's probabilities). E11 and E12 already log every probability with its truth. A
    logistic model over the per-question confidences could be the Router's gate and would be cheap to
    refit whenever a prompt changes.

## Not yet run
- Options labelled with descriptive text vs letter IDs.
- Two-stage call (goal, then motion) vs a single call using the previous tick's goal.
- A Jev vs LLM baseline via `system-one-adapter-python`.
- Scores spread across non-adjacent levels, to pin down the Score confidence formula.
- Behaviour on scenes rendered from the real sim.
