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

## Not yet run
- Options labelled with descriptive text vs letter IDs.
- Two-stage call (goal, then motion) vs a single call using the previous tick's goal.
- A Jev vs LLM baseline via `system-one-adapter-python`.
- Scores spread across non-adjacent levels, to pin down the Score confidence formula.
- Behaviour on scenes rendered from the real sim.
