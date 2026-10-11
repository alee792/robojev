# Live models in the slice (2026-10-11)

The L1 slice (tests/test_slice.py: the brain on the MuJoCo WidowX, realtime) with a real decision
model and a real planner in place of e12v2's mocks, plus a replay of e12v2's logged decision requests
to both decision models. Everything else is unchanged: the same request building
(core/decision.py), combine() and its gates, the brain's rules and the physics server.

| | |
|---|---|
| Deciders | Jev `jev-1.13.0` (typesafe /v1/systemone), OpenAI Decisions API `gpt-6-luna` (/v1/decisions) |
| Planner | `gpt-6-luna` on the Responses API, strict JSON schema, effort low, `prompt_cache_key` |
| Code | `src/robojev/brain/backends/{jev,decisions_api,openai_planner}.py`; tests `tests/test_backends.py` (mocked HTTP) |
| Switch | `ROBOJEV_DECIDER=mock\|jev\|decisions`, `ROBOJEV_PLANNER=mock\|openai`, `ROBOJEV_SHADOW=1` (tests/live_models.py) |
| Logs | `experiments/results/live/`: `slice_*.jsonl.gz` (one per run: the server's frames and events, then every decision, plan, brain log line, and every decision request with the other backend's answer and the oracle's truth), `pytest_slice_runs.txt`, `slice_report.md`, `replay_*.jsonl` + `replay_report.txt`, `v1_*` |
| Reproduce | `ROBOJEV_DECIDER=jev ROBOJEV_PLANNER=openai ROBOJEV_SHADOW=1 uv run --frozen python -m pytest -s tests/test_slice.py`; `uv run --frozen python experiments/live_slice_report.py`; `uv run --frozen python experiments/live_decisions_replay.py --n 600 --plants 60` |

**The Decisions API in one request.** It takes an array of named questions, so all three groups
(right_now, in_plan_fix, route) and any progress and plan-check Nouls go in one call, as they do to
Jev. Choice → `choice` (criteria `what` + "Not for: `not_for`" as each option's description), Noul →
`predicate`, state → one `input_text` part (JSON), frames → `input_image` (inline base64). Answers come
back in Jev's shape; a `refusal` answer is dropped (read by combine() as confidence 0). None were seen
in about 850 requests.

**Confidence.** The API's own `confidence` field is Jev's formula `(p_max − 1/n)/(1 − 1/n)` rounded
to two places: over 1,371 choice answers the largest difference is 0.005. Both are exposed
(`confidence_api`, `confidence_formula`); the backend gives combine() the formula. The gate behaves
the same with either (table below).

A fifth scenario was added to the slice: **(e)** "hang on a second" typed mid-carry, then "ok, go on"
2 s later. The arm must pause within 0.5 s and resume, and no restated task may mention the wait (H5).

## 1. The slice, live (3 runs per scenario and decider)

| scenario | mock (l1_slice.txt) | Jev + gpt-6-luna | Decisions API + gpt-6-luna |
|---|---|---|---|
| (a) 3-block sort | pass, 22.9 s | **3/3**, 25.9–27.1 s | **3/3**, 25.8–26.4 s |
| (b) correction mid-carry | pass, hold 0.100–0.105 s | **0/3**: never holds (Jev: `carry_on`, conf 0.83, every run); end order right, 34.7–36.0 s | **2/3**: hold 0.156, 0.121 s; one run **1.115 s** (that decision took 1.1 s); 31.5–34.7 s |
| (c) hand 0.15 m mid-carry | pass, pause 0.097–0.105 s | **3/3**, pause 0.091–0.097 s, resume 0.088–0.118 s, closest 0.122 m | **3/3**, pause 0.146–0.150 s, resume 0.094–0.144 s; 10 replans over 3 runs (Jev: 1) |
| (d) STOP mid-carry | pass | **3/3**, parked in 2.45–2.53 s, block held | **3/3**, parked in 2.45–2.57 s, block held |
| (e) wait, then go on | pass, pause 0.109 s | **3/3**, pause 0.093–0.141 s, resume 0.080–0.160 s, no leak | **2/3**: pause 0.101–0.213 s, resume 0.103–0.154 s; **run 1 leaked** the wait into the task |

What the failures were:

- **Jev on the correction.** Asked about "actually, highest on the left" while carrying block 1 to
  slot 0, Jev answered `carry_on` (p 0.89, conf 0.83) and `stay_local` (0.65), the same in all three
  runs. The right-now answer cleared its 0.5 gate, so the arm carried on and put block 1 in slot 0.
  The route answer fell under 0.8, so a replan went out; it came back 4–6 s later and the arm moved
  block 1 again (4 steps instead of 3). The end state was right, but the hold the user is owed never
  came. The Decisions API answered `hold` (0.66) on the same request each time.
- **The Decisions API's 1.1 s hold.** One right-now decision took 1,097 ms (the slowest of 84 in the
  loop). The hold arrived after the 0.5 s bar.
- **The Decisions API's leak (H5).** Its route confidence on "hang on a second" was low (0.48), so the
  wait went to the planner. The plan that came back read "…after the user said 'hang on a second' and
  then 'ok, go on,' continue the arrangement." Jev kept both lines local in all three runs.

**Timing against the mock.** Hand to pause: Jev 0.091–0.097 s (mock 0.097–0.105 s), Decisions API
0.146–0.150 s. Correction to hold: Jev none (it never held), Decisions API 0.121–0.156 s when its
decision was quick (mock 0.100–0.105 s).

**Decision latency in the loop** (every decision the brain acted on; the other backend answered the
same request in the background):

| | n | p50 | p95 | max |
|---|---|---|---|---|
| Jev, acting | 58 | 102 ms | 130 ms | 146 ms |
| Jev, shadow | 84 | 102 ms | 141 ms | |
| Decisions API, acting | 84 | 129 ms | 254 ms | 1,097 ms |
| Decisions API, shadow | 58 | 128 ms | 448 ms | |

**Planner (gpt-6-luna, effort low).** First plan p50 2.78 s, p95 3.7 s (Jev runs; 2.74 / 3.4 s in
the Decisions API runs). Replans p50 3.9 s, p95 6.4 s, max 6.5 s (Jev runs, n 8); p50 3.8 s, p95
5.6 s (Decisions API runs, n 24, three times as many). One plan needed the validator's retry; none
failed. The mock planner took ~0.3 s, so a live correction holds the arm about 4 s, not 0.3 s.
Prompt-cache hits are not in these logs: the brain's planner doesn't record `cached_tok`.

## 2. Decisions API vs Jev on the same events

**Replay.** 600 decision requests from e12v2's live Jev runs (core and held-out scenarios; every
user text, hand, failure and plan-check request in the sample, and routine ones to fill the rest),
plus 60 planted bad plans. Each request was sent unchanged to both backends, 4 at a time. Truth is
e12v2's oracle, logged with each request.

| | errors | p50 | p95 | max | server-side p50 / p95 |
|---|---|---|---|---|---|
| Jev | 0/660 | 101 ms | 149 ms | 304 ms | not reported |
| Decisions API | 0/660 | 143 ms | 395 ms | 1,130 ms | 69 / 153 ms (`openai-processing-ms`) |

| group | n | agree | Jev right | Decisions API right | Jev in September, same requests |
|---|---|---|---|---|---|
| right_now | 552 | 83.5% | 87.1% | 89.5% | 87.3% |
| in_plan_fix | 258 | 93.8% | 95.0% | 96.5% | 95.0% |
| route | 561 | 85.2% | 89.1% | 88.2% | 87.9% |

On the slice's own events (live, acting + shadow, n 142): agreement right_now 96%, in_plan_fix 99%,
route 93%; right against the oracle (n 104 with truth): Jev 95 / 99 / 94%, Decisions API 99 / 100 / 94%.

**The gate** (e12v2's `gate_sweep`: wrong right-now answers, and wrong fixes when routed stay_local,
caught by falling under the gate; escalated = every answer under the gate):

| gate | Jev: caught | Jev: escalated | Decisions API: caught | Decisions API: escalated | Decisions API, its own `confidence`: caught / escalated |
|---|---|---|---|---|---|
| 0.5 | 56/93 (60%) | 12.0% | 51/64 (80%) | 23.5% | 80% / 23.4% |
| 0.7 | 81/93 (87%) | 23.2% | 59/64 (92%) | 36.0% | 92% / 36.0% |
| **0.8** | **88/93 (95%)** | **30.9%** | **63/64 (98%)** | **45.4%** | 98% / 45.2% |
| 0.9 | 92/93 (99%) | 46.5% | 64/64 (100%) | 58.7% | 100% / 58.6% |

At 0.8 both catch nearly all their wrong answers (accuracy of what is acted on: Jev 99.1%, Decisions
API 99.8%). The Decisions API makes fewer wrong answers but is less sure of its right ones: it
escalates half again as often (45% vs 31%). This replay is on e12v2's toy world; the slice shows the
same pattern on physics (Decisions API: 24 replans in 15 episodes, Jev 8).

## 3. H4 and H5

**H4: the plan check still sends good plans back, at 49% with Jev and 66% with the Decisions API.**
Counted the way e12v2 counted its 39%: a new plan the oracle says to keep, sent back to the LLM by
combine() at the default gates.

| | good new plans | sent back | unsure stay_local (< 0.8) | confident replan | plan-check Noul | planted bad plans caught |
|---|---|---|---|---|---|---|
| Jev, September answers | 185 | 91 (49%) | 61 | 30 | 0 | |
| Jev, now | 185 | 91 (49%) | 67 | 24 | 0 | 60/60 |
| Decisions API | 185 | 123 (66%) | 92 | 31 | 0 | 59/60 |

The plan-check Nouls never flag a good plan (0/165 for both). Alone they catch 59/60 (Jev) and 47/60
(Decisions API) of the planted plans. The send-backs come from the route question on `plan_arrived`.
The router is unsure that a fresh plan "still holds as written", or confidently asks for a replan.
The 39% → 49% difference is the sample (this one is weighted to plan-check requests), not drift: Jev's
September answers to the same requests give the same 49%. In the live slice, Jev sent back 0 of 20 new
plans and the Decisions API 8 of 30. H4's bar (under 10%) is not met by either. The fix is in what the
router is asked on `plan_arrived`, not in the plan check.

**H5: a transient wait still leaks, when it reaches the planner.** Live slice: Jev 0/3 (both lines
stayed local, no replan), Decisions API 1/3 (an unsure route sent the wait to the planner, and its
restatement kept it). The September logs show the same mechanism: 15 restatements from the full Jev
arm mention a wait (14 in handover, 1 in sort_wait_go), every one after a replan. Nothing in the
planner prompt says a restatement holds only the end result.

## 4. V1 preview: one Decisions API call per scene change, with a frame

`experiments/v1_frame_preview.py`: mid-carry, the overhead camera rendered on the Mac (512×384 PNG),
one request per frame with the hand facts removed from the state ("judge from the camera image"),
against the same request text-only. **The MuJoCo scene has no person**, so the hand is drawn into the
frame only: a skin-coloured forearm capsule and palm box (see `results/live/v1_*.png`).

| | hand near (≤ 0.20 m) judged right | latency p50 / max | input tokens |
|---|---|---|---|
| frame attached | 4/12 (all 6 near frames read "no hand", p ≤ 0.04) | 150 / 195 ms | 954 |
| text only (hand bands in the state) | 9/12 (the 0.40 m "mid-range" hand sat at p 0.50) | 129 / 249 ms | ~726 |

A frame costs about +230 tokens and +20 ms. The accuracy number is not evidence about the API: the
prop is a pink stick that does not look like a hand, and near the gripper the palm washes out against
the lit table. V1 needs real frames: a photographed hand on the bay's camera, or a hand mesh in the
twin. The latency and token cost are usable; the accuracy is not.

## Fixed along the way

- Harness, not brain: `tests/live_models.py` first passed the oracle a world state instead of its
  truth source, so Jev runs 1–2 have no truth labels (commit ccda8f5). No brain rule or gate was
  changed.
- The two mock-shaped assertions in the slice (exactly three `pick_and_place` starts; the oracle's
  first two starts in (b)) and "the correction decision is `hold`" are recorded rather than asserted
  when live. Scenarios are judged on world(), the 0.5 s bars, drops and STOP as before.
