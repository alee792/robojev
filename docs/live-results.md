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
| Switch | `ROBOJEV_DECIDER=mock\|live\|jev\|decisions`, `ROBOJEV_PLANNER=mock\|openai`, `ROBOJEV_SHADOW=1` (tests/live_models.py). **Default decider: Jev** (`live` = `jev`, per the recommendation below); the Decisions API is selected with `decisions`; the test run itself defaults to `mock` and needs no keys |
| Logs | `experiments/results/live/`: `slice_*.jsonl.gz` (one per run: the server's frames and events, then every decision, plan, brain log line, and every decision request with the other backend's answer and the oracle's truth), `pytest_slice_runs.txt`, `slice_report.md`, `replay_*.jsonl` + `replay_report.txt`, `v1_*` |
| Reproduce | `ROBOJEV_DECIDER=jev ROBOJEV_PLANNER=openai ROBOJEV_SHADOW=1 uv run --frozen python -m pytest -s tests/test_slice.py`; `uv run --frozen python experiments/live_slice_report.py`; `uv run --frozen python experiments/live_decisions_replay.py --n 600 --plants 60`; offline, no network: `uv run --frozen python experiments/replay_plan_route.py` (the section "After the fixes") |

**The Decisions API in one request.** It takes an array of named questions, so all three groups
(right_now, in_plan_fix, route) and any progress and plan-check Nouls go in one call, as they do to
Jev. Choice → `choice` (criteria `what` + "Not for: `not_for`" as each option's description), Noul →
`predicate`, state → one `input_text` part (JSON), frames → `input_image` (inline base64). Answers come
back in Jev's shape; a `refusal` answer is dropped (read by combine() as confidence 0). None were seen
in about 850 requests.

**Confidence.** The API's own `confidence` field is Jev's formula `(p_max − 1/n)/(1 − 1/n)` rounded
to two places: over 1,371 choice answers the largest difference is 0.005. Both are exposed
(`confidence_api`, `confidence_formula`). **The gate should use the API's field.** It is the same
number, so the gates set on Jev apply unchanged. But computed from the two-decimal probabilities, the
formula comes out 0.7999999999999999 where the API says 0.80, and that put 13 of 1,520 replayed
answers just under the 0.8 or 0.5 gate. The backend now defaults to the API field and rounds the
formula (commit 2814046; the runs below used the unrounded formula; no slice decision fell
on a gate edge).

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
  slot 0, Jev answered `carry_on` (p 0.89, conf 0.83 in run 1) and `stay_local` (0.65), the same in all
  three runs. The right-now answer cleared its 0.5 gate, so the arm carried on and put block 1 in
  slot 0. The route answer fell under 0.8, so a replan went out; it came back 4–6 s later and the arm
  moved block 1 again (4 steps instead of 3). The end state was right, but the hold the user is owed
  never came. The Decisions API answered `hold` (0.66) on the same request each time.
  **Why: the restated task.** The decision's `task` is the planner's restatement
  (core/decision.py, by design). Here it read "Place block_1, block_3, and block_5 in the tray from
  left to right in ascending numeric order". Jev does not read "highest on the left" as contradicting
  that. The same request, 5 calls per variant:

  | state | Jev right_now | Decisions API right_now |
  |---|---|---|
  | as recorded | carry_on ×5 (0.56–0.67) | hold ×5 (0.86) |
  | `task` = the user's "Line the blocks up in the tray, lowest number on the left." | hold ×5 (0.45–0.61) | hold ×5 (0.97) |
  | `arm_in_step` as e12v2's literal phase ("holding block 1 at travel height, carrying it to…") | carry_on ×5 | hold ×5 (0.28) |
  | both | hold ×5 (0.43–0.66) | hold ×5 (0.85) |

  In e12v2's replay Jev also leaned `carry_on` on 6 of 14 corrections, but always under 0.5, so the
  cautious fallback held. In the slice the restatement made it confident. This is not a code bug, and
  nothing was changed. It is the restatement (H5's subject too) hiding the words a correction
  contradicts.
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

On the 53 user-text requests alone, right-now right: Jev 45, Decisions API 51.

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

## After the fixes (offline replay)

Three design changes followed from sections 1 and 3 (the code is in `src/robojev/brain/`; the tests in
`tests/test_brain_decision.py`; nothing in the backends' HTTP behaviour changed):

- **P1, the correction.** The decision request now carries `user_words` (the task the user typed and
  every correction the brain took as about the end result, verbatim, newest last; a transient "wait"
  or "go on" never joins it) next to `task` (the plan's reading, kept: it stops the old replan loop).
  The right-now and route questions say a correction is judged against `user_words`, "what the user
  asked for, in their words", not against `task`, "how the current plan reads it". The planner is told
  the restatement keeps the user's own words for the end result ("lowest number on the left", not
  "ascending numeric order"). **Pending a live re-run (Handoff 7):** this needs Jev; offline, the unit
  tests cover the field (present, verbatim, newest last, transients excluded) and the question text.
- **P2, good plans sent back.** On `plan_arrived` the route is decided by the plan check alone: a
  flagged object sends the plan to the fast LLM with the problems attached, otherwise it stays local;
  the route and in-plan-fix questions are no longer asked for that event and their answers, if present,
  are ignored (core/combine.py has the reasoning). The 3-rejections replan-loop limit stays as the
  backstop. `experiments/replay_plan_route.py` re-runs the recorded requests and both deciders'
  recorded answers (the replay above, and every plan_arrived of the live slice, acting and shadow)
  through the brain's combine() as it is now, against a replica of the old rule that reproduces the
  numbers in section 3 exactly (91/185, 123/185, 60/60, 59/60):

  | source | decider | good new plans | sent back before | sent back after | planted bad plans | caught before | caught after |
  |---|---|---|---|---|---|---|---|
  | replay | Jev, September answers | 185 | 91 (49%) | **0 (0%)** | | | |
  | replay | Jev | 185 | 91 (49%) | **0 (0%)** | 60 | 60 | **59** |
  | replay | Decisions API | 185 | 123 (66%) | **0 (0%)** | 60 | 59 | **47** |
  | slice (acting + shadow, with truth) | Jev | 40 | 9 (22%) | **0 (0%)** | | | |
  | slice (acting + shadow, with truth) | Decisions API | 40 | 8 (20%) | **0 (0%)** | | | |

  H4's bar (under 10%) is met by both. Planted bad plans caught stays 59/60 with Jev (the one miss is
  a check Noul at 0.12, which the route question had caught at stay_local 0.64, i.e. by the gate).
  **With the Decisions API it falls to 47/60**: its 13 misses are confident "no" predicates (p ≤ 0.19)
  on the plan check, and the old rule caught 12 of them only through low-confidence route answers
  (fast_llm at 0.27–0.75, or an unsure stay_local) that are the same answers that sent 123 good plans
  back. No gate on the route answer separates them: counting a replan answer at confidence ≥ 0.5
  recovers 3 planted plans for 9 good plans sent back (5%); at ≥ 0.3 (the fast_llm gate) 9 for 28
  (15%). So the ≥ 59/60 bar holds for the default decider only; for the Decisions API the plan check
  question itself is the weak part (47/60 alone, section 3), which is a prompt question for the plan
  check, not a routing rule.
- **P3, the leaked wait.** The planner instructions and the `reading` schema descriptions say the
  restatement holds ONLY the end result, never a pause, wait, resume or acknowledgement. In code, once
  the user has typed a transient line (rules.is_transient: wait, hang on, hold on, one sec, pause, go
  on, resume, ...), a plan or replan whose reading quotes it or uses any transient phrase
  (rules.transient_mentions) keeps the previous reading instead, and the log records why
  (`reading_kept`); tested with the mock planner forced to leak the slice's exact wording. Run over
  every plan_arrived in the September logs (all arms, 3,051 readings): the 20 readings that restate a
  wait after one was typed are all caught (the planner paraphrases: "hang on a sec" came back as
  "wait a moment as requested" and "pause the handover", so matching only the user's own phrase caught
  12); 18 more are flagged, every one carrying the go-on ("Continue …", "Resume …", "…; continue placing
  blocks 7, 11 and 12 …": a resume is not the end result either); no reading is flagged in an episode
  without a transient line. The
  slice's run (e) leak is caught. The cost: a replan that leaks and also folds in a correction keeps
  the previous reading until the next clean plan (its steps are installed either way).

## Fixed along the way

- Harness, not brain: `tests/live_models.py` first passed the oracle a world state instead of its
  truth source, so Jev runs 1–2 have no truth labels (commit ccda8f5). No brain rule or gate was
  changed.
- The two mock-shaped assertions in the slice (exactly three `pick_and_place` starts; the oracle's
  first two starts in (b)) and "the correction decision is `hold`" are recorded rather than asserted
  when live. Scenarios are judged on world(), the 0.5 s bars, drops and STOP as before.

## Live re-run after the fixes (2026-10-12/13)

The spelling battery live with both deciders and both controls on the real planner (gpt-6-luna),
then the slice's (b) and (e) five times per decider and the full five. Brain rules unchanged; one
harness fix (below). Files: `experiments/results/battery/20261012-1*-sim-{jev,decisions,rules,always_llm}/`
(`live_tables_20261012-160448.txt` holds the four tables as printed), `experiments/results/live/slice_*2026101[23]-*.jsonl.gz`,
`pytest_slice_runs_20261012.txt`; read with `experiments/live_battery_report.py` and
`experiments/live_slice_rerun_summary.py`.

**Harness fix first (b6eaebc).** `--paraphrase` rewrites every line with a fresh model call, so the
four commands as written (same seed, each with `--paraphrase`) would have run four different
wordings. Every run now writes `episodes.json`; `--episodes FILE` runs those again. The Jev run
paraphrased; the other three ran its `episodes.json`, so all four tables are the same ten episodes,
word for word.

### 1. The battery, side by side (sim, seed 1, 10 episodes, held-out phrasings paraphrased)

| | Jev | Decisions API | rules | always_llm |
|---|---|---|---|---|
| mean outcome score (9 judged; 1 has a STOP) | 0.907 | 0.907 | **1.000** | **1.000** |
| finish rate (exact row, done) | 0.89 | 0.89 | 1.00 | 1.00 |
| outcomes | done 8, stopped 1, timeout 1 | done 8, stopped 1, timeout 1 | done 9, stopped 1 | done 9, stopped 1 |
| median s to first change: new_word (n 3) | 0.13 | 0.17 | 0.00 | 3.90 |
| wait / go-on (n 1) | 0.07 / 0.14 | 0.13 / 0.17 | 0.00 / 0.00 | 3.80 / 1.78 |
| hand_near (n 1) | 0.12 | 0.07 | 0.01 | 4.74 |
| letter_taken / added / nudged / placed (n 1-2 each) | 1.30 / 3.38 / 4.26 / 0.65 | 1.26 / 3.35 / 4.29 / 0.90 | 1.15 / 3.03 / 3.21 / 1.15 | 1.15 / 3.05 / 5.27 / 1.16 |
| planner calls / episode | 8.10 | 6.50 | 1.80 | 2.60 |
| decisions / episode, latency p50 | 15.3, 152 ms | 16.2, 152 ms | 9.4, 1 ms | 10.3, 1 ms |
| decision latency p95 / max | 250 / 441 ms | **2,649 / 4,757 ms** | | |
| drops / episode | 0 | 0 | 0 | 0 |
| hand contacts / episodes with a hand | 0 / 1 | 0 / 1 | 0 / 1 | **1 / 1** |
| STOP ok | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| median wall s | 63.2 | 62.1 | 53.1 | 54.6 |

Ten episodes carry one hand, one wait, one STOP and three new words; every per-kind median is over
1-3 events, and the finish-rate gap is one episode, the same one for both live deciders. What the
table says: on this sample the live deciders add nothing to the outcome over the code rules and
cost 4x the planner calls; they are 0.1-0.2 s on the fast reactions where always_llm takes 2-5 s and
touches the hand. The rules control met held-out paraphrases here and still spelled every word.

### 2. The episode no live decider finished: ep 5, a hand mid-carry (both deciders, the same way)

The episode: OPENAI; at 15 s a hand comes 15 cm from the gripper for 1.5 s, with the remark "Sorry,
just reaching over." Both arms paused within 0.12 s (Jev) / 0.16 s (Decisions API) and the hand left
at 16.7 / 16.3 s. Then neither resumed. Every 5 s a `paused_idle` event re-asked; both deciders
answered `resume` every time, under the 0.5 right-now gate (Jev 0.19-0.37 on all 41 asks; Decisions
API 0.17 on 42 of 43, the first a `hold` at 0.44), and `combine.CAUTIOUS` maps a low-confidence `resume` to `carry_on`, which in a pause
means stay paused (`RIGHT_NOW_CRITERIA["carry_on"]`). The route answer was `stay_local` (Jev 0.64-0.97;
the API 0.84 after its first ask), so nothing escalated and nothing asked the user. 225 s later the episode
timed out with "O" placed. With Jev the pause reason the decider saw was "the user said 'Sorry, just
reaching over.'" (the user_text decision was applied before the hand's, `loop._right_now`), so it was
told the pause was for something the user said, and the user never said go on; with the Decisions
API it was "a person's hand came near". Rules and always_llm finished the same episode (rules' hand
rule is code; always_llm never paused for the hand at all, which is its contact).

This is a liveness gap in the brain's rules, not a model failure: a pause whose cause has gone can
last forever when the decider is unsure, and the paused_idle re-ask only repeats the question. Not
changed here (a rule change; the task says not to tune rules to the numbers). Candidate fixes: after
N identical sub-gate `resume` answers on paused_idle, ask the user or route to the planner; or take
`resume` as the fallback once no hand is in view and no wait is outstanding.

### 3. (b), the correction mid-carry: Jev now holds

| | runs | right-now on "actually, highest on the left" | text to hold | final order | drops | steps |
|---|---|---|---|---|---|---|
| Jev, before | 3 | carry_on (0.83), 0/3 held | none | right | 0 | 4 |
| **Jev, now** | 7 | **hold, 7/7**, conf 0.73-0.80 | 0.093-0.134 s | 5, 3, 1 every run | 0 | 3 (6 runs), 4 (1, sleep-hit) |
| Decisions API, before | 3 | hold (0.66), 2/3 in time | 0.121, 0.156, 1.115 s | right | 0 | |
| **Decisions API, now** | 6 | **hold, 6/6**, conf 0.88-0.94 | 0.105-0.181 s | 5, 3, 1 every run | 0 | 3 (4), 4 (2) |

Seven Jev runs: five in the (b, e) loop and two in full-five passes; six Decisions API runs likewise.
In the 3-step runs the hold did what it is for: in Jev 165437 the arm held 0.106 s after the text,
carried block 1 for 3.6 s until the new plan came, and took it straight to slot 2, never setting it
down in slot 0. **In the three 4-step runs it did not**: the replan failed while the arm held (the
Decisions API's 170218, a clean run: the planner's plan failed validation, "move_object needs a slot,
a bin or table as target, not tray", 11 s after the hold; Jev 210550 and the API's 042753, sleep-hit:
"Server disconnected" after 22 s, a read timeout after 30 s), the hold released, the arm finished
block 1 into slot 0 under the old plan, and the next plan moved it to slot 2. The end order was right
every time, but the corrected move completed. The new
reading keeps the user's words: "Line the blocks up in the tray, highest on the left." (every run,
both deciders). Jev's confidence sits just under the 0.8 stay-local gate and above the 0.5 right-now
gate, so the hold is taken and the correction goes to the planner, as designed.

### 4. (e), wait then go on: no leak; the guard never had to fire

Jev 6/6 and Decisions API 7/7 paused and resumed (pause 0.109-0.144 s Jev, 0.107-0.150 s API in the
clean runs; 0.265-0.411 s in runs hit by the sleep below, all under 0.5 s). No restated task in any
run carries the wait or the go-on, and `reading_kept` was logged 0 times in all runs (battery and
slice). That is the prompt half of P3 working, not the code guard: the Decisions API's unsure route
sent "hang on a second" to the planner in all 7 of its runs (pause conf 0.2-0.8, `fast_llm`), and the
planner's restatement stayed clean every time, so the guard had nothing to catch. The battery's one
wait ("Hold on a moment." / "Okay, keep going.", paraphrased) was recognised by `is_transient` and
kept out of `user_words` with both deciders.

### 5. Good new plans sent back

| | slice (b, e, full five) | battery (new plans) | battery (first plans) |
|---|---|---|---|
| Jev | 0 / 27 | **14 / 33 (42%)** | 3 / 10 |
| Decisions API | 0 / 26 | 3 / 39 (8%) | 1 / 10 |

The battery send-backs all come from the plan check, and every one flags a SPARE letter: "the
second letter A may be something the task asks about that the plan leaves out (unsure)". The table
holds two or three A's and OPENAI needs one, so leaving one out is what a right plan does (counted
by `live_battery_report.py`: every flagged object's letter is one the word in force needs fewer of
than the table holds; 17 of 17 for Jev, 4 of 4 for the API). "Unsure" is a check Noul with
`|2p - 1| < 0.5` (`Gates.check`); Jev put the spare A at p 0.27-0.42. Under P2 the plan check alone
routes a new plan, so these go straight back; 4 of 10 Jev episodes then hit the 3-rejection limit
and ran 8 plans unchecked (`plan_trusted`). The offline 0% was measured on e12v2's world, where
every object is one the task asks about; the spelling table has spares, and Jev's plan-check
question does not tell a spare from a needed letter. Outcomes did not move (the sent-back plans were
re-issued and ran), but it is most of Jev's 8.1 planner calls per episode.

### 6. What the fixes made worse or did not fix

- **A failed replan releases a hold and the corrected move completes** (section 3, 3 of 13 (b) runs,
  1 of them clean). Not new with the fixes, but the hold is now taken, so it now shows: the brain's
  answer to a failed replan is to run the old plan, which is the plan the user just corrected.
- **Decisions a little slower for Jev.** The slice's clean runs, like for like with section 1: Jev
  acting p50 123 ms, p95 204, max 331 (n 74; before 102 / 130 / 146); Decisions API 120 / 179 / 202
  (n 61; before 129 / 254 / 1,097). The request now carries `user_words`; whether that is the +20 ms
  is not separated from Jev's own variance here.

- **Spare letters and the plan check** (section 5): a new failure mode of P2 on this task, Jev only
  in practice (42% vs 8%). The fix belongs in the plan-check question (spare objects are not the
  task's), not in the routing rule.
- **Remarks join `user_words`.** A remark whose route came back unsure enters the user's task-level
  words: "It slid a little.", "See what I found!", "I've finished that one for you.", "Sorry, just
  reaching over." (both deciders). `_user_words` takes any user_text routed off stay_local; nothing
  in these runs went wrong because of it, but the field is meant to hold the task and corrections.
- **More planner calls after a correction.** In (b) every step_done after the correction went back
  to the planner (route `fast_llm`, confident for Jev): 5-6 planner calls per run against 3 before,
  both deciders. Over the slice's 19 Jev runs, 45 replans (2.4 per run; 8 in 15 before); the
  battery's live deciders 7.1 and 5.5 replans per episode against 0.8 for rules. Whether `user_words`
  (the old task and the correction side by side) is what makes the route unsure after each step is a
  guess not tested here.
- **No plan the check should have caught got through, as far as these runs show**: no step failed in
  any battery episode, and the only misfits were caught by the validator (one Jev plan naming a done
  condition on an absent object; three always_llm plans into occupied slots). Live runs plant no bad
  plans, so the Decisions API's 47/60 offline is neither confirmed nor refuted.
- **Decisions API latency tail.** In battery ep 8 all 18 of its decisions took 1.1-4.8 s (p50 1.6 s)
  while the planner answered at its usual speed in the same episode, so it was the API's side; the
  episode still finished. Jev never went over 500 ms in 153 battery decisions.
- **Not the fixes:** planner latency this run was p50 4-7 s per call (2.8 s in the first live run),
  OpenAI-side.

### Runs lost and re-run

At 17:20 the network dropped (DNS) and the Mac then slept between tests (recording stamps an hour
apart while pytest counted ~3 min; `pmset` shows DarkWake on battery). The two full-five passes
failed outright on DNS (empty recordings, removed); the re-runs ran in a sleeping machine: in them
(c) failed for both deciders (pause 0.854 s after a Jev error at 835 ms that fell back to pause,
`jev_error`; 0.931 s after a 924 ms Decisions API call; planner "Server disconnected"), and the
Decisions API's (a) had a failed planner call. A `caffeinate -i` pass did not prevent it either.
Kept and labelled in `pytest_slice_runs_20261012.txt`. **No full-five pass ran on an awake machine
for either decider; both are owed** (on power, lid open). What the sleep-hit passes show: Jev passed
(a), (b), (c) (pause 0.108 s, resume 0.099 s, closest 0.122 m, in the last pass) and (d) (parked
2.47 s, block held) at least once each, and (e) in 6 runs; the Decisions API passed (b), (d) (parked
2.55 s, block held) and (e), its (a) finished with one failed planner call, and its (c) paused in
0.931 s. (b) and (e) in the clean loop are in the tables above.

### 7. Default decider

**Jev stays the default.** It now holds on the correction 7/7, it pauses and resumes on the wait
6/6 without escalating (conf 0.97, stay_local), and its latency tail stayed under 0.5 s in every
battery decision. The Decisions API is surer on the correction (0.88-0.94) and sends fewer plans
back on this task, but it put 18 decisions in a row over 1 s in one episode and 1.1 s on last run's
hold; the demo's bars are 0.5 s, and a tail the brain cannot fix is worse than planner calls it can
(the spare-letter plan check). Neither beats the rules control on outcome in this sample, and both
share the paused_idle deadlock and the hold that a failed replan releases; those, with the
spare-letter plan check, not the decider, are what to fix next.

## Rules added after the re-run (2026-10-13, offline)

The re-run's four gaps (sections 2, 3, 5 and 6) are rule gaps, and the brain now has rules for them
(`src/robojev/brain/loop.py` "Liveness", `core/combine.py` "A routine step_done",
`core/decision.py` `unmentioned_groups`; `tests/test_brain_liveness.py` holds one test per rule on
the stub). In plain words:

1. **A pause must end.** Code records what each pause is for. A pause for a hand ends in code one
   idle period (5 s) after the hand has gone out of view, whatever the decider says: the hand is
   the one thing code judges on its own. A remark made while a hand is in view counts as a hand
   pause (ep 5 with Jev recorded its pause as "the user said 'Sorry, just reaching over.'"). Any
   other pause (a typed wait, anything else) that outlasts two idle re-asks with the decider still
   unsure becomes a question to the user: "The arm has been paused for 10 s (…). Should it go on?"
   Their answer ends the pause and is decided like any other line.
2. **A failed replan keeps the hold.** The old plan is the one the event said is wrong, so it is
   never run as a fallback. The planner is asked once more; if that fails too, the holding arms
   pause and the user is asked what to do. Before, the hold was released and the corrected move
   completed (section 3, 3 of 13 (b) runs).
3. **Spare letters.** The plan check asks one question per label group, and when the plan already
   uses objects of that label it asks about the count: "the plan already uses 1 object labelled A;
   does the task need more than 1?", with "spare" as the confident no. Before, "the second letter A"
   got the generic "is it something the task asks about?" and Jev answered unsure (section 5).
4. **`user_words` keeps remarks out.** A typed line joins the user's task-level words only when the
   decider itself routed it to a planner or to the user, not when the confidence gate escalated an
   unsure stay_local (section 6: "It slid a little." and the like).
5. **A correction already folded in.** Every request now says which of `user_words` the current
   plan has already read (`user_words_in_task`), and the route question says a correction `task`
   already states is not a request for a different end result. The offline reading of the (b)
   logs: after the correction every request showed the old task beside the correction, and both
   deciders (the Decisions API confidently, Jev unsurely) read that as the user asking for a change
   on every later step; the same day's runs with a one-line `user_words` stayed local.
6. **A routine step_done stays local.** The 0.8 gate guards a wrong local fix; on a step that
   finished normally with no fix and no plan-check problem, keeping local only starts the next
   planned step, so the decider's stay_local stands at any confidence. This is most of the battery's
   planner calls: Jev's stay_local on routine steps sat at 0.3-0.77 and the gate turned 28 of them
   into replans, with or without a correction (the replan-loop limit counts only plans sent back).

What the recorded runs would have done under them, counted from the decisions alone
(`experiments/replay_rules.py`, `experiments/results/battery/replay_rules.txt`; the spare-letter
question needs a live answer, so its column is the size of the pattern, not a result):

| run | step_done replans | of which now local (routine) | paused_idle re-asks that stayed paused | ended by code / asked | plans sent back, unsure-only | remarks that joined user_words | failed replans (hold released) |
|---|---|---|---|---|---|---|---|
| battery, Jev | 42 | 28 | 41 (all ep 5) | asked at the 2nd re-ask | 15 of 17 | 2 | 0 |
| battery, Decisions API | 38 | 17 | 42 (all ep 5) | code resumes at the 1st | 4 of 4 | 3 | 0 |
| slice, Jev | 34 | 13 | 0 | | 0 | 0 | 9 (incl. sleep-hit runs) |
| slice, Decisions API | 34 | 9 | 0 | | 0 | 0 | 6 |

Under the new rules ep 5 ends 5 s after the hand leaves with the Decisions API (its pause was the
hand's) and 10 s after with Jev (its pause was the remark's: the user is asked, and the battery's
user answers). The step_done replans not counted as routine were the decider's own `fast_llm`
answers after a correction (rule 5 speaks to those; a live run says whether it is enough).

The rules are not tuned to these numbers: each is the cost argument the module docstrings give,
and each has a test that fails without it. What only a live run can show: the spare-letter answer
with the count question, the post-correction route with `user_words_in_task`, and whether the
remark-as-hand-pause rule misfires on a correction typed while a hand is in view (then the pause
ends when the hand goes, which is what a hand pause should do; the correction's replan is separate).

## Handoff 8: the battery and the slice under the liveness rules (2026-10-13, live)

The six rules of the previous section, live. AC power, `caffeinate -dims`, and a TLS check to both
providers before every run (at 10:38 something on the network started intercepting TLS, "self-signed
certificate", which killed the first always_llm attempt from its episode 2 and the first slice
block; both were run again whole; the failed attempt's log is `pytest_slice_runs_20261013_tls_failed.txt`).
All four battery tables ran the previous run's `episodes.json`, so the episodes and their
paraphrases are the same as section 1 above, word for word. Files: `experiments/results/battery/20261013-1*-sim-*/`,
`live_tables_20261013-100540.txt`, `experiments/results/live/slice_*20261013-1[01]*.jsonl.gz`,
`pytest_slice_runs_20261013.txt`, `results/battery/replay_rules_after_20261013-095000.txt`.

Two fixes in their own commits: `replay_rules.py --after STAMP` (16ed045; it grouped by decider only,
so `--write` would have merged this run with the last and overwritten its table), and a brain bug
(a597da3, below).

### 1. The battery side by side (sim, seed 1, the same 10 episodes)

| | Jev | Decisions API | rules | always_llm |
|---|---|---|---|---|
| mean outcome score (9 judged) | **1.000** (0.907) | **1.000** (0.907) | 0.889 (1.000) | 0.978 (1.000) |
| finish rate | 1.00 (0.89) | 1.00 (0.89) | 0.89 (1.00) | 0.89 (1.00) |
| s to first change: new_word / wait / go-on / hand | 0.15 / 0.10 / 0.10 / 0.13 | 0.13 / 0.18 / **14.77** / 0.14 | 0.01 / 0.00 / 0.00 / 0.01 | 4.54 / 3.80 / 1.80 / 6.15 |
| s to first change: taken / added / nudged / placed | 1.42 / 3.02 / 5.89 / 0.65 | 1.28 / 6.29 / 5.01 / 0.94 | 1.15 / 3.68 / 4.88 / 1.15 | 1.16 / 2.89 / 3.62 / 1.16 |
| planner calls / episode | **2.0** (8.1) | 4.9 (6.5) | 1.8 (1.8) | 2.7 (2.6) |
| decisions / episode, p50 / p95 / max | 9.3, 152 / 281 / 354 ms | 11.2, 146 / 251 / 579 ms | 9.3, 1 ms | 10.0, 1 ms |
| drops / contacts / STOP ok | 0 / 0 of 1 / 1 of 1 | 0 / 0 of 1 / 1 of 1 | 0 / 0 of 1 / 1 of 1 | 0 / **1 of 1** / 1 of 1 |
| asks | 0 | 1 | 0 | 0 |
| median wall s | 57.4 | 64.5 | 54.0 | 54.6 |

(Last run in brackets.) Both live deciders now finish every judged episode. The two controls each
lost one: rules spelled OPENAI in ep 7 after "Could you spell PANE instead, please?" (a brain bug,
section 7), and always_llm left a stray A after PINE in ep 4 (its replans misfit twice while the
arm carried on, then the third plan kept the A; the control's own cost). The same small-sample
caveat as before: one hand, one wait, one STOP, three new words.

### 2. Ep 5, a hand mid-carry with "Sorry, just reaching over."

Finished with both deciders (last run: both timed out at 240 s with "O" placed).

- **Jev**: the hand's decision landed first this time (pause 0.125 s after the hand); when the hand
  left (16.56 s) Jev answered `resume`, confidently, 0.10 s later. The new rule was not needed.
- **Decisions API**: it paused in 0.14 s; when the hand left (18.13 s) it answered an unsure `hold`;
  code ended the pause at 23.29 s (`resumed`: "paused for a hand that has gone; nothing else holds
  the arm"), 5.2 s after the hand left, with no re-ask and no question. Score 1.0 in 67.2 s.

The slice's (c) shows the same split: Jev resumed 0.099 s after the hand left, the Decisions API
5.105 s (code).

### 3. Plans sent back, and the spare-letter question

| | new plans sent back | first plans sent back | loop limit hit |
|---|---|---|---|
| Jev | **3 / 9** (14 / 33 last run) | 1 / 10 | 1 (4) |
| Decisions API | 5 / 22 (3 / 39) | 2 / 10 | 1 (0) |
| slice, both | 0 | 0 | 0 |

Every send-back is still a spare A (`live_battery_report.py`: 4 of 4 for Jev, 7 of 7 for the API).
The count question did go out ("`plan` already uses 1 object labelled A (…). the third letter A is
another one …"). Answers to it, as `check.letter_A*` confidences on plan_arrived (|2p − 1|; under 0.5
is "unsure"): Jev 19 answers, 0.02-0.82, median 0.68, 4 under 0.5; the Decisions API 32, 0.0-1.0,
median 0.92, 4 under 0.5. Three of Jev's four send-backs are in ep 0, where a cube is taken and a
third A comes into play: it answered 0.16 ("the task needs it", p ≈ 0.58), 0.02 and 0.40 on three
successive plans, and the loop limit ran the fourth unchecked; the fourth is a first plan in ep 9. Jev's send-back count fell from 17 to 4, but mostly because
rule 6 cut its new plans from 33 to 9; as a rate it is 33% of new plans against 42%. The count
question narrows the pattern; it does not remove it.

### 4. Planner calls after a correction (b)

| | runs | hold on the correction | text to hold | planner calls per run | post-correction step_done |
|---|---|---|---|---|---|
| Jev | 6 | hold 6/6, 0.77-0.80 | 0.103-0.182 s | **2** (5-6 last run) | 18 of 18 stay_local, the decider's own, 0.88-0.99 |
| Decisions API | 6 | hold 6/6, 0.58-0.74 (0.88-0.94 last run) | 0.113-0.199 s | 3 | 12 of 18 stay_local (6 of them kept local by the routine rule at 0.29-0.61); the 3rd step_done went to the planner in every run, its own `fast_llm` at 0.40-0.65 |

Final order 5, 3, 1 every run, no drops. Jev's post-correction re-judging is gone. In the battery
the Decisions API also sent 17 step_done to the planner across 8 episodes, each its own `fast_llm`
answer (route conf 0.36-0.75): 12 in 6 episodes with no correction at all, 5 after one (eps 4 and 8).
The routine rule keeps an unsure stay_local, not a replan the decider chose (Jev: 0).

The Decisions API's lower confidence on the correction (0.58-0.74 against 0.88-0.94) is not
separated from day-to-day drift: the "all three as before" variant of section 7's ablation is the
request the last run recorded at 0.68, and today it answers 0.88. Only the (e) regression is pinned
to the rules (within one day, one request, one field at a time).

### 5. Failed replans

None that mattered in the slice (last run 3 of 13 (b) runs ended with the corrected move completed
under the old plan; this run 0 of 12: block 1 was never set down in slot 0). `replay_rules` counts 2
"failed replans" for the slice's Decisions API: both are in one gave_up (e) run, at 84.4 s, unusable
("new step id n1 is already used") and already stale when they came back, with the arm paused, not
holding; the column counts every unusable replan, whatever happened to it. In the battery one, always_llm ep 4:
the plan failed validation, `plan_retry` asked once more, the retry came back usable. always_llm does
not hold while it plans, so there was no hold to keep.

### 6. `user_words`

- Jev: only corrections, in the battery (PINE, PANE, IDEA) and the slice (6/6 "actually, highest on
  the left"). No remark joined.
- Decisions API: the corrections plus "See what I found!" (ep 6, its own `fast_llm` route, which
  rule 4 allows). And in the slice, **the correction itself was kept out in 2 of 6 (b) runs**: the
  API's route was an unsure stay_local (0.40) that the gate escalated, and rule 4 takes a line only
  when the decider itself routed it off the plan. The replan still read the line (the order came out
  right), but the later requests showed `user_words` without it.
- rules and always_llm: every routed line, remarks included ("It slid a little.", "Sorry, just
  reaching over.", "I've finished that one for you."); the controls route by keyword or always, so
  rule 4 does not filter them.

### 7. What the rules made worse, and the bug

- **The Decisions API cannot get past "ok, go on" in the slice: (e) 0 / 6** (7 / 7 last run). On the
  go-on it answered `carry_on` 0.28 (resume p 0.46) or `pause` 0.11, under the gate, so the arm
  stayed paused; after two re-asks the rule asked the user ("Should it go on?"), the slice's scripted
  user never answers (`ScriptedUser.ask` records and returns), and every later re-ask routed to the
  planner until the 12-request cap ended the run (`gave_up` at 76-88 s). Rule 5 is the cause.
  `experiments/ablate_go_on.py` sends the recorded request with rule 5's additions put back one by
  one, 5 calls each (answers were identical across the 5):

  | the "ok, go on" request | Decisions API right_now |
  |---|---|
  | as sent after the rules | resume, conf 0.28 (p 0.46) |
  | without `user_words_in_task` | resume 0.56 (p 0.67) |
  | `about_the_task` as before | resume 0.39 (p 0.54) |
  | route `ask_about` as before | resume 0.28 (p 0.46) |
  | all three as before | resume 0.88 (p 0.91) |

  Jev on the same scenario: 6 / 6, pause 0.94-0.97, resume 0.91-0.95, stay_local. The battery's
  wait episode shows the same API answer (carry_on on "Okay, keep going.", the ask, then the
  battery user's scripted reply, which answers the end-result question, not "should it go on";
  the arm ran again 23 s after the go-on). Not changed: the fix is in the wording of rule 5, and
  the rules say not to tune to the numbers; the ablation says where.
- **Questions that should not have been asked**: seven "Should it go on?" asks (battery ep 6 and all
  six slice (e) runs, Decisions API), each after the user had already said go on. The cause is the
  decider's `carry_on` on the go-on under rule 5, not the ask rule, which did its job as a backstop
  where an answer came (the battery) and could not where none does (the slice).
- **A question the slice cannot answer turns into planner calls**: once asked, each paused_idle
  re-ask with an unsure route goes to the planner (8-10 per (e) run). With a real user it would be
  waiting; with nobody answering, the request cap is the only stop.
- **A hand pause ended while a hand was near**: none. Both `resumed` notes (battery ep 5, slice (c),
  Decisions API) came 5 s after "the person's hand has gone out of view" with no hand reported after.
- **A routine step kept local that a replan would have fixed**: none seen; no step failed and no
  `plan_done_unmet` in any battery episode or slice run.
- **The bug (a597da3, fixed)**: a line that needs the planner, typed while the first plan is still
  on its way, was dropped. `_start_replan` returned with "the first plan is on its way; it will read
  user_messages", but that request was built at t = 0. Battery ep 7 with the rules control: "spell
  PANE instead" at 9.8 s, an 11 s first plan reading OPENAI, OPENAI spelled. The brain now asks for
  the first plan again with the line in it (`plan_rerequested`; the earlier request goes stale), and
  `words_planned` counts the `user_words` a request was built with. Test:
  `test_a_correction_typed_before_the_first_plan_arrives_is_planned_for`. The live re-run of ep 7
  after the fix spelled PANE, but its first plan came in 5.5 s, before the line, so the fixed path
  is covered by the test only. Not the rules' doing: the path existed before; the last run's first
  plans were all faster than the line.

### 8. Decision latency (slice, all clean)

Jev 127 / 167 / 200 ms (p50 / p95 / max, n 84); Decisions API 127 / 270 / 3,801 ms (n 156).

### 9. Default decider

**Jev stays the default, more clearly than before.** Under the new rules it finished every battery
episode at 2.0 planner calls per episode, passed every slice scenario on an awake machine (the full
five 5 / 5, (b) 6 / 6, (e) 6 / 6), held the correction at 0.77-0.80 and stayed local after it, and
its slowest decision in the slice was 200 ms. The Decisions API finished the battery too, but rule 5
cost it the wait scenario (0 / 6; the ablation shows why), its confidence on the correction fell to
0.58-0.74, it sends routine steps to the planner on its own, and it had a 3.8 s decision. Next:
rule 5's wording for the Decisions API (or a per-decider state), the spare-letter question for
Jev's taken-cube case, and an answer to "should it go on?" in the slice's scripted user so the ask
rule can be exercised there.

## After Handoff 8 (2026-10-13, offline)

Three changes from the Handoff 8 findings, each with a test in `tests/test_brain_liveness.py`:

1. **Rule 5 is scoped to events about the plan's progress.** `user_words_in_task` and the route
   question's sentence about a folded-in correction go out on step_done, plan_arrived, scene_change,
   step_failed and plan_done_unmet, never on a typed line or a pause re-ask, and the right-now
   question's wording is as it was before rule 5. Section 7's ablation put the Decisions API's
   "ok, go on" at 0.28 with the field and the right-now sentence and 0.88 without; on a typed line
   the question is the line itself, so the field had nothing to add there. Jev's post-correction
   steps (section 4, 18 of 18 local) are step_done events and keep the field.
2. **A line the planner changed the plan for is a correction.** When a replan triggered by a typed
   line comes back and changes the steps (new steps, a different order or set of pending steps, a
   done condition or constraint added or dropped; a new reading alone does not count), the line
   joins `user_words` whatever the route's confidence was. Section 6: the correction itself was kept
   out in 2 of 6 (b) runs because the gate, not the decider, had routed it. A remark whose replan
   changes nothing still stays out.
3. **The slice's scripted user answers "Should it go on?"** (`ScriptedUser(on_ask=...)`, used by
   scenario (e) with "ok, go on"), so the ask rule can be exercised there instead of running to the
   request cap. The battery's user already answered.

Also: the count question now says how many objects of the label are in view ("Of the 3 objects
labelled A in view, `plan` already uses 1"), for Jev's taken-cube case (section 3); only a live run
says whether that helps.

Not changed: the Decisions API's own `fast_llm` on routine steps (section 4; the routine rule keeps
an unsure stay_local, not a replan the decider chose). A rule that overrode the decider's own
replan on a finished step would make the router meaningless on the event it is most often asked
about; if the live rate stays where it is, the fix belongs in the route criteria's text, not in a
gate. Handoff 9 re-runs (e) with the Decisions API, and the battery for both.

## Handoff 9: the wait and the correction after the rule 5 fix (2026-10-13, live)

After 3314cfb: rule 5's text only goes out on plan-progress events, a line the planner changed the
plan for joins `user_words`, the slice's user answers "Should it go on?" with "ok, go on", and the
count question says how many of the label are in view. The Mac was on battery this time, so
`caffeinate -dims`, the TLS check before every run, and the wall clock per run (no run's wall was
longer than pytest's own count, so no sleep). Files: `experiments/results/live/slice_*20261013-17*.jsonl.gz`,
`pytest_slice_runs_20261013_h9.txt`; `experiments/results/battery/20261013-171549-sim-jev/`,
`20261013-172528-sim-decisions/`, `live_tables_h9_*.txt`; `replay_rules_after_20261013-170000.txt`.
Brain rules unchanged; no bug fixes this time.

### 1. (e), wait then go on

| | runs | pass | "ok, go on" | go-on to resume | asks | gave_up |
|---|---|---|---|---|---|---|
| Jev | 3 | **3** | resume, stay_local | 0.085-0.096 s | 0 | 0 |
| Decisions API | 6 | **2** (0 / 6 in Handoff 8) | resume 0.88 in runs 1-2; 0.27-0.49 (unsure) in runs 3-6 | 0.437, 0.255 s; 14.7-18.4 s via the ask in runs 3-5 (run 6: the go-on was decided before the wait, not measured) | 4 | **0** |

All six Decisions API runs ended `done`. In the four that failed, the go-on came back unsure, the
arm stayed paused, after two re-asks the brain asked "Should it go on?", the scripted user said "ok,
go on", and the arm resumed and finished (the ask rule now exercised, as intended). They fail the
test on its 0.5 s bars: the pause on "hang on a second" took 0.70, 1.58, 1.13 and 2.51 s.

**Why runs 3-6 differ from 1-2: the Decisions API's latency, not rule 5.** Its decisions in those
four runs had p50 349-1,242 ms and max 1.05-2.75 s (runs 1-2: p50 161-252 ms); the planner, on the
same OpenAI account, answered at its usual 2-8 s in the same runs, and Jev ran at p50 92-118 ms over
the same network a minute later, so the slow window (17:07-17:10) was the API's own, as in Handoff
8's ep 8. A slow pause let the arm move on before it stopped, and the go-on request then described a
different moment: run 1's said "paused: carrying block_1 to slot_0", run 4's "paused: lowering
block_1 at slot_0, not yet released" (the only differences between the two requests' states, besides
a nearby block), and the API is unsure about resuming mid-lowering. In run 6 the wait took 2.5 s
and the go-on (typed 2 s after it) was decided first.

**The ablation on the new recording** (run 1's "ok, go on" request, 5 calls each): as sent now,
resume 0.88; every variant 0.88. Rule 5's text is no longer in the request (the only field that
still differs from last run's request is the route's `ask_about`, which it does not move).

### 2. (b), the correction

| | runs | hold | conf | text to hold | correction in `user_words` | order, drops |
|---|---|---|---|---|---|---|
| Jev | 3 | 3 / 3 | 0.77-0.85 | 0.091-0.120 s | **3 / 3** | 5, 3, 1; 0 |
| Decisions API | 3 | 3 / 3 | **0.88-0.93** (0.58-0.74 in Handoff 8) | 0.108-0.151 s | **3 / 3** (4 / 6 in Handoff 8) | 5, 3, 1; 0 |

The Decisions API's route on the correction was its own `fast_llm` in all three (0.76-0.84), so the
new rule for a line the planner changed the plan for was not needed to keep it in.

### 3. The battery (the same 10 episodes)

| | Jev | Decisions API |
|---|---|---|
| mean score, finish rate | **1.000, 1.00** (1.000, 1.00) | 0.978, 0.89 (1.000, 1.00) |
| s to first change: new_word / wait / go-on / hand | 0.14 / 0.13 / 0.10 / 0.10 | 0.12 / 0.30 / 0.13 / 0.13 |
| planner calls / episode | 2.3 (2.0) | 4.6 (4.9) |
| decision latency p50 / p95 / max | 142 / 181 / 204 ms | 133 / 228 / 399 ms |
| new plans sent back | **1 / 9** (3 / 9) | 2 / 21 (5 / 22) |
| drops / contacts / STOP ok / asks | 0 / 0 / 1 of 1 / 0 | 0 / 0 / 1 of 1 / 0 |

(Handoff 8 in brackets.) The taken-cube episode, ep 0, `check.*` answers on each plan:

| | first plan | 2nd | 3rd |
|---|---|---|---|
| Jev | A2 0.68, D 0.90: kept | **A3 0.04, "is something the task asks about"**: sent back | D 0.92: kept |
| Decisions API | A2 0.98, D 1.0: kept | A3 0.48 (unsure): sent back | A3 0.82, D 1.0: kept |

One send-back each in ep 0 (Jev: three in Handoff 8, then the loop limit). The others: Jev's first
plan in ep 9 (a spare letter, as before); the Decisions API's letter D unsure (eps 1 and 7; D is in
none of the words) and, at the end of its ep 8, "the letter I is something the task asks about but the plan leaves out" on a
plan with no steps left, after IDEA was spelled (I was in place; a false flag, harmless).

**The Decisions API's ep 4 is the one miss**: PINE then an O in position 4 (score 0.80). Its first
replan after "Could you spell PINE instead?" failed validation (the arm was holding P and the plan
did not put it down first), `plan_retry` asked once more (the hold kept, as the rule says), and the
retried plan moved the O from position 0 to position 4 rather than off the row. The plan check asks
about objects a plan leaves out; an object it misplaces is not its question, so the plan ran, and
the brain declared done on its done conditions. A planner error the brain has no check for, not a
rule effect (Jev's run of the same episode, and both of Handoff 8's, spelled PINE).

### 4. Worse than Handoff 8

- The Decisions API's (e) latency window above (its decisions only).
- The Decisions API's ep 4 (a misplaced letter, no check covers it).
- Still: the Decisions API sends routine step_done to the planner on its own `fast_llm` (15 in the
  battery, 4 in the slice; Jev 0), which the routine rule does not cover.
- Nothing worse for Jev: (e) 3 / 3, (b) 3 / 3, the battery 1.000 at 2.3 planner calls, max decision
  204 ms.

**Default decider: Jev, unchanged.** The rule 5 fix brought the Decisions API's go-on back (0.88 when
its decisions are quick, and the ask path now ends a stuck pause), but its latency tail decided
four of six (e) runs this time, and it is the tail, not the rules, that keeps it second.
