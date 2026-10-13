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
