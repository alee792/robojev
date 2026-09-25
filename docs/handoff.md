# Handoff: run E11 and E12 against Jev

Paste the block below into a Claude Code session on a machine that has a TypeSafe key (the bay Mac).
It is self-contained.

---

You are picking up robojev v2 spikes that were built in a sandbox with no Jev access. Your job is to
run them against the real Jev API, analyse the results, and push them back to the same branch.

**Setup**
1. `git fetch origin claude/first-principles-approach-ow3jo2 && git checkout claude/first-principles-approach-ow3jo2`
2. Read `docs/v2.md` and `docs/v2-diagrams.md` (the design: Perception, a harness, and three Jev
   layers: the Sequencer checks progress against the task, the Spotter classifies changes the robot
   didn't cause, the Router decides who handles an event), `docs/assessment.md` (what v1 showed), and
   the E11 and E12 sections of `context/07-experiments-to-run.md`.
3. Confirm `TYPESAFE_API_KEY` is set in the environment or in `.env` at the repo root. Never print it.
4. `uv sync --frozen` at the repo root, then `uv run --frozen pytest -q` (expect 90 passed, 3 skipped).

**Important: E12 was built against an earlier version of the design.** Its code has a "Listener"
(which only classified user text, now generalised into the Router), a Spotter that only handles hands
(now any change the robot didn't cause), a Sequencer that picks the next skill (now it checks progress
against the task), and older names (knobs = parameters, utterance = correction, orders = constraints).
Run it as is. Its numbers on Jev's accuracy, latency and confidence carry over; its conclusions about
which layers earn their place apply to the old design. Keep the two apart in the write-up. Do not
rework E12 in this session.

**Run, in this order. Stop and report if any live run errors on more than 2% of requests.**
```
# E11: one decision after "actually, reverse it"   (~$0.05)
cd experiments
uv run python e11_reorder.py --dry-run
uv run python e11_reorder.py --n-scenes 60
cd ..

# E12: closed-loop text blocks world   (~$0.03 per run, ~$0.10 per sweep)
uv run --frozen python experiments/e12_blocksworld.py --dry-run
uv run --frozen python experiments/e12_blocksworld.py --backend jev --latency-ticks auto
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto --seed 2
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto --seed 3
```
Optional, only if `ANTHROPIC_API_KEY` is set (about $0.5-1 each): the same questions answered by
claude-haiku-4-5, for accuracy and latency comparison.
```
cd experiments && uv run --extra baseline python e11_reorder.py --baseline claude && cd ..
uv run --frozen --extra vlm python experiments/e12_blocksworld.py --backend claude --scenario reverse_midway,hand_in_path,ambiguous
```
Save each run's printed tables to `experiments/results/e11_run.txt` and `e12_run.txt` (append, with
the command line above each). Logs land in `experiments/results/e11_reorder.jsonl` and
`e12_blocksworld.jsonl`.

**Questions to answer, with numbers**
1. Headline: can Jev alone handle "actually, reverse it", or does it need the fast LLM? From E11:
   `raw` (Jev works the new order out itself) vs `solved` (code gives the answer for each order and
   Jev only maps the correction onto it). At which row size does `raw` break? Is `direction` more
   robust than `target_slot` and `next_pick`? Dense vs sparse numbering?
2. Can confidence route mistakes? From the gate tables: accuracy and coverage at each threshold, and
   whether `escalate` / `ask_a_smarter_model` fires more on the requests Jev gets wrong. This decides
   whether the Router can work.
3. E12: completion, recoveries and violations per scenario, `solved` vs `raw`. Which scenarios fail,
   and what did Jev pick at the failure (read the JSONL log)?
4. E12 ablations: what breaks with `--no-spotter`, and what `--no-listener` costs (ticks from a
   correction to changed behaviour; needless escalations on `chatter`)?
5. Latency with real answer timing (`--latency-ticks auto`): p50/p95 per layer, and whether it
   changed outcomes.
6. Where Jev disagrees with the oracle, per question; and, if the baseline ran, whether Jev and Haiku
   make the same mistakes.

**Known weaknesses of the harness, to keep in mind when interpreting**
- E12 `solved` gives the Sequencer code's answer (`next_step`), so it picks a branch rather than
  plans. `raw` is the real test.
- E12's Spotter instructions restate the oracle's thresholds in words, so it is close to a lookup. If
  it scores near 100%, that is why; say so.
- The code stop-distance for a hand is 3 cm, inside the 5 cm contact zone, on purpose, so the
  no-Spotter run can show something. It is not a safety value.
- A confident wrong `stop` from the Listener stalls a run (nothing resumes it). The current design
  fixes this (a model can only pause); if it happens, report it as confirming that fix.

**Rules**
- Do not change the harness to improve the numbers. If you find a harness bug, fix it in its own
  commit, say what it changed, and rerun the affected runs.
- Do not remove or soften the weaknesses above in the write-up.

**Deliverables**, pushed to `claude/first-principles-approach-ow3jo2` (PR #1):
1. `experiments/results/e11_reorder.jsonl`, `e12_blocksworld.jsonl`, and the two `*_run.txt` files.
2. New "E11" and "E12" sections in `context/08-experiment-results.md`, in that file's existing style:
   tables, then takeaways with numbers, each marked measured or inferred.
3. A short "What the spikes say" section at the end of `docs/v2.md`: can the fast path handle
   corrections without an LLM, which facts code must precompute, whether confidence can drive the
   Router, and what E12 must change before it tests the current design.
4. A comment on PR #1 answering questions 1-6 in a few lines each.

---

## Handback (2026-09-23, bay Mac session)

Done: all deliverables are on this branch and PR #1 has the answers comment. Results:
`context/08-experiment-results.md` (E11, E12, and "Getting more out of Jev"). The E12 tick log is
`e12_blocksworld.jsonl.gz` (133 MB uncompressed, over GitHub's limit). No Claude baseline was run (no
`ANTHROPIC_API_KEY`).

**Where things went wrong, for reformulating the approach**

1. **E12 `raw` never tested what it was meant to.** All 30 runs stuck at the first `release` on
   block 1: Jev picked `retreat` 7,162 of 7,168 times, which lifts the lowered block, and then `place`
   again, until the cap. Sorting, reversal and sparse numbers were never reached, so E12 says nothing
   closed-loop about E11's question. The likely cause is a missing fact: the state says
   `gripper_is_above: "nothing: the gripper is low"`, `"down low, among the blocks"`, and the target
   slot shows empty (the block is still in the gripper). Nothing says "lowered at the destination, not
   yet released", and after `retreat` the history is gone. Test it before concluding that Jev can't
   sequence skills.
2. **The confidence gates were too loose, and one question had none.** A flat 0.3 let 99% of the
   wrong `retreat` picks through (their confidence averaged 0.48 vs 0.82 for right picks; AUROC 0.96).
   `when` had no gate and applied a 0.06-confidence answer, which probably cost 7.5 s. Nothing
   detected the place/retreat loop.
3. **The `escalate` question doesn't work as a router.** AUROC 0.43 / 0.75 / 0.36 across E11's
   variants; the weakest confidence in the request did better (0.71 / 0.70 / 0.93).
4. **`solved` is close to a lookup, and so is the Spotter.** E12 `solved` (40/40) hands the Sequencer
   the oracle's `next_step`, and the Spotter's instructions restate the oracle's thresholds (96%). Both
   show that the harness and the gates work, not that Jev can plan or perceive.
5. **A harness bug (fixed in `a06a48c`).** `plan.above()` used a per-axis box and `pick` a radius, so
   a replan that stopped a move short could offer a `pick` that never succeeds (the oracle loops too).
   Only `--no-listener` was affected; it was rerun live for all three seeds.
6. **The seeds are near-replicates.** Without `--noise` the scenes barely change between seeds, so the
   three sweeps measure Jev's nondeterminism and latency, not scene variety.
7. **Cost ran over the estimate:** about $1.22 in total, against about $0.35. `raw` accounted for
   $0.83 because every run hit the cap. Any rerun of `raw` needs a loop detector or a lower
   `--max-requests`.
8. **Minor issues.** The E11 answer-log file includes the progress lines. A stale `.venv` from another
   branch broke `cv2` until `opencv-contrib-python-headless` was reinstalled (an environment problem,
   not the repo's). One E11 `intent` miss: a coworker request was read as `new_task`.

**Cheapest next steps** (suggested in the results doc):
- (a) Replay both logs offline with per-question thresholds on the weakest confidence. This costs
  nothing.
- (b) Rerun E12 `raw` with one "step in the per-block sequence" fact plus a loop detector, about
  $0.10–0.30.
- (c) An E11 variant whose options carry their meaning ("to the smallest-number end") instead of slot
  numbers.

---

# Handoff 2: run E13 (LLM-prepared branches)

Paste the block below into a Claude Code session on the bay Mac. It needs `TYPESAFE_API_KEY` and
`OPENAI_API_KEY`.

---

You are running experiment E13 for robojev. It tests the assumption the v2 design now rests on: a
fast LLM, not hand-written task code, prepares the plan's branches (the final position of every
block for each variant of the task a user might ask for); when the user corrects the task mid-run,
Jev picks which prepared branch they mean; and when Jev's confidence is low, or no branch fits, the
correction goes back to the LLM. The harness is built and tested offline; nothing has been run
against the real models.

**Setup**
1. `git fetch origin claude/first-principles-approach-ow3jo2 && git checkout claude/first-principles-approach-ow3jo2`
2. Read `docs/v2.md` (especially "Example: a mid-task correction" and "Routing"), the E11 replay and
   "Getting more out of Jev" sections of `context/08-experiment-results.md`, and the E13 section of
   `context/07-experiments-to-run.md`.
3. Confirm `TYPESAFE_API_KEY` and `OPENAI_API_KEY` are set (environment or `.env`). Never print them.
4. The LLM is "ChatGPT Terra". Its exact model id is not known to the harness and must not be
   guessed: run `cd experiments && uv run --extra llm python e13_branches.py --llm openai --jev mock --tasks 1`
   with no model set, which lists the account's models whose id contains "terra" and exits. If there
   is exactly one, use it; if several or none, stop and ask the user. Set it as `OPENAI_MODEL`.
5. `uv run --frozen pytest -q` at the repo root (expect 137 passed, 3 skipped).

**Run, in this order, from `experiments/`. Stop and report if more than 5% of LLM calls or 2% of Jev
calls fail.**
```
uv run python e13_branches.py --dry-run
# small first: 7 tasks, capped LLM calls
uv run --extra llm python e13_branches.py --llm openai --jev jev --tasks 7 --max-llm-calls 50
# full default run (28 tasks x 5 corrections; ~190 LLM calls, ~$0.005 of Jev)
uv run --extra llm python e13_branches.py --llm openai --jev jev
# offline gate sweep from the log (free)
uv run python e13_branches.py --replay results/e13_branches.jsonl --gate 0.5
uv run python e13_branches.py --replay results/e13_branches.jsonl --gate 0.7
```
Before the full run, look up the model's price and run the dry run with `--llm-price-in` /
`--llm-price-out` to estimate the LLM cost; if it is over $5, ask the user first. Save every printed
table, with its command line, to `experiments/results/e13_run.txt`.

**Questions to answer, with numbers**
1. Can the fast LLM prepare branches? Valid plans first time and after one retry; default branch
   correct; share of coverable corrections for which it prepared the right branch. Which task
   kinds does it get wrong, and how (read the log)?
2. Can Jev pick the right branch from a spoken correction? Route accuracy by correction kind
   (coverable, not coverable, chatter, pause); branch-pick accuracy.
3. Does low confidence catch Jev's mistakes? Accuracy, escalation rate and mistakes caught at each
   gate from the replay sweep. Which gate would you set, and why?
4. End to end against "always ask the LLM": accuracy and latency (p50/p95), overall and by
   correction kind. Where does the Jev path win, and where does it lose?
5. LLM latency and token use per plan and per escalation; the real LLM cost of the run.

**Known limits, to keep in mind when interpreting**
- Open loop: each correction is judged on its own, from the default branch; no arm or simulation.
- The evaluation oracle is lenient in places (gaps in tray lines; "any order" tasks accept any
  valid arrangement). An uncoverable correction can occasionally match the current goal by chance.
- Block ids and destinations are fixed lists in the output schema, so the LLM cannot name a
  destination that doesn't exist; validity failures are only missing, duplicate or clashing blocks.
- The always-LLM baseline and the system's escalation share one LLM call per correction.

**Rules**
- Do not change the harness or its prompts to improve the numbers. If you find a harness bug, fix it
  in its own commit, say what it changed, and rerun what it affected.
- Keep the limits above in the write-up.

**Deliverables**, pushed to `claude/first-principles-approach-ow3jo2` (PR #1):
1. `experiments/results/e13_branches.jsonl` (gzip it if it is over 50 MB) and `e13_run.txt`.
2. An "E13" section in `context/08-experiment-results.md` in that file's style: tables, then
   takeaways with numbers, each marked measured or inferred.
3. Add E13 to "What the spikes say" at the end of `docs/v2.md`: does the LLM-prepares-branches,
   Jev-picks design hold, what gate to use, and what to change.
4. A comment on PR #1 answering questions 1-5 in a few lines each, in plain language.

---

# Handoff 3: run the E12 update (e12v2, the v2 design closed loop)

Paste the block below into a Claude Code session on the bay Mac. It needs `TYPESAFE_API_KEY` and
`OPENAI_API_KEY`.

---

You are running e12v2 for robojev: a text-only, closed-loop simulation of the v2 design
(`docs/v2.md`), with controls. Its results decide whether we move on to building the real harness
in MuJoCo. It is built and tested offline; nothing has been run against the real models.

**Setup**
1. `git fetch origin claude/first-principles-approach-ow3jo2 && git checkout claude/first-principles-approach-ow3jo2`
2. Read `docs/v2.md` and `docs/v2-diagrams.md` (the design), `docs/spike-outcomes.md` (what earlier
   spikes found), and the "E12 update (e12v2)" section of `context/07-experiments-to-run.md` (arms,
   scenarios, pass criteria).
3. Confirm `TYPESAFE_API_KEY` and `OPENAI_API_KEY` are set (environment or `.env`). Never print them.
4. `uv run --frozen pytest -q` at the repo root (expect 250 passed, 3 skipped).
5. Use `OPENAI_MODEL=gpt-6-luna` with `--llm-effort low` (the cheapest accurate model in E13). Look up
   its price and pass `--llm-price-in/--llm-price-out`.

**Run, in this order. Stop and report if more than 2% of Jev calls or 5% of LLM calls fail.**
```
uv run --frozen python experiments/e12v2.py --dry-run
uv run --frozen python experiments/e12v2.py --show-rules
# small live check: live Jev, mock LLM
uv run --frozen python experiments/e12v2.py --jev jev
# full run: live Jev and LLM, 3 seeds (about 1,500 Jev requests, ~600 LLM calls)
cd experiments && OPENAI_MODEL=gpt-6-luna uv run --extra llm python e12v2.py --jev jev --llm openai --llm-effort low --seeds 3 --llm-price-in <in> --llm-price-out <out> && cd ..
# the same with perception noise, 1 seed
cd experiments && OPENAI_MODEL=gpt-6-luna uv run --extra llm python e12v2.py --jev jev --llm openai --llm-effort low --noise --llm-price-in <in> --llm-price-out <out> && cd ..
# offline gate sweep from each run's log (free)
uv run --frozen python experiments/e12v2.py --replay experiments/results/e12v2_<run>.jsonl
```
If the dry run's cost estimate for the full run is over $5, ask the user first. Save every printed
table, with its command line, to `experiments/results/e12v2_run.txt`.

**Questions to answer, with numbers**
1. The five pass criteria, as printed. For each FAIL, why (read the log).
2. Where does `jev` beat `rules`, and where do the rules win? Name the scenarios and events.
3. Against `always_llm`: correction-to-behaviour time, LLM calls, time the arm spent holding, cost.
4. What each ablation (`jev-no-right-now`, `jev-no-in-plan-fix`, `jev-no-router`) breaks.
5. Jev's mistakes by question group, and whether the gate caught them; the gate you would set.
6. LLM plan and replan latency and tokens, and whether prompt caching happened (cached tokens are
   logged).
7. Held-out results, separately.

**Known limits, to keep in mind**
- The world, skills and person are simulated in text; perception is ground truth unless `--noise`.
- The rules control is a fair rule set but written knowing the core scenarios' phrasing; criterion 2
  is therefore judged on the held-out scenarios.
- Seeds vary little without `--noise`.

**Rules**
- Do not change the harness, prompts or rules control to improve the numbers. If you find a harness
  bug, fix it in its own commit, say what it changed, and rerun what it affected.
- Keep the limits above in the write-up.

**Deliverables**, pushed to `claude/first-principles-approach-ow3jo2` (PR #1):
1. The run logs in `experiments/results/` (gzip any over 50 MB) and `e12v2_run.txt`.
2. An "E12 update (e12v2)" section in `context/08-experiment-results.md`: tables, then takeaways
   with numbers, each marked measured or inferred.
3. The findings added to `docs/spike-outcomes.md` (not `docs/v2.md`, which holds only durable design).
4. A comment on PR #1 answering questions 1-7 in plain language, ending with a recommendation:
   move to MuJoCo or not, and why.

---

# Handoff 4: rerun e12v2 after the correction fix

Paste the block below into a Claude Code session on the bay Mac (`TYPESAFE_API_KEY`, `OPENAI_API_KEY`).

---

You are rerunning e12v2 for robojev after three fixes to the correction failure found in the last run
(`sort_correction` looped until the LLM-call cap because Jev saw the original task and the
correction as separate fields):

1. Decisions see the task as it stands now: the planner restates it with every plan (`reading`),
   corrections folded in; `user_said_earlier` is gone. While a replan for a correction is pending, the
   state carries `change_being_planned`.
2. A replan loop is detected: once new plans have been sent back to the LLM 3 times since the user
   last spoke, later plans run as they come until the user says something new.
3. The stay-local gate defaults to 0.8.

**Setup**
1. `git fetch origin claude/first-principles-approach-ow3jo2 && git checkout claude/first-principles-approach-ow3jo2`
2. Read `docs/v2.md`, the e12v2 sections of `context/08-experiment-results.md` and
   `docs/spike-outcomes.md` (the last run and why it failed), and this handoff's Handoff 3 for the
   commands and rules, which still apply.
3. `uv run --frozen pytest -q` at the repo root (expect 252 passed, 3 skipped).

**Run** (same as Handoff 3; `gpt-6-luna`, `--llm-effort low`, prices from the last run):
```
uv run --frozen python experiments/e12v2.py --dry-run
cd experiments && OPENAI_MODEL=gpt-6-luna uv run --extra llm python e12v2.py --jev jev --llm openai --llm-effort low --seeds 3 --llm-price-in <in> --llm-price-out <out> && cd ..
cd experiments && OPENAI_MODEL=gpt-6-luna uv run --extra llm python e12v2.py --jev jev --llm openai --llm-effort low --noise --llm-price-in <in> --llm-price-out <out> && cd ..
uv run --frozen python experiments/e12v2.py --replay experiments/results/e12v2_<run>.jsonl
```
Append the printed tables, with command lines, to `experiments/results/e12v2_run.txt`.

**Questions**
1. The five pass criteria, as printed, and for each FAIL why (read the log).
2. `sort_correction`: does it complete now; how many LLM calls; did the replan-loop rule fire, and was
   the plan it let through right?
3. Did the fixes change anything else (hand contacts, held-out, correction time, LLM calls, cost)?
4. The gate sweep at 0.7-0.9: is 0.8 still the right setting?

**Rules** as in Handoff 3: don't change the harness to improve numbers; fix real bugs in their own
commits and say so.

**Deliverables**, pushed to `claude/first-principles-approach-ow3jo2` (PR #1): logs; a short "e12v2
rerun" section in `context/08-experiment-results.md`; an update to the e12v2 lines of
`docs/spike-outcomes.md`; a plain-language comment on PR #1 answering 1-4 and ending with a
recommendation: move to MuJoCo or not.
