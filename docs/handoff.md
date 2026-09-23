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
