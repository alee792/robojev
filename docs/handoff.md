# Handoff: run E11 and E12 against Jev

Paste the block below into a Claude Code session on a machine that has a TypeSafe key (the bay Mac).
It is self-contained.

---

You are picking up robojev v2 spikes that were built in a sandbox with no Jev access. Your job is to
run them against the real Jev API, analyse the results, and push them back to the same branch.

**Setup**
1. `git fetch origin claude/first-principles-approach-ow3jo2 && git checkout claude/first-principles-approach-ow3jo2`
2. Read `docs/v2.md` (the architecture and vocabulary: Planner, Sequencer, Spotter, Listener; Scene,
   Status, Task, Orders, Plan, Knobs, Facts, Utterance, State, Pick, Intent), `docs/assessment.md`
   (what v1 showed and did not), and the E11 and E12 sections of `context/07-experiments-to-run.md`.
3. Confirm `TYPESAFE_API_KEY` is set in the environment or in `.env` at the repo root. Never print it.
4. `uv sync --frozen` at the repo root, then `uv run --frozen pytest -q` (expect 90 passed, 3 skipped).

**Run, in this order. Stop and report if any live run errors on more than 2% of requests.**
```
# E11: one decision after "actually, reverse it"   (~$0.015 per run)
cd experiments
uv run python e11_reorder.py --dry-run
uv run python e11_reorder.py --n-scenes 60                      # tighter cells, ~$0.05
cd ..

# E12: closed-loop text blocks world   (~$0.03 per run, ~$0.10 per sweep)
uv run --frozen python experiments/e12_blocksworld.py --dry-run
uv run --frozen python experiments/e12_blocksworld.py --backend jev --latency-ticks auto
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto --seed 2
uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep --latency-ticks auto --seed 3
uv run --frozen python experiments/e12_blocksworld.py --backend jev --latency-ticks auto --noise
```
Optional, only if `ANTHROPIC_API_KEY` is set (about $0.5-1 each): the same questions answered by
claude-haiku-4-5, for accuracy and latency comparison.
```
cd experiments && uv run --extra baseline python e11_reorder.py --baseline claude && cd ..
uv run --frozen --extra vlm python experiments/e12_blocksworld.py --backend claude --scenario reverse_midway,hand_in_path,ambiguous
```
Save each run's printed tables to `experiments/results/e11_run.txt` and `e12_run.txt` (append, with
the command line above each).

**Questions to answer, with numbers**
1. E11: does Jev do the reordering itself (`raw`), or only map the utterance onto code's answer
   (`solved`)? At which row size does `raw` break, and is `direction` more robust than `target_slot`
   and `next_pick`? Dense vs sparse numbering?
2. E11 and E12: can confidence route mistakes? From the gate tables: accuracy and coverage at each
   threshold. Does `escalate` / `ask_a_smarter_model` fire more on the requests Jev gets wrong?
3. E12: completion, recoveries and violations per scenario for `solved` vs `raw`. Which scenarios fail,
   and what did Jev pick at the failure (read the JSONL log)?
4. E12 ablations: what breaks with `--no-spotter` (hand contacts), and what `--no-listener` costs
   (utterance-to-change ticks, needless escalations on `chatter`)?
5. Latency with real answer timing (`--latency-ticks auto`): p50/p95 per layer, and whether it
   changed outcomes.
6. How far are Jev's answers from the oracle's per question, and do Jev and Haiku make the same mistakes?

**Known weaknesses of the harness, to keep in mind when interpreting**
- E12 `solved` Facts contain code's answer (`next_step`), so a `solved` Sequencer picks a branch
  rather than plans. `raw` is the real test.
- E12's Spotter instructions restate the oracle's thresholds in words, so the Spotter is close to a
  lookup. If it scores near 100%, that is why; say so.
- The code stop-distance for a hand is 3 cm, inside the 5 cm contact zone, on purpose, so the
  no-Spotter ablation can show something. It is a tunable assumption, not a safety claim.
- `--noise` has no tracker, so it overstates Spotter requests.
- A confident wrong `stop` from the Listener stalls a run (nothing resumes it). If this happens live,
  report it as a design finding.

**Rules**
- Do not change the harness to improve the numbers. If you find a harness bug, fix it in its own
  commit, say what it changed, and rerun the affected runs.
- Do not remove or soften the weaknesses above in the write-up.

**Deliverables**, pushed to `claude/first-principles-approach-ow3jo2` (the branch's PR is #1):
1. `experiments/results/e11_reorder.jsonl`, `e12_blocksworld.jsonl`, and the two `*_run.txt` files.
2. New sections "E11" and "E12" in `context/08-experiment-results.md`, in that file's existing style:
   the tables, then takeaways with numbers, each marked measured or inferred.
3. A short "What the spikes say about v2" section at the end of `docs/v2.md`: which layers earned
   their place, which facts code must precompute, whether confidence works as a router, and what to
   change before building crawl.
4. A comment on PR #1 summarising the answers to questions 1-6 in a few lines each.
