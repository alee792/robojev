# Spike outcomes

What each spike found and what it changed in `docs/v2.md`. Full tables and logs:
`context/08-experiment-results.md` and `experiments/results/`.

| Spike | Question | Status |
|---|---|---|
| E1-E10 | How fast and stable is Jev; how good with positions and numbers? | Done |
| E11 | After "actually, reverse it", can Jev work out the move itself? | Done (2026-09-23) |
| E12 | Does a layered design survive a whole task with a person interfering? | Done, against an earlier design |
| E11 replay | Does low confidence flag Jev's mistakes? | Done (offline) |
| E13 | Can a fast LLM prepare variants up front for Jev to pick from, compared with sending every correction to the LLM? | Done (2026-09-24) |
| E13 fast-model probe | Is any small OpenAI model much faster at replanning? | Done (2026-09-24) |

## Findings

- **Jev reads corrections almost perfectly and can't work out moves (E11).** "Actually, reverse it"
  and similar were understood 719/720 times; the move itself was right only ~75% of the time, worse
  with arbitrary block numbers. Given the answer for each reading, it chose right 301/303. So Jev
  decides what a correction means and where it goes; it does not compute arrangements.
- **Confidence flags mistakes; asking Jev whether it is sure doesn't (E11, E11 replay).** Trusting
  Jev only at 70% confidence or more caught 71 of its 72 mistakes. A self-assessment question did no
  better than chance (AUROC 0.36-0.75 vs 0.71-0.93 for the weakest confidence).
- **State facts literally (E12).** Every E12 `raw` run stuck at the first block because nothing said
  "lowered at the goal, not yet released".
- **Positions by role, not number (E11).** Arbitrary block numbers halved slot accuracy; Jev's
  direction contradicted its own slot in 24/103 answers.
- **The reflex group earns its place (E12).** With the Spotter off, the simulated hand was touched in
  3/3 runs; with it on, 0/4. E12's Spotter restated the oracle's thresholds, so this is close to a
  lookup.
- **Decisions are fast enough for events (E12).** 150/260 ms p50/p95.
- **Prepared variants don't happen (E13).** Asked for a plan with optional variants,
  `gpt-5.6-terra` and `gpt-6-luna` wrote valid plans with the right arrangement 28/28 times but
  prepared a variant for only 4-7% of the corrections one could have covered. v2 dropped variants.
- **Jev routes corrections well (E13).** 135-138/140 routed right; every correction it kept was
  right (58/58). The system matched sending every correction to the LLM (137/140) and was ~20× faster
  only for chatter and pause.
- **Replanning takes seconds, whatever the model (E13, probe).** 2.7-4.1 s median, 6-7 s p95, with
  11-30 s outliers. Six small OpenAI models all landed at 2.2-3.7 s median. `gpt-6-luna` at low effort
  was accurate (35/35) and cheapest (~$0.0003 per replan): the default fast LLM. `gpt-4.1-mini` was the
  only model that prepared variants unprompted, letting Jev keep 18/30 corrections, at 87% accuracy.

- **The v2 loop works closed-loop except after a correction (e12v2).** Live Jev + `gpt-6-luna`, 3
  seeds: `jev` completed 24/27 core and 9/9 held-out episodes with 0 hand contacts. All three misses
  are `sort_correction`: after the replan, Jev is shown the original task and the correction
  separately, can't reconcile them, and keeps routing new plans back to the LLM until the 16-call cap.
  Keyword rules and always-LLM completed 27/27. The state should carry the task as corrected, and a
  replan loop needs the loop detector.
- **Right-now earns its place; the router is the weak group (e12v2).** No hand contacts with
  right-now; 3/3 without it and with always-LLM. Corrections change behaviour in 0.1-0.2 s vs 7.7 s
  (always-LLM). Route agrees with the oracle 88% (right-now 92%, in-plan fix 98%); removing the router
  raised completion to 26/27. Jev beat the keyword rules only on held-out phrasing (0.1 s vs 8.4 s).
- **Gate 0.8 (e12v2).** 0.7 caught 89% of wrong right-now / fix answers; 0.8 caught 98% for 21%
  escalation (vs 15%). Prompt caching worked (99% of plan input, 59% of replan input). Replans:
  3.0 s median, 7.3 s p95.

## What the E12 update must do to test v2

1. One decision request per event, with the three question groups, for every source (user text,
   scene change, step finished or failed, new plan).
2. Plans as skill-call steps, done conditions and constraints; replan with an LLM on escalation (a
   mock LLM offline), from the current world state, with no prepared variants.
3. Progress checks as one question per done condition; code compares.
4. Escalate on the weakest relevant confidence, with a threshold per route.
5. The Spotter group handles any change the robot didn't cause, with instructions that don't restate
   the oracle's thresholds.
6. A literal "where the arm is in the plan step" fact; rerun `raw`.
7. A loop detector.
8. Held-out tasks (a stack, a handover, a standing rule) reported separately.
