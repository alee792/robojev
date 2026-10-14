# The spelling battery

How robojev is measured on the demo it is for (`docs/showcase-demo.md`: spell a word with letter
blocks, the person changes their mind and gets in the way), with the fast decision loop set beside
two controls that have to be beaten. Code: `src/robojev/battery/`; CLI `robojev-battery`; tests
`tests/test_battery.py`.

```
uv run --frozen robojev-battery --n 20 --seed 0 --world stub --compare          # offline, a few minutes
uv run --frozen robojev-battery --world sim --n 4 --seed 0 --compare             # physics, ~10 min
uv run --frozen robojev-battery --world sim --n 10 --seed 1 --decider jev --planner openai --paraphrase   # the Mac
uv run --frozen robojev-battery --world sim --decider decisions --planner openai --episodes <that run>/episodes.json
```

## Two corrections this battery is built on

1. **The task is the demo's.** Earlier evaluations (e12v2, the slice) sorted numbered blocks into a
   tray and judged with an oracle written for that sort. The demo is "spell OPENAI", and a correction
   is a different word. The battery spells words; the blocks carry letters (`ObjectObs.label`, now set
   by the sim and the stub; a cube is "the letter A", "the second letter A"); the tray's slots are the
   row of positions ("position 2, counting from the left"; ids `slot_i` kept).
2. **Outcomes only.** Nothing in a generated episode says what the right reaction to an event is.
   There is no per-event oracle of carry on / hold / replan, and no field of the judge's record
   grades a decision. The brain reacts however it likes; it is scored on what the row reads at the
   end against the latest word asked for, on time, and on cost. A fast loop that holds within 150 ms
   but spells the wrong word loses to one that never holds and gets the word right. That is what
   the controls are for.

The live slice (`docs/live-results.md`) weighted the "hand" and "correction" cases by whether the
decider said the oracle's word; it could not tell a hold that mattered from one that did not. Here
the hand case is judged by the closest approach of the gripper to the hand and whether the word
still got spelled, which is what the audience sees.

## The grammar (`battery/generate.py`)

Seeded; `episodes(n, seed, split)` gives the same `n` episodes every time. One episode:

- The opening ask, a phrasing of "spell {W}" with the demo's word OPENAI (`Grammar.opening`).
- The cubes: the demo's O P E N A A I D, shuffled into the layout, plus 0-2 spare cubes drawn from
  N E P A, and sometimes a parked cube (not on the table yet) for a `letter_added` event.
- Six positions in a row.
- 0-3 events (weights 0.2 / 0.4 / 0.3 / 0.1), kinds drawn with weights and caps (at most two
  `new_word`, one of anything else), ordered by their nominal time; with probability 0.1 a STOP as
  the last event.
- The expected END STATE: the latest word asked for. Every word a `new_word` asks for is spellable
  from the cubes on the table at that point, and a taken cube comes back whenever no spare carries
  its letter, so the end state is always reachable.

| event | what happens | when |
|---|---|---|
| `new_word` | a different word from the 20 the demo's letters spell (PANDA, PAID, IDEA, OPEN, ...) | mid-carry after k letters are in place, or at a time |
| `other_letter` | "use the other A" / "take the other one instead": a letter of the word that has a twin free on the table; the word is unchanged, so the outcome score is too. The judge records how long the arm's target took to change (a `start()` for another cube) and whether a planner was asked for it (`planner_called`); the fast loop's in-plan fix `swap_target` answers it without one, the controls go round the planner | while the arm reaches for that letter, not yet grasped, after k letters are in place |
| `letter_taken` | a cube the word still needs goes off the table; back 1-2 picks later if it must (`if_needed`) or anyway (`always`) | after k letters are in place |
| `letter_added` | a parked cube is put on the table | after k letters, or at a time |
| `letter_nudged` | a cube is moved 3 cm where it lies | after k letters |
| `letter_placed` | the person puts the cube for position j into position j | after k letters |
| `chatter` | a line that asks for nothing | mid-carry or at a time |
| `wait_go_on` | a wait, then a go-on 2 s later | mid-carry |
| `hand_near` | a hand 15 cm from the gripper for 1.5 s | mid-carry |
| `stop` | the STOP button | mid-carry, last |

Times are in picks (one pick_and_place's worth: 0.8 s on the stub, 7.5 s in the sim), so one
episode runs on both worlds. Conditions are judged from the true world with a fallback time, so an
event whose condition never comes still fires.

**Phrasings** (`battery/phrasings.py`): every event that is said or that a person might say
something about (the opening, `new_word`, `chatter`, `wait`, `go_on`, the remark beside each world
event, the letter's return, the reply to a question) has at least four `train` and four `held_out`
phrasings, disjoint. Offline runs use `train`; the Mac uses `held_out`, and `--paraphrase` rewrites
every line with a model first (`battery/paraphrase.py`; offline the hook is a no-op). Phrasing
counts: opening 4 + 4; new_word 5 + 5; other_letter 5 + 5; chatter 5 + 5; wait 5 + 5; go_on 5 + 5; six
remark types at 4 + 4 each; reply 4 + 4: 57 train, 57 held-out.

A world event carries a remark with probability 0.5 ("I'm borrowing this one"); a remark asks for
nothing, the scene change is the event.

## The judge (`battery/judge.py`)

From the server's own `world()` after the brain returns, and the perception layer's log of tool calls
and arm modes with wall times. Per episode:

| field | what |
|---|---|
| `word`, `row`, `prefix`, `span`, `score`, `exact` | the latest word asked; the labels in the positions left to right; how many read the word from the left; how far the row extends; `score = prefix / span` (half a word is 0.5; the word plus a stray cube after it is 5/6; an empty row 0); exact = 1.0 with every gripper empty |
| `outcome`, `wall_s`, `done_s` | the brain's outcome (done, timeout, gave_up, stopped) and time |
| `events[]` | per thing the person did: kind, what, when, the phrasing; `time_to_change_s` and `change`: the first visible change in the robot's behaviour after it (a hold, pause, resume, retarget, a new start, STOP, or the server reporting an arm holding, paused, tripped or stopped), descriptively (for `other_letter`: the arm's target changing, a start for another cube); `planner_called`: a planner request between the event and that change; `finished_right` |
| `planner_calls`, `replans`, `failed_plans`, `decisions`, `decision_ms_p50`, steps started/done/failed | cost |
| `drops` | spells a cube was in no gripper and 2 cm above the table for over 0.2 s (physics) |
| `hand_seen`, `hand_min_dist_m`, `hand_contact` | closest approach of the gripper to a reported hand; contact under 5 cm |
| `stop` | when STOP was in the script: time to park, nothing sent after, every cube held or set down, drops |
| `asks` | questions the brain put to the person |

The table (`run.table`) per decider: mean score and finish rate over episodes without a STOP, median
time to first change per event kind, planner calls, replans, decisions and latency per episode,
drops, contacts over episodes with a hand, STOPs ok, median wall time.

## The deciders (`battery/models.py`)

- **mock** (`battery/decider.py`): the OFFLINE STAND-IN for the decision model, behind the brain's
  own `JevDecider` (request building, `combine()`, the gates), answering from GROUND TRUTH: the
  script's account of each line and the server's true world. It says what a perfect decider would,
  so the offline numbers are a ceiling for the fast loop, not evidence about any model. It sleeps
  150 ms, Jev's measured median in the loop.
- **rules** (`battery/controls.py`, `RulesOnly`): a code decider with keyword rules written ONLY
  from the `train` phrasings (the module docstring lists them: U1-U7 for text, H1-H4 hands, M1-M5
  cubes, S1-S4 steps, P1-P3). No model, no ground truth, 1 ms. Held-out phrasings and paraphrases
  are what it has never seen.
- **always_llm** (`AlwaysLLM`): no decision model; every event but a step finishing or a plan
  arriving goes to the planner and the arm carries on meanwhile. The brain's rules still apply
  (STOP, models only pause, the replan-loop limit, the reading guard). It shows what the slow loop
  alone costs: waits that pause nothing, hands nobody stops for, corrections answered seconds later
  (and with the real planner, several seconds).
- **jev**, **decisions**: the live deciders (`brain/backends`), the Mac only.

Planners: **mock** (`battery/spell.py`, `SpellLLM`, offline only) writes catalog plans from the
request alone: the word is the newest uppercase word in the user's messages, the letters in view
and their labels come from the request's world; left to right, a cube already in its position stays,
a cube in a wrong position is moved straight to a position that wants it if free, else to the
"table" (a free spot, `scene.FreeSpots`), the held cube is put down first, a letter with no cube in
view is left out and the reading says so. **openai**: the real planner.

## Two arms

`--arms 2` runs the stub pair (`stub.widowx_pair`): letters in arm_0's half, positions in arm_1's,
the shared zone between; the one-arm plan goes through `mocks.give_arms` (hand-over through a zone
slot), one mock decider per arm. The physics pair is not wired: the pair's halves are shallow at
the default spacing and a 6-position row does not fit arm_1's side line at the 8 cm pitch without a
layout study, which is a separate job.

## Running

Offline, here or on any machine: `robojev-battery --world stub --compare` (about 6 s an episode and
decider), `--world sim` (about 60 s an episode; the stub and the sim run the same episodes). Tests:
`uv run --frozen --extra mcp python -m pytest -q tests/test_battery.py` (the two physics episodes
need `TROSSEN_ARM_MUJOCO_DIR`).

Live, on the Mac (`OPENAI_API_KEY`, `TYPESAFE_API_KEY`): `robojev-battery --world sim --n 10 --seed 1
--decider jev --planner openai --paraphrase`, then the others on THAT run's episodes:
`--decider decisions --planner openai --episodes <jev run dir>/episodes.json`, and the controls with
the real planner, `--decider rules` and `--decider always_llm`, the same way. The seed alone does not
make the tables share their episodes once `--paraphrase` is on: each paraphrase is a fresh model call,
so four `--paraphrase` runs on seed 1 meet four different wordings. Every run writes the episodes it
ran to `episodes.json` in its directory. Held-out phrasings are the default whenever anything live
runs. Episode files land under `experiments/results/battery/<stamp>-sim-<decider>/`
(`ep_<seed>_<index>_<decider>.jsonl`: the episode, what the director did and when, the brain's log,
decisions and planner records, the judge's record; `summary_<decider>.json`; `episodes.json`).
`experiments/live_battery_report.py DIR ...` reads the brain's log beside the table (plans sent
back and why, escalations, decision latency, `reading_kept`). The pytest route
(`ROBOJEV_DECIDER=jev ROBOJEV_PLANNER=openai uv run --frozen --extra mcp python -m pytest -q -s
tests/test_battery.py -k physics`) runs the two physics episodes live and writes to
`experiments/results/live/`.

## Results so far (offline, 2026-10-12, `experiments/results/battery/`)

The mock decider is a ceiling (it answers from ground truth), so the offline question is whether the
controls fall short of it, and where. 20 episodes of seed 0 on the stub (`stub-seed0-compare/`):

| | mock | rules | always_llm |
|---|---|---|---|
| mean outcome score (19 episodes; 1 has a STOP) | 0.974 | 0.939 | 0.974 |
| finish rate (exact row, done) | 0.95 | 0.90 | 0.95 |
| median s to first change: new_word / wait / go-on | 0.15 / 0.15 / 0.15 | 0.00 / 0.00 / 0.00 | 0.38 / 0.39 / 0.14 |
| median s to first change: hand_near | 0.15 | 0.00 | 0.39 |
| planner calls / replans per episode | 1.70 / 0.70 | 2.00 / 1.00 | 3.75 / 2.75 |
| decisions per episode, latency p50 | 9.4, 150 ms | 9.8, 1 ms | 11.2, 1 ms |
| hand contacts / episodes with a hand | 0 / 3 | 0 / 3 | 1 / 3 |
| STOP ok | 1 / 1 | 1 / 1 | 1 / 1 |
| drops | 0 | 0 | 0 |

What it says: on train phrasings with a 0.3 s mock planner every decider spells the word in all but
the one episode no one can (a letter taken away and returned after the arm had given up on it,
PAID left as "PA_D"); the rules miss one more (a taken E after a hand, "OP_NAI"). The always-LLM
control reaches the same score at twice the planner calls and reacts in 0.4 s instead of 0.15 s,
and the one hand it meets it does not stop for (1 contact in 3). The stub cannot drop anything. The
gaps that matter are the live planner's seconds and the held-out wording, which only the Mac runs
show.

On the physics sim, the first 4 episodes of seed 0 (`sim-seed0-compare/`; the same episodes as the
stub's first four, about 48 s each), all three deciders spell every word with no drops. The timing
column is where they part: median seconds to the first visible change after a `wait` / the go-on /
a `new_word`: mock 0.14 / 0.15 / 0.15, rules 0.00 / 0.00 / 0.00, always_llm 2.37 / 2.92 / 0.24; the
always-LLM arm pauses only when its planner has answered and a step boundary comes. Planner calls
per episode 1.25 / 1.50 / 2.75. The two physics episodes in `tests/test_battery.py` (seed 1: a
letter taken away, and OPENAI corrected to PINE with the person placing the N) score 1.0 in 50 and
63 s with the mock decider.

## What the spelling task exposed (offline, 2026-10-12)

- **A cube taken off the table never becomes a scene event.** e12v2's change detector
  (`brain/core/changes.py`) counts an object missing only on the first frame it is gone (it compares
  against the previous frame, which no longer has it), so its "gone for 3 ticks" never triggers and
  `object_gone` is never reported. The brain learns of a taken letter only when the step for it
  fails to start ("unreachable" or "not in view"), a pick later; it then replans to a spare in one
  planner call (sim: 0.19 s with the mock planner). The demo's beat 3 ("the arm re-targets to the
  spare or asks for it back") works through that failure, not through perception. A core fix is out
  of this battery's scope; the mock decider's `object_gone` rule is there for when it lands.
- **A correction right after a grasp plans without the held cube.** The replan request carries the
  brain's last world snapshot (up to 200 ms old). When a correction lands just as a cube is grasped,
  the planner sees it on the table, writes a plan that ignores it, the brain's merge rule keeps the
  running step first, and the plan misfits ("slot_0 is still occupied"); the automatic retry with
  the fresh world fixes it, at the cost of one planner call and ~200 ms more holding.
- **Duplicate letters work through ids, not names.** Two A's are "the letter A" and "the second
  letter A"; plans and done conditions name ids. The planner may pick either; the judge reads labels.
- **"Reuse what is already placed" needs the planner to see the row.** The mock's first version
  cleared a wrongly placed P to the table and fetched another P; preferring cubes that must move
  anyway for the positions that want them saves a pick (OPENAI → PINE: P moves from position 1 to 0
  after O is cleared).
- **A nearly finished step is dropped on a replan.** A new plan that has nothing for the running
  step's cube (because it is already placed and the arm is retreating) drops that step; the brain
  holds the arm mid-retreat and starts the next step, which cancels the retreat. Harmless on the
  sim; worth knowing when reading `time_to_change` after a `letter_placed`.
- **The sim needed a place to put a cube down anywhere.** The scene had slots only; clearing a
  wrong letter needs "table". `scene.FreeSpots` adds one (the candidate farthest from every cube and
  slot, at least 7.5 cm clear), opt-in per scene so K1's scenes and tests are unchanged.
