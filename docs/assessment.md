# Honest assessment

robojev is a one-day experiment (2026-09-17). It works, on real hardware, and it is also more
overfitted to one paper cup on one table than the README's result list suggests. This file is the
counterweight: what is actually demonstrated, what is weak, and what would have to change before
any of it should be believed in general.

## What is genuinely demonstrated

**A small text-only model can sit inside a 10 Hz control loop.** Measured on the real arm, over
two runs of about 1,080 ticks each:

| | demo2 | real8 |
|---|---|---|
| requests sent | 135 | 210 |
| ticks | 1072 | 1085 |
| latency p50 | 144 ms | 143 ms |
| latency p95 | 258 ms | 250 ms |

The change detector suppresses 81 to 87% of requests. Answers arrive one to three ticks after the
state they describe, are tagged, and stale or out-of-order ones are dropped. The loop keeps acting
on its last good answers while Jev is late, and walks a hold-then-rise ladder when Jev is silent.
This part is ordinary engineering and it transfers.

**Jev is choosing, not being funnelled.** The worry with a primitive library is that preconditions
narrow the options until only one is left and the model rubber-stamps it. Measured across four runs,
the share of ticks offering two or more genuine (non-safety) primitives was 97 to 99.5%, usually
three or four. The choice was real.

**Three things code could not have done.** Grounding "the paper cup" or "the object breaking the
rule" to a specific track; turning "on the mat" or "to the left of the phone" into a relation code
can resolve; and deciding that the moving thing 7 cm away is a hand to evade while the cup 7 cm the
other way is a thing to approach. That last one is the interesting one: it is a semantic
distinction, not a distance threshold.

**It completes the task on real hardware.** Run real13 did four grasp-to-release cycles in 234 s
unattended, carrying 12 to 32 s each. Run demo2 did one with the force-controlled grip, the fingers
stopping at 30 mm per side on the cup instead of crushing it.

**In sim, the dynamic scene works end to end.** In the intruder scenario a hand slides between the
gripper and the cup mid-carry: the first evasive command comes 1.7 s after the intrusion and the
task resumes 9.8 s after it, completing normally.

## What is weak

**Four of the seven channels barely earn their place.** `target`, `place` and `evade` do real work.
`orders_violated` almost never fires since its gate went to 0.9 (it was demoted because it froze the
arm in situations `evade` already handled). `task_done` is marginal. `speed` is near-constant.
The three paraphrases of `next` with a 2-of-3 majority are more ceremony than signal.

**The pick-and-place sequence itself is scriptable.** A state machine over the same primitive
library would produce the same order of operations. What a state machine could not do is take a new
rule in English at run time, which is the claim worth making and the one the demo does not foreground.

**A visitor watching the arm move a cup sees 2010-era robotics.** Nothing on screen shows the part
that is new.

## What is overfitted

**Perception constants.** The depth-only pipeline accumulated roughly a dozen hand-tuned numbers in
a day, each fixing a real observed failure, together fitted to this table, camera and cup: drop the
lower 40% of the wrist image and everything closer than 11 cm (the finger pads sit inside the D405's
minimum range), a tool-axis cylinder for the fingers and carriages, a shoulder-to-EE line for fixed
cameras, oblique views placed by near edge plus across-width, no track motion from views within
15 cm of the gripper, slivers under 2 cm ignored, mats found by darkness under 60 with a fill-ratio
test. This is the wrong tool being patched. `perception/segment.py` (FastSAM, about 50 ms on MPS)
is the intended replacement and would subsume perhaps half of these, but it has never run in a live
control loop.

**Grasp geometry.** Tips 2 cm above the table, 2.5 cm past the object's centre, wrist pitch 0.5 on
approach and 0.25 to 0.35 on the advance, 6 N of grip force. All of it fitted to a 7 to 9 cm paper
cup against an 8 cm pad opening. A different object needs most of it re-derived.

**The prompt.** This is the worst one, because it is invisible in the logs. The question rules in
`questions/v1.py` contain "a paper cup is white or light coloured; a cardboard box is brown, orange
or tan", added because Jev picked the wrong object in a sim run of the mat routine. That is fitting
language to one scene. It is marked with a comment in the source and should be removed before any
claim is made about generalisation.

## What was never tested

- Jev-driven evade against an actual human hand on the real arm. Only in sim.
- The scene camera in the control loop. It streams, but calibration never produced a fit good
  enough to save, so every real run used the wrist camera alone.
- The VLM naming tier against a real key. Stub only.
- The segmenter in a live run. Off by default (`--segment`).
- The mat routine on real hardware. It reached a working sim run only at the end.

## What would make this a real demonstration

Stop tuning perception and put the weight on the part a state machine cannot do: the task arriving
in language at run time. Type "keep the cup on the mat" into the dashboard, watch it comply, then
type "actually keep it off the mat", then "don't touch the white one at all", with no restart and no
code edit. Run the same scene against a scripted baseline and show the baseline cannot accept the
second sentence. That experiment is mostly built and was not run.

## Process notes

Findings were driven by real failures, and the run logs in `runs/` (gitignored) are the evidence
trail; `scripts/analyze.py` computes the pick-change, evade-latency and resumption numbers. Several
numbers quoted in the README come from single runs, not distributions. The arm overheated once
(a shoulder rotor reached 96 C against a 95 C limit) which is why the thermal governor exists; the
gripper crushed several cups before the switch from position to force control.
