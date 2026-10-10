# Harness and skill library: spikes before hardware

Goal: when a WidowX AI arrives, robojev runs on it the same day. Assume at least one WidowX AI; a
second arm (a leader with handle jaws, or a second follower) is a bonus the design must allow but not
need. The demo's specifics wait.

## What exists

| Piece | Where | State |
|---|---|---|
| v2 harness: events, decisions, plans, replans, safety rules | `experiments/e12v2/core` | Passes closed loop, but against a kinematic toy world, on simulated ticks |
| Motion code for the real arm: workspace box, speed cap, effort trip, park | `src/robojev/arm/real.py` (v1) | Worked on the follower |
| MuJoCo WidowX backend with IK and a rendered wrist camera | `src/robojev/arm/sim.py` (v1) | Worked; cup scene, one arm |
| v1 primitives (move above, grasp, place, ...) | `src/robojev/skills.py` | v1's per-tick vocabulary, not v2's skills |
| Decision backends | Jev; OpenAI Decisions API checked by hand | No Decisions API backend in code |

The gap: v2's harness has never driven a real motion stack. The spikes below close that, mostly in
sim.

## Where each spike runs

- **Here (cloud):** MuJoCo physics works headless; rendering doesn't (no GL). No API keys.
- **Mac:** everything, including rendering and API keys.
- **Arm:** needs the WidowX.

## Spikes

### Harness (no arm)

| # | Question | Pass when | Runs |
|---|---|---|---|
| H1 | Can e12v2's core run on wall-clock time with real async events instead of simulated ticks? | e12v2 scenarios pass with a real clock and an injected-latency mock | Here |
| H2 | Can the harness treat "the arm" as a list of one or two arms with no other change? | One-arm and two-arm toy runs pass; the step schema carries `arm` | Here |
| H3 | Does the OpenAI Decisions API work as a decision backend, and does the 0.8 gate still work on its confidence? | Replay of e12v2 scenarios: answers within a few points of Jev's, gate catches the same share of wrong answers, p95 under ~400 ms | Mac |
| H4 | Can the plan check stop sending good plans back (39% today)? | Under 10% of good plans rejected, with planted bad plans still caught | Mac |
| H5 | Can a transient "wait" stop leaking into the restated task? | No restated task contains a transient command across the scenarios | Mac |
| H6 | Can the harness record episodes in a LeRobot-style format? | One sim episode saved and replayed frame-accurately | Here |

### Skill library (sim)

| # | Question | Pass when | Runs |
|---|---|---|---|
| K1 | A MuJoCo WidowX scene with blocks and a tray, behind one skill interface (`start`, `status`, `hold`, `pause`, `resume`, `stop`, heartbeat) | Scene loads; the interface has one sim and one real implementation stubbed | Here |
| K2 | `move_object` as real motion: approach, grasp, lift, carry, place, release | 19 of 20 sim picks and places at random positions within reach | Here |
| K3 | Interrupting mid-motion: hold at a safe point keeping the grip, pause, resume, re-target to a moved block | Each works from every phase of `move_object` | Here |
| K4 | `stack_on`, `push`, `hand_over` (to a hand position), `survey` | Each 9 of 10 in sim | Here |
| K5 | Two arms in one scene (two WidowX models) with the shared-zone lock | 10 runs of moves in parallel with no arm-arm contact | Here |
| K6 | The skill server over a local socket with the heartbeat watchdog | Killing the harness holds the arm within the timeout, in sim | Here |

### Perception (sim and offline)

| # | Question | Pass when | Runs |
|---|---|---|---|
| P1 | Block positions from the rendered scene camera | Within ~1 cm of MuJoCo ground truth | Mac |
| P2 | Reading block labels (numbers or letters) from a frame: local detector vs OpenAI vision vs the Decisions API with an image | Right on 95% of frames; latency per update recorded | Mac |
| P3 | A hand entering the frame, fast enough to pause | Detected within ~300 ms of entering | Mac |

### Closed loop

| # | Question | Pass when | Runs |
|---|---|---|---|
| L1 | The e12v2 harness driving K2-K4 skills on the MuJoCo WidowX, with ground-truth positions | The e12v2 core scenarios pass in physics sim, not the toy world | Here (mock decisions), Mac (live) |
| L2 | The same with perception (P1-P3) instead of ground truth | Same bar, with noise from real rendering | Mac |

### Day-one hardware kit (prepare now, run on the arm)

| # | Question | Prepared now | Runs |
|---|---|---|---|
| D1 | Does the real arm match sim for the skill interface? | A script running K2's 20 picks and K3's interrupts against the real backend, with the same pass bar | Arm |
| D2 | Calibration: scene camera to arm base | A ChArUco or touch-point routine and its check | Arm |
| D3 | If there's a second arm: is it a follower or a leader, and can it grasp? | W1 and W7 from `docs/widowx-pair.md` as scripts | Arm |
| D4 | First real closed-loop run | A checklist: teleop off, workspace box set, STOP tested, one sorting episode | Arm |

## Order

1. Here, now: K1 → K2 → K3, with H1 and H2 alongside. These are the riskiest unknowns that need no
   hardware.
2. L1 with mock decisions, here; then live on the Mac with H3.
3. Mac: P1-P3, H4, H5, then L2.
4. Last: K4-K6 and H6, then write D1-D4 so day one is scripted.
