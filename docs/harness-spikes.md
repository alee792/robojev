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
| M1 | Is the brain hardware-agnostic? | The same brain, unchanged, runs against two manifests (the sim WidowX and a stub push-only arm), with the planner schema generated from each | Here |

### Skill library (sim)

| # | Question | Pass when | Runs |
|---|---|---|---|
| K1 | A MuJoCo WidowX scene with blocks and a tray, behind one skill interface (`start`, `status`, `hold`, `pause`, `resume`, `stop`, heartbeat) | Scene loads; the interface has one sim and one real implementation stubbed | Here |
| K2 | `pick_and_place` as real motion: approach, grasp, lift, carry, place, release | 19 of 20 sim picks and places at random positions within reach | Here |
| K3 | Interrupting mid-motion: hold at a safe point keeping the grip, pause, resume, re-target to a moved block | Each works from every phase of `pick_and_place` | Here |
| K4 | `stack_on`, `push`, `hand_over` (to a hand position), `survey` | Each 9 of 10 in sim | Here |
| K5 | Two arms in one scene (two WidowX models) with the shared-zone lock | 10 runs of moves in parallel with no arm-arm contact | Here |
| K6 | The robot server as a real MCP server (stdio or HTTP) with the heartbeat watchdog | The brain runs over the transport unchanged; killing it holds the arm within the timeout, in sim | Here |

### Perception (sim and offline)

| # | Question | Pass when | Runs |
|---|---|---|---|
| P1 | Block positions from the rendered scene camera | Within ~1 cm of MuJoCo ground truth | Mac |
| P2 | Reading block labels (numbers or letters) from a frame: local detector vs OpenAI vision vs the Decisions API with an image | Right on 95% of frames; latency per update recorded | Mac |
| P3 | A hand entering the frame, fast enough to pause | Detected within ~300 ms of entering | Mac |
| V1 | The Decisions API with a camera frame: latency, and accuracy on hand-near-gripper, grasp-succeeded and block-knocked-over, against text-only | Accuracy at least matches text-only on each; latency recorded (sets the visual loop's rate) | Mac |
| V2 | The planner with an annotated frame (object ids drawn on): does it catch planted disagreements between the written state and the picture? | 9 of 10 planted disagreements caught, no false alarms on 10 clean frames | Mac |

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

## Order (revised after review, 2026-10-10)

Build a thin end-to-end slice first, then widen — interface mismatches between harness and skills
should surface in days, not after the skill library is "done".

1. **The slice:** the robot protocol (`protocol.py`, MCP-shaped), K1, a basic `pick_and_place`, H1,
   and H6's recorder from the first version of the
   skill server (every tuning run is training data; retrofitting recording gets skipped).
2. **L1 on one scenario** (sort three blocks, one correction), mock decisions here, then live on the
   Mac with H3.
3. **Widen:** K2's 20-trial bar, K3's interrupts from every phase, more scenarios, H2, H4, H5.
4. **Mac:** P1-P3 before L2 — ground truth hides block identity and occlusion, the hardest real
   perception problems; don't let these slip to the end.
5. **Last:** K4-K6, then D1-D4 so day one is scripted.

During the spikes all three "processes" run in one process with the interfaces enforced; the socket
arrives at K6. Code lives in `experiments/skills_sim/` and is promoted into `src/robojev` only after
L1 passes. H1 is a small new asyncio loop around e12v2's pure functions (combine, gates, plan), not
a port of its tick loop.

## Findings so far

- **Grasp geometry (K1 probe, corrected by K2, 2026-10-10).** At v1's 75° pitch a 4 cm cube can't
  be gripped without the fingertips in the table; v1 never saw this because its sim snapped objects
  to the gripper and its tuning was for side-grasping an 11 cm cup. Fix: grasp straight down (90°),
  fingertips 0.5 cm above the table. K2 then found the finger collision boxes are tapered (~7°), so
  a 4 cm cube is held by a fingertip pinch on its lower 1.5 cm, not by the pads; it holds through a
  0.15 m/s carry. Straight-down reach is better than v1's note: position error 0 out to x = 0.46 at
  grasp height, 0.42 at z = 0.10, 0.38 at z = 0.14. D1 must confirm the real arm streams a 90° pitch
  (v1 verified 75° only) and loosen the vertical tolerances: every sim body has gravity
  compensation, so steady-state error is ~0 here and will not be on the real arm.
- **Gripper mapping.** v1's sim "closed" setting leaves a 4.8 cm gap — built for the fake grasp.
  K2 commands the joint's true 0-0.044 range.
- **K2 result (2026-10-10).** `pick_and_place` 80/80 over seeds 0-3 (`trials.py`), 7.3 s sim per
  pick. Parameters: open to width + 3 cm, close to width − 8 mm (4 N per side), yaw = block yaw mod
  90° nearest the wrist's zero-roll direction, 0.15 m/s travel / 0.10 vertical / 0.05 for the last
  2 cm, travel_z 0.10. Grasp detected from both finger sides touching the same block with the width
  within 2.5 mm of its size. Hold / pause / resume / retarget work from every phase; a lag trip,
  a missed heartbeat and every failure code are exercised by `tests/test_skills_sim.py`.
- **Sim grasp numbers are logic-checks only.** The sim actuator allows 400 N against a real grip of
  tens of N, and pad friction here is hand-tuned. K2's 19/20 proves phases and geometry; real grasp
  reliability is D1's bar, with the same script pointed at the real backend.
