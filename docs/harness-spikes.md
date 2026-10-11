# Harness and skill library: spikes before hardware

Goal: when a WidowX AI arrives, robojev runs on it the same day. Assume at least one WidowX AI; a
second arm (a leader with handle jaws, or a second follower) is a bonus the design must allow but not
need. The demo's specifics wait.

## What exists

| Piece | Where | State |
|---|---|---|
| v2 harness: events, decisions, plans, replans, safety rules | `experiments/e12v2/core` (since promoted: `src/robojev/brain`) | Passes closed loop, but against a kinematic toy world, on simulated ticks |
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
| H1 ✔ | Can e12v2's core run on wall-clock time with real async events instead of simulated ticks? | Done 2026-10-10: a new asyncio loop around e12v2's pure functions (now `src/robojev/brain/`), 34 tests with ~150 ms mock latency | Here |
| H2 | Can the harness treat "the arm" as a list of one or two arms with no other change? | One-arm and two-arm toy runs pass; the step schema carries `arm` | Here |
| H3 | Does the OpenAI Decisions API work as a decision backend, and does the 0.8 gate still work on its confidence? | Replay of e12v2 scenarios: answers within a few points of Jev's, gate catches the same share of wrong answers, p95 under ~400 ms | Mac |
| H4 | Can the plan check stop sending good plans back (39% today)? | Under 10% of good plans rejected, with planted bad plans still caught | Mac |
| H5 | Can a transient "wait" stop leaking into the restated task? | No restated task contains a transient command across the scenarios | Mac |
| H6 ✔ | Can the harness record episodes in a LeRobot-style format? | Done 2026-10-10: `recorder.py` (per-tick JSONL in the server) + `lerobot_export.py` (LeRobot v2.1 layout, npz columns until pyarrow is in the venv, per-episode tasks and events, frame-accurate replay check). The committed K2 recording exports to 20 episodes / 7014 frames / 1.1 MB, byte-identical on re-export | Here |
| M1 ✔ | Is the brain hardware-agnostic? | Done: the same brain runs on the stub's WidowX-like and push-only manifests and on the physics sim; the planner schema and robot description are generated from each, a non-conforming manifest is refused | Here |

### Skill library (sim)

| # | Question | Pass when | Runs |
|---|---|---|---|
| K1 ✔ | A MuJoCo WidowX scene with blocks and a tray, behind one skill interface (`start`, `status`, `hold`, `pause`, `resume`, `stop`, heartbeat) | Done: `scene.py`, `protocol.py`; the policies run off an `ArmIO` surface a real backend can implement | Here |
| K2 ✔ | `pick_and_place` as real motion: approach, grasp, lift, carry, place, release | Done: 80/80 over seeds 0-3 (`trials.py`) | Here |
| K3 ✔ | Interrupting mid-motion: hold at a safe point keeping the grip, pause, resume, re-target to a moved block | Done inside K2: hold / pause / resume / retarget from every phase, tested | Here |
| K4 ✔ | `stack_on`, `push`, `hand_over` (to a hand position), `survey` | Done 2026-10-10: stack_on 80/80, push 80/80 (seeds 0-3), hand_over tested with an injected hand; begin-from-held and hold/resume mid-skill for each | Here |
| K5 | Two arms in one scene (two WidowX models) with the shared-zone lock | 10 runs of moves in parallel with no arm-arm contact | Here |
| K6 ✔ | The robot server as a real MCP server (stdio or HTTP) with the heartbeat watchdog | Done 2026-10-10 (`mcp_server.py`, `mcp_client.py`): the brain's 3-block sort passes over a stdio subprocess unchanged; event delivery 1.6 ms p50 / 2.5 ms p95, heartbeat 1.6 ms, world 2.1 ms | Here |

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
| L1 ✔ (mock decisions) | The e12v2 harness driving K2-K4 skills on the MuJoCo WidowX, with ground-truth positions | Done 2026-10-10 (`tests/test_l1_slice.py`, `experiments/results/l1_slice.txt`): 3-block sort 22.9 s; correction mid-carry → hold in 0.10 s, order reversed, no drops; hand mid-carry → pause in 0.11 s, 12 cm clearance, resume 0.17 s after it leaves; STOP mid-carry → parked in 2.5 s, block held. Live decisions on the Mac remain | Here (mock decisions), Mac (live) |
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
arrives at K6. Code lived in `experiments/skills_sim/` and was promoted into `src/robojev` once L1
passed (2026-10-10; the layout is in `docs/architecture.md`, choice 7). H1 is a small new asyncio loop around e12v2's pure functions (combine, gates, plan), not
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
  a missed heartbeat and every failure code are exercised by `tests/test_robots_widowx_sim.py`.
- **First brain-on-physics run (H1 × K2, 2026-10-10).** Placing block 2 jostled block 1 in the
  next slot; its `where` flickered slot → table → slot for ~200 ms, e12v2's change detector called
  that "moved by someone else", the decider re-queued it, and re-picking from the crowded tray
  stalled. Two fixes: the brain holds an object change back until it has lasted 0.4 s (hands still
  pass through at once), and the tray's slot pitch is 8 cm (the gripper opens to block + 3 cm; at
  6 cm a finger landed on the neighbour). Result: 2 blocks in 14.8 s, no false scene events.
  Follow-up for the skills: pre-grasp opening should shrink to the gap to the nearest neighbour,
  so picking from a crowded tray works at any pitch.
- **The slice passes end to end (L1, 2026-10-10).** Brain + physics server + mock decisions, all
  four scenarios first time, no behavioural bug found. Two things it taught: (1) the 0.5 s
  hand-to-pause bar holds only because perception *notifies* the brain (0.11 s); on the brain's
  200 ms poll alone it is 0.3 s typical and ~0.5 s worst case, so the perception process must push
  scene-change events, not just update the world resource. (2) A correction replaces the running
  skill by a new `start()` on a busy arm; it works because `pick_and_place` notices its object is
  already between the fingers and begins at "lift". That is a requirement on every grasping skill
  and should be stated in the catalog.
- **Pushing (K4).** The closed gripper never pushes with its fingertips: the pad boxes stand proud
  of the tips on every side, so contact lands ~1.9 cm up a 4 cm cube. At the scene's friction of
  1.0 the cube rolled over; at 0.5 (wood on wood, the more honest number) it slides with 0.3° of
  tilt. A blocked push doesn't stop the arm under position control (joints creep ~1.5 mm per
  0.2 s), so `blocked` is detected from the gap to where the slide should be, not from progress.
- **MCP as the transport (K6).** Viable with a wide margin (table above). What mcp 2.x made
  awkward for a long-running, stateful, event-emitting server: no connection-opened hook (sessions
  are per request, so pushing events means remembering the last request's session); a closed
  notification vocabulary (events ride a custom `notifications/robot/event`, which the client must
  know to bind); the in-process client drops server notifications (tests use memory streams); tool
  errors carry text only (the `code: text` convention survives, a code field doesn't); the
  high-level server derives schemas from Python signatures, so manifest-driven tools need the
  low-level one. The transport cost is not the problem; the SDK's request/response shape is.
- **Sim grasp numbers are logic-checks only.** The sim actuator allows 400 N against a real grip of
  tens of N, and pad friction here is hand-tuned. K2's 19/20 proves phases and geometry; real grasp
  reliability is D1's bar, with the same script pointed at the real backend.

## Before the recorder runs on real hardware (from H6)

- Record the exact snapshot the policy was given, with its capture time; today the observation
  is re-read after the tick (identical in sim, newer than the policy saw on hardware).
- Full-precision monotonic timestamps, both sensor capture and tick; today rounded to 0.1 ms.
- A header line: manifest, catalog version, recorder schema version, fps, units, base frame, a
  run id (skill ids like `sk1` are unique only per server). Record skill name and args at
  `start()`, not inferred from the done event.
- Joint states and joint commands, and camera frames, are not recorded yet; object poses carry
  no source, timestamp or confidence. Flush frames as they're written, not at close.
- The action a learned policy must output is 6 values: goal xyz, yaw, gripper, speed.

## Follow-ups from the first slice

Small, recorded so they aren't lost; none blocks the next spike.

- ~~`hold` is both a control tool and a standard skill~~: settled at K6 without a rename; the wire
  names are `hold_arm`, `pause_arm`, `resume_arm`, `retarget_skill`, so skills keep catalog names.
- Place ids aren't standardised (`slot_0..` in the sim, `tray_slot_1..` in the stub). Pick one in the
  catalog.
- The stub's `ArmObs.skill` is still `None`; the brain doesn't read the field yet. Wire both when
  reconnect moves from inference to `status()`.
- A shared `catalog.check_args(spec, args)` would replace the stub's and the brain's separate
  argument checks.
- Decision text inside e12v2 still says `move_object`; fixing it means editing e12v2, which we're
  retiring, so it waits for promotion into `src/robojev`.
- Extension skills are advertised but not plannable: e12v2's validator can't model their effects.
- The pre-grasp opening should adapt to neighbours (above).
- `hand_over` has no end state in the catalog: the sim's released block falls to the table
  (`where` "table"), the stub reports "person". Decide what a successful hand-over's world looks
  like and give the skill an optional `place` for where the person is.
- No way to advertise a maximum stack height; the sim refuses a base that would put the carry above
  travel height with `unreachable`.
- Across processes, `RobotEvent.t` and `WorldState.t` sit on the server's private clock; expose
  the epoch so a client can place events on its own clock without calibrating.
