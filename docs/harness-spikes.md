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
| H2 ✔ | Can the harness treat "the arm" as a list of one or two arms with no other change? | Done 2026-10-11: plan steps carry `arm` only when the manifest has more than one (single-arm planner requests hash-identical to before); reach checked per arm; parallel steps unless they share an object or slot; one decision per arm per event; STOP parks all. 18 tests on the two-arm stub | Here |
| H3 ✔ | Does the OpenAI Decisions API work as a decision backend, and does the 0.8 gate still work on its confidence? | Done 2026-10-11 on the Mac (`docs/live-results.md`): its `confidence` field is Jev's formula, so the gate carries over. Accuracy similar (right-now 90% vs Jev 87%; better on user text, 51/53 vs 45/53); latency p50 129 / p95 254 ms in the loop vs Jev 102 / 130, with a 1.1 s outlier; it escalates more (45% vs 31%). Default decider: Jev for now | Mac |
| H4 ◐ | Can the plan check stop sending good plans back (39% today)? | Measured live: 49% (Jev) / 66% (Decisions API) of good plans sent back, all from the generic route question on a new plan; the per-object plan-check questions never flag a good plan and catch 59/60 planted bad ones. Fix in progress: route a new plan by the plan-check answers alone, as `docs/v2.md` already says | Mac |
| H5 ◐ | Can a transient "wait" stop leaking into the restated task? | Measured live: leaks whenever a wait reaches the planner (1/3 Decisions API runs; 15 cases in the September logs, every one after a replan). Fix in progress: planner told a restatement holds only the end result, plus a code guard that keeps the previous reading if the new one mentions the transient command | Mac |
| H6 ✔ | Can the harness record episodes in a LeRobot-style format? | Done 2026-10-10: `recorder.py` (per-tick JSONL in the server) + `lerobot_export.py` (LeRobot v2.1 layout, npz columns until pyarrow is in the venv, per-episode tasks and events, frame-accurate replay check). The committed K2 recording exports to 20 episodes / 7014 frames / 1.1 MB, byte-identical on re-export | Here |
| M1 ✔ | Is the brain hardware-agnostic? | Done: the same brain runs on the stub's WidowX-like and push-only manifests and on the physics sim; the planner schema and robot description are generated from each, a non-conforming manifest is refused | Here |

### Skill library (sim)

| # | Question | Pass when | Runs |
|---|---|---|---|
| K1 ✔ | A MuJoCo WidowX scene with blocks and a tray, behind one skill interface (`start`, `status`, `hold`, `pause`, `resume`, `stop`, heartbeat) | Done: `scene.py`, `protocol.py`; the policies run off an `ArmIO` surface a real backend can implement | Here |
| K2 ✔ | `pick_and_place` as real motion: approach, grasp, lift, carry, place, release | Done: 80/80 over seeds 0-3 (`trials.py`) | Here |
| K3 ✔ | Interrupting mid-motion: hold at a safe point keeping the grip, pause, resume, re-target to a moved block | Done inside K2: hold / pause / resume / retarget from every phase, tested | Here |
| K4 ✔ | `stack_on`, `push`, `hand_over` (to a hand position), `survey` | Done 2026-10-10: stack_on 80/80, push 80/80 (seeds 0-3), hand_over tested with an injected hand; begin-from-held and hold/resume mid-skill for each | Here |
| K5 ✔ | Two arms in one scene (two WidowX models) with the shared-zone lock | Done 2026-10-11: both arms picking at once 20/20 each with 0 trips; hand-over through the zone 20/20, every trial exercising the lock; one-arm trials unchanged. `robojev-conformance --arms 2` | Here |
| K6 ✔ | The robot server as a real MCP server (stdio or HTTP) with the heartbeat watchdog | Done 2026-10-10 (`mcp_server.py`, `mcp_client.py`): the brain's 3-block sort passes over a stdio subprocess unchanged; event delivery 1.6 ms p50 / 2.5 ms p95, heartbeat 1.6 ms, world 2.1 ms | Here |

### Perception (sim and offline)

| # | Question | Pass when | Runs |
|---|---|---|---|
| P1 | Block positions from the rendered scene camera | Within ~1 cm of MuJoCo ground truth | Mac |
| P2 | Reading block labels (numbers or letters) from a frame: local detector vs OpenAI vision vs the Decisions API with an image | Right on 95% of frames; latency per update recorded | Mac |
| P3 | A hand entering the frame, fast enough to pause | Detected within ~300 ms of entering | Mac |
| V1 ◐ | The Decisions API with a camera frame: latency, and accuracy on hand-near-gripper, grasp-succeeded and block-knocked-over, against text-only | Preview 2026-10-11: a 512×384 frame costs ~230 tokens and ~20 ms (p50 150 ms). Accuracy not measurable yet: the sim has no person and a drawn-in hand isn't one. Needs real frames (the bay camera, or a hand mesh in the twin) | Mac |
| V2 | The planner with an annotated frame (object ids drawn on): does it catch planted disagreements between the written state and the picture? | 9 of 10 planted disagreements caught, no false alarms on 10 clean frames | Mac |

### Closed loop

| # | Question | Pass when | Runs |
|---|---|---|---|
| L1 ✔ (mock decisions) | The e12v2 harness driving K2-K4 skills on the MuJoCo WidowX, with ground-truth positions | Done 2026-10-10 (`tests/test_l1_slice.py`, `experiments/results/l1_slice.txt`): 3-block sort 22.9 s; correction mid-carry → hold in 0.10 s, order reversed, no drops; hand mid-carry → pause in 0.11 s, 12 cm clearance, resume 0.17 s after it leaves; STOP mid-carry → parked in 2.5 s, block held. Live decisions on the Mac remain | Here (mock decisions), Mac (live) |
| L2 | The same with perception (P1-P3) instead of ground truth | Same bar, with noise from real rendering | Mac |
| L1-pair ✔ (mock decisions) | The two-arm brain on the two-arm physics sim (the crawl slice) | Done 2026-10-11 (`tests/test_crawl_slice.py`, `experiments/results/crawl_slice.txt`): 3-block sort across the table through the zone, 51.5 s, both arms in flight 17.6 s, 0 trips; correction → both arms' decisions in 0.17 / 0.24 s, order reversed, no drops; hand → arm_1 pauses in 0.27 s at 9.9 cm while arm_0 works; STOP parks both in 5.9 s with both blocks held. One bug found: the sim reported headings in the arm's frame, not the common one | Here |

### Day-one hardware kit (prepare now, run on the arm)

| # | Question | Prepared now | Runs |
|---|---|---|---|
| D1 ◐ prepared | Does the real arm match sim for the skill interface? | Prepared 2026-10-11: `robots/widowx_real/` runs the sim's policies unchanged over v1's driver code (`RealRobotServer`, `StaticWorld` of typed-in positions, a fake driver for tests); `robojev-conformance --robot real` runs the same trials; `docs/day-one.md` is the checklist. Found in v1's driver wrapper and worked around at the driver boundary: no yaw command, and any opening under 3.6 cm per side closes by force. Runs on the arm remain | Arm |
| D2 | Calibration: scene camera to arm base | A ChArUco or touch-point routine and its check | Arm |
| D3 | If there's a second arm: is it a follower or a leader, and can it grasp? | W1 and W7 from `docs/widowx-pair.md` as scripts | Arm |
| D4 ◐ prepared | First real closed-loop run | `docs/day-one.md`: power, teleop off, workspace box from a touch-off, STOP tested, then the conformance trials and the slice | Arm |

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
- **Two facing arms collide wrist to wrist, not fingertip to fingertip (K5).** The wrist-roll
  housing stands 10.4 cm ahead of the fingertips toward the facing arm, at every straight-down pose,
  so the physics forces fingertip separation of at least 25 cm along the base line. A proximity
  check on end-effector points is meaningless for a pair; the server checks the fingertips, the
  wrist housing and the gripper bar as the model has them, and holds the later-moving arm at a 4 cm
  gap. Consequence: base spacing must be at least 0.72 m (0.76 m default, an 8 cm strip on the
  centre line); own-half work beyond an arm's side line cannot overlap in time with the other arm's
  zone work. Also found: a carry that keeps the pick's wrist yaw can roll the 19 cm gripper bar into
  the folded upper arm at a close destination; the carry now re-aims the closing axis for the
  destination (one-arm trials unchanged).
- **The evaluation was mis-weighted, and the task was wrong (2026-10-12).** The hand-to-pause number
  led every report because it is easy to time, yet a hand near the gripper is the one event where a
  model adds least (the code stop distance and perception own it); the fast loop's real job is
  cheap judgment on language and scene events, and that was the weak part live. And the numbered
  sort into a tray was inherited from old experiments; the demo spells words with letter blocks.
  Replaced by `src/robojev/battery` (`docs/battery.md`): generated spelling episodes, judged on
  outcomes only (no expected actions anywhere), with always-LLM and rules-only controls. Offline on
  the stub the fast loop reacts in 0.1-0.3 s vs always-LLM 0.4-0.8 s with half the planner calls.
  The battery also found that `object_gone` could never fire in the change detector (fixed).
- **Live models on the slice (Mac, 2026-10-11; `docs/live-results.md`).** Sort, hand and STOP pass
  with both deciders; the planner replans in 3.9 s p50 / 6.4 s p95, so a correction holds the arm
  ~4 s, not the mock's 0.3 s. Three design problems, all brain-side: (1) Jev confidently said
  "carry on" (0.83) to "actually, highest on the left" because the decision saw the planner's
  restatement ("ascending numeric order"), not the user's words; with the user's wording it holds 5/5;
  (2) half of good plans go back to the planner, all from the generic route question on a new plan;
  (3) a "wait" leaks into the restatement whenever it reaches the planner. Fixes in progress (H4, H5
  rows and P1 in the next entry). The top risks for the real arm, from that run: a confident wrong
  answer no gate catches; latency tails and missing hand detection (the code's stop distance is the
  day-one safety); replan churn.
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
- ~~`test_robots_widowx_real.py` was load-sensitive~~: fixed 2026-10-12. The test counted recorded
  frames, which only held when the machine kept up with real time (a starved box dropped the 50 Hz
  policy to 11 Hz). The real-arm tests now run on a lockstep virtual clock that only the test
  advances (`widowx_real/clock.py`); assertions unchanged, the frame trace byte-identical every run,
  10/10 under the load that reproduced the failure, and the file runs in 11 s instead of 50.

## One arm (2026-10-14)

The demo will have one arm, and a Franka is available for a test session. Decisions: the brain runs
in its crawl mode (one brain, one arm), which is what every live run has used; the two-arm code
stays (`tests/test_brain_two_arms.py`, the pair scene) but the walk and run stages
(`docs/widowx-pair.md`) are shelved; the Franka gets a robot server behind the same protocol
(`src/robojev/robots/franka/`, built against a fake driver first) and its own checklist
(`docs/franka-day-one.md`). The WidowX day-one kit stays as written for whenever that arm is at hand.

## Follow-ups from the live re-run (2026-10-13)

The re-run's rule gaps are fixed in code (`docs/live-results.md`, "Rules added after the re-run"):
a pause that must end (hand gone: code resumes; otherwise the user is asked), a failed replan that
keeps the hold, the plan check that knows a spare letter, `user_words` without remarks, the
folded-in correction, the routine step_done. Owed: Handoff 8 (the battery and the slice live again
under them). Still open from the re-run:

- Both full-five slice passes on an awake Mac (on power, lid open), either decider.
- The Decisions API's latency tail (18 decisions over 1 s in one episode) is the API's; nothing in
  the brain can fix it, so the default stays Jev until a run shows otherwise.
- Planner latency p50 4-7 s per call this run (2.8 s in the first); OpenAI-side, not ours.
- (Handoff 9) A misplaced letter passes the plan check: the check asks about objects a plan leaves
  alone, not about a wrong step. Candidate: one Noul per step destination on plan_arrived, tried
  with the battery's send-back count watched (`docs/live-results.md`, "After Handoff 9").
- (Handoff 9) The Decisions API sends routine finished steps to the planner on its own `fast_llm`
  (15 per 10 battery episodes; Jev 0). If it stays, the fix is the route criteria's text, judged on
  the recorded requests first, not a gate.
- (Handoff 9) The Decisions API's latency tail (0.7-2.5 s decisions in one window) decided 4 of 6
  wait runs; the API's side. Jev's max in the slice was 204 ms.
