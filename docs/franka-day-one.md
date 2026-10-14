# Day one on the Franka

The checklist for the first session with the Franka arm (Panda or FR3), in order. It mirrors
`docs/day-one.md` (the WidowX version) with the Franka's own constraints; the robot server is
`src/robojev/robots/franka/`, built against a fake driver, so every step that touches the arm is
marked **verify**. Keep the external activation device (the stop button) in hand for every motion.

## 0. Before the session: three answers that shape the day

1. **What drives the arm.** libfranka needs a Linux box with a PREEMPT_RT kernel on the robot's
   control network, holding a 1 kHz loop. The Mac cannot do that. Options, in order of likelihood:
   a lab PC that already has libfranka and franky or panda-py; the Pi 5 (stock kernel, not RT:
   `docs/pi-setup-log.md` measured 97 us worst wake-up headless, one 782 us spike with the desktop
   up; libfranka wants under 1 ms with no spikes, so it is a maybe, headless only); nothing yet
   (then the day is the control box, not the arm).
2. **Which robot and system version.** Desk shows it (Settings, Dashboard). It fixes the libfranka
   version (`docs/panda-setup.md`, S1) and whether FCI must be activated in Desk (system 4.2.0+).
3. **Is there a camera** above the table, and can it record a few hundred frames during the session
   (section 8). The multimodal spike needs real frames of cubes and a hand; this is the cheapest
   time to get them.

## 1. Robot and network

- [ ] Robot on, Desk open at its IP, joints unlocked, FCI active (**verify** the version rule).
- [ ] Control box on the controller's **control** port (not the arm base), static IP on the robot's
  subnet; `ping` answers; `communication_test <robot-ip>` for 10 minutes with almost no lost packets
  (S2). A second network reaches the internet for the planner and the decision model.
- [ ] Nothing else holds the FCI connection (Desk's own apps included; one client at a time).
- [ ] Dry run of everything below with no arm, on any machine:
  `uv run --frozen robojev-conformance --robot franka --driver fake --blocks blocks.json --skill pick_and_place --n 2`.

## 2. The driver

The server talks to the arm through `robots/franka/driver.py`'s small interface: read the end
effector, stream a Cartesian setpoint with a straight-down tool and a yaw, move the gripper, stop,
recover, park. The fake implements it; the real one is one file (`franky_driver.py`, named by
the conformance tool's error until it exists) over whichever library the control box has:

- **franky**: `Robot.move(CartesianMotion(...), asynchronous=True)` for the setpoint,
  `Gripper.move_async` / `grasp_async`, `Robot.recover_from_errors()`, `Robot.stop()`. Build must
  match the libfranka version (their releases page has builds for older versions). **verify**
- **panda-py**: the same shape with its own names.

- [ ] Before the server: move the arm between two poses 50 times with the library alone, interrupt
  one motion and hold, open and close the gripper (S4). Note the time from the stop call to the arm
  standing still.
- [ ] Write `franky_driver.py` against the interface (`FrankyDriver(ip)`); then
  `--driver franky --preflight-ack "<name>"`. What its author needs to know, and where it comes from:
  - the robot and system version (Desk), hence the libfranka version (Panda 0.9.x; 0.10+ is FR3
    only) and the franky or panda-py build to install (their releases pages carry builds for older
    libfranka; the default PyPI wheel targets the newest);
  - the flange-to-fingertip offset for the installed hand and fingers, so `read()` reports the
    fingertip point (about 0.1034 m to the hand plus the fingers), and the closing-axis direction in
    the end-effector frame, which sets the sign of the yaw;
  - the grasp epsilons: the policies close 8 mm narrower than the block, so `epsilon_outer` must be
    at least 0.01 m or libfranka reports a held block as a failed grasp; force 20-30 N to start;
  - the collision thresholds (Desk or `set_collision_behavior`), what the error strings look like,
    and whether `recover_from_errors` works from software alone;
  - a safe park joint pose reachable from everywhere in the box without sweeping the tray;
  - whether franky's asynchronous move with a changed target preempts cleanly when the same target
    is re-sent at 50 Hz (a repeat must be a no-op, or it restarts the trajectory every tick).
- [ ] The fake cannot show: joint limits and self-collision at the box's edges for a straight-down
  tool, the real arrival behaviour (sag, overshoot), the gripper's latency, what a reflex looks like,
  how long the park takes and what it sweeps through. Each is a **verify** below.

## 3. Workspace box and the table

- [ ] Touch-off: guide the fingertips to the table (guiding mode, the buttons on the wrist) and
  read the end-effector z. That is `table_z` in `blocks.json`.
- [ ] `FrankaLimits` (`robots/franka/arm.py`) is the box: reach, height, speed cap 0.25 m/s. Walk
  it against the real table: tray lip, clamps, the camera stand all outside it. The far edge must be
  reachable with the tool straight down (**verify**: the Franka's reach straight down is shorter
  than its sideways reach).
- [ ] Set the collision thresholds in Desk or the driver to the demo's values (soft: a brush of a
  hand must trip, a 4 cm cube must not); **verify** what a reflex looks like to the server (a
  `safety_trip` event, the arm reads tripped until `resume()`, which calls `recover()`).

## 4. STOP and the reflexes

With the gripper empty and the table clear:

- [ ] `srv.start("arm_0", "survey", {})` then `srv.stop()`: motion stops within a control step,
  the park runs in a thread, status reads `stopped`, `start()` refuses. **verify** the park pose
  clears the tray.
- [ ] The external activation device mid-`survey`: the arm locks, the driver reports an error, every
  skill fails with "stopped: ...". Recover: release the device, `recover()` in software (S11).
- [ ] A gentle bump mid-motion: the collision reflex, as a `safety_trip`, then `resume()`.
- [ ] `hold` / `pause` / `resume` during `survey` and during a carry: stops within a tick, grip kept.

## 5. Conformance trials

- [ ] `--skill pick_and_place --n 20` with the cubes at typed positions, first trial yaw 0, second
  yaw 30 deg (**verify** the wrist turns the right way; the sign of the yaw is the first thing a
  new driver gets wrong). Watch the descent: the Franka Hand's fingers are longer than the
  WidowX's, so `tip_height` and the grasp height may need the pads lower on the cube.
- [ ] Then `stack_on`, then `push`. Bar: 19 of 20 (S7), judged by the bookkeeping and agreed by eye.
- [ ] The Franka Hand's grasp: width 0.08 m max, force 20-70 N; a 4-5 cm wooden cube at ~30 N
  holds through a 0.25 m/s carry (**verify**). `holding` comes from the gripper's reported width
  after a grasp (the fake does the same).

## 6. The brain's slice

- [ ] `tests/test_slice.py`'s five scenarios pointed at `FrankaRobotServer` with live models, from
  the Mac over the network to the control box if the server runs there (the protocol is in-process
  today; `robots/mcp/server.py` carries it over stdio; a TCP transport is a small addition if the
  box and the Mac are different machines). Pass as before: correction holds under 0.5 s, hand to
  pause under 0.5 s, STOP parks with the block held.

## 7. The demo

- [ ] `docs/showcase-demo.md` beats 1-4 with typed lines, cubes at known positions, no camera: task,
  correction mid-carry, a taken cube, a hand (the hand is a typed scene event until perception
  exists). Beat 5 (the handover) needs a hand in view; typed as well on day one.

## 8. Frames for the multimodal spike

If a camera is above the table, record while the trials run, and stage the four situations the spike
scores (`docs/harness-spikes.md`, V1): a cube showing each letter face up, cubes upright and tipped,
the gripper holding a cube and holding nothing after a failed grasp, and a hand reaching toward the
gripper against a hand passing over the table. Two hundred frames across those, with a line per
frame saying what is true (which letter, upright or tipped, in the gripper or not, reaching or
passing) is enough to run the spike offline afterwards. A phone video with a timestamp is fine if
the camera is not wired up yet.

## What to record

Every run with `--out`; by hand: the stop-to-still time, the first yawed grasp's wrist direction,
the lowest fingertip height on the descent, every reflex and how it was recovered, rotor
temperatures are not a concern on the Franka but the joint torque limits trips are: note each.

## What "done" looks like

19/20 on the three skills; STOP, the activation device and a reflex each handled once; one slice
scenario with the brain; two hundred labelled frames; the three answers in section 0 written down.
