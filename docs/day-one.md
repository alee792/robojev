# Day one: the real WidowX AI under the skill server

The checklist for the first day with the follower arm, in order. The code is `src/robojev/robots/widowx_real/`
(spike D1, `docs/harness-spikes.md`), built against a fake driver; every step below that touches the
hardware is marked **verify** because nothing in it has run on an arm yet. Keep a hand on the e-stop
for every motion until "STOP tested" has passed.

Example `blocks.json` (positions in the base frame, metres; yaw in degrees; `table_z` from the touch-off):

```json
{"table_z": -0.02,
 "blocks": [{"name": "block_1", "xy": [0.30, -0.08], "yaw_deg": 0, "size": 0.04},
            {"name": "block_2", "xy": [0.32, 0.10], "yaw_deg": 0, "size": 0.04}],
 "tray": {"origin": [0.36, 0.12], "step": [0.0, -0.08], "n": 5, "slot": 0.05}}
```

## 1. Power and network

- [ ] Follower on, controller booted. `ping 192.168.1.5` answers (`docs/widowx-pair.md`: follower .5, leader .3).
- [ ] The Mac has `trossen-arm==1.9.3` (`uv sync --extra real`). `python -c "import trossen_arm"` works.
- [ ] Nothing else is connected to the follower. The driver connection is exclusive; a stale session
  from another tool leaves an error on the controller that `configure(..., clear_error=True)` clears.
- [ ] Dry run of the whole day with no arm: `uv run --frozen robojev-conformance --robot real --driver fake
  --blocks blocks.json --skill pick_and_place --n 2` passes. It exercises everything below except the hardware.

## 2. Teleop OFF

In the kit, the leader's motions drive the follower. If that process is alive while robojev commands
the follower, two controllers fight over one arm.

- [ ] On the machine that runs the teleop (the Mac, or wherever the kit's leader-follower script
  lives): `ps aux | grep -i -E "teleop|leader|trossen"` shows nothing of the kit's. Kill anything it shows.
- [ ] Move the leader by hand: the follower does not follow.
- [ ] Only then pass `--teleop-off-ack "<your name>"`. `RealRobotServer.preflight()` records who
  acknowledged it and when (a `preflight` note in the recording); `open()` refuses without it. The
  check is a human one: no code here can see the teleop process.

## 3. Workspace box set for the table

- [ ] Touch-off: drive the fingertips to the table by hand (or with the kit's tool) and read the EE z
  from `get_cartesian_positions()`. That is `table_z` (base frame; about -0.02 in the sim and the bay notes).
  Put it in `blocks.json`.
- [ ] `day_one_config(table_z)` (`widowx_real/arm.py`) gives the sim's box moved onto that table:
  x 0.15-0.42, y +-0.25, z from 4 mm above the table to 16 cm above it, speed cap 0.15 m/s. Check
  the box against the real table: nothing (tray lip, clamp, camera stand) inside it; the far edge
  reachable straight down (sim: x 0.42 at travel height; **verify** on the real arm, which sags).
- [ ] The e-stop is within reach of the operator, not of the arm.

## 4. STOP tested

With the gripper empty and the table clear:

- [ ] Start the server and let it stage (`RealArm.start`: joint move to STAGED, then a blocking
  Cartesian move to the hover start at travel height, wrist pitched 0.5 rad; then 1 s of effort
  baseline, status "live"). **Verify** the hover start is reached (an `arm_event` note warns if it missed).
- [ ] From a Python prompt: `srv.start("arm_0", "survey", {})` then `srv.stop()`. The arm parks:
  STAGED then SLEEP in two 3 s joint moves, gripper left as it was. Status reads `stopped`, further
  `start()` raises "the arm is stopped (STOP)".
- [ ] Repeat with a hardware e-stop press mid-`survey`: the driver raises in RealArm's thread, the
  snapshot reads `error`, the server fails every skill with "stopped: the arm's I/O thread died".
- [ ] `hold`/`resume` during `survey`: the arm stops within a tick and continues on resume.

## 5. Conformance trials

Each trial prompts for the layout, runs the skill, judges from bookkeeping, and records (`--out`).
Re-lay the block where the prompt says before pressing Enter; the server's world is reset to the
typed positions at each trial.

- [ ] `uv run --frozen robojev-conformance --robot real --robot-ip 192.168.1.5 --blocks blocks.json
  --teleop-off-ack "<name>" --skill pick_and_place --n 20 --out experiments/results/d1_pick.jsonl`
  First trial with a hand on the e-stop and the block yaw 0 (closing axis along +y: the one orientation v1
  streamed). Watch, in this order: the pitch ramps to 90 deg (1.8 s at 0.6 rad/s) and the fingers point
  straight down (**verify**: v1 verified 75 deg); the descent stops with the fingertips 5 mm above the
  table (**verify**: sag; see 7); the fingers close on the block and `holding` appears ~1.1 s later; the
  lift, carry, lower, release, retreat.
  Second trial with the block yawed 30 deg: the wrist rolls so the fingers meet the faces squarely
  (**verify** the sign: a wrist turning the wrong way means `orientation.down_orientation` needs its yaw
  negated; the sim and the driver may disagree on the handedness of the rotation vector).
- [ ] Then `--skill stack_on` (two blocks in `blocks.json`), then `--skill push`.
- [ ] Bar, as in sim: 19 of 20. A failed trial's line says what the skill thought (`stalled`, `grasp_failed`,
  `dropped`, `timeout`) and what the bookkeeping saw; a success the person disagrees with (block visibly
  not in the slot) is a failure: write it down, the judge cannot see it.

## 6. The brain's slice

- [ ] `tests/test_slice.py`'s scenario (sort three blocks, one correction) with the stub user, pointed at
  `RealRobotServer` instead of `SimRobotServer`, heartbeat on. Pass as in L1: no drops, the correction
  holds within 0.5 s, STOP parks with the block held.

## What to record

- Every run with `--out`: frames at 50 Hz (EE, yaw, width, holding, setpoint, lag, external force, the
  driver's rotation vector, arm status), actions (goal, speed, yaw, the angle-axis streamed, gripper,
  frozen), events, and notes: `preflight`, `skill_start`, every RealArm log line (`arm_event`), the verdict.
- By hand, per trial: did the block end where the judge says; the lowest fingertip height you saw on the
  descent; anything the arm did that the status text did not say.
- The first yawed grasp: which way the wrist turned.
- The closing: how long from the close command to the fingers stopping, and to `holding` (v1: ~0.5 s after
  the 1.1 s window; the policy waits 1.3 s of still jaws, `REAL_PARAMS.settle_s`).
- Rotor temperatures over 20 trials (`temps` in the snapshot): RealArm slows at 78 C and rests at 86 C.

## What "done" looks like

- 19/20 `pick_and_place`, 19/20 `stack_on`, 19/20 `push`, each judged by the bookkeeping and agreed by eye.
- `hold` / `resume` / STOP behaved in every phase they were tried in.
- One closed-loop sorting episode with the brain.
- The recordings in `experiments/results/`, and this file's **verify** items each ticked or turned into a change.

## Known unknowns (in the order they will bite)

1. **Orientation streaming.** v1 streamed `[0, pitch, 0]` only, at 75 deg. The yaw goes through
   `OrientingDriver` (`widowx_real/arm.py`), which rewrites RealArm's pitch-only triple into the full
   rotation vector `Rz(yaw - pi/2) . Ry(pitch)` on its way to the driver, without editing `arm/real.py`.
   Unverified: that the controller accepts a 90 deg pitch; that the driver's triple is a rotation vector
   and not roll-pitch-yaw (identical for a pure pitch, which is all v1 saw); the sign of the yaw.
2. **Gravity sag.** Every sim body is gravity-compensated; the real arm is not. Tolerances loosened:
   `arrive_tol` 1.5 cm (EE vs goal with the setpoint there), `xy_tol` 12 mm, `z_tol` 10 mm (`REAL_PARAMS`).
   Two tolerances are still hardcoded in the policies and will show as `stalled` in `lower` or
   `approach` if sag exceeds them: `Grasping.run` uses 10 mm at travel height and 6 mm at the lower; they
   must become `Params` if the real arm misses by more. The descent's 5 mm fingertip height is set from
   the table z: a sagging arm puts the tips in the table. Raise `tip_height` before loosening anything else.
3. **The gripper.** The driver's position is one carriage's travel (0-0.04 m; width = twice that,
   8 cm open) **verify**. RealArm closes BY FORCE below 0.036 per side (its paper-cup rule), at
   `grip_force_n` 6 N: enough for a 4 cm wooden cube? **verify**, and whether the fingertips' pinch on
   the lower 1.5 cm holds through a 0.15 m/s carry. Openings are widened to at least 7.2 cm so RealArm
   does not read them as closes; the neighbour-aware opening is lost until `real.py` gets an explicit mode.
4. **`holding` detection.** v1's flag means "the fingers stopped more than 6 mm per side from closed",
   set when the 1.1 s force-close window ends, with no object identity: the adapter names the block the
   bookkeeping had between the fingers at the close. A block that slips out later is seen only if the
   carriage closes 4 mm further (`DROP_Q`); v1 never checked. **verify** both.
5. **Pushing.** The sim's push relies on wood-on-wood friction 0.5 and pads 1.9 cm up the cube; the real
   pads, table and cube differ. `blocked` is read from the EE lagging where the slide should have it.
6. **Thermal.** 20 trials back to back at 0.15 m/s: the shoulder reached 96 C in a v1 run. RealArm's
   governor freezes the arm (a `safety_trip` here, "motor at N C: resting") and resumes on its own;
   the skill stays held until `resume()`.
7. **The world is bookkeeping.** No camera on day one: `where` is where the arm was told to put things.
   P1 (`PerceptionWorld`) replaces it.

## Changes `arm/real.py` will need (not made: D1 wraps it)

- An orientation command: `command(goal, speed_cap, gripper, pitch, orient=None)` streaming the full
  rotation vector, so `OrientingDriver` can go.
- An explicit gripper mode in `command` (open to a width / close by force) instead of the 36 mm rule.
- `import trossen_arm` inside `start()` and `_run()` even when a `driver_factory` is given: the enums
  should come through the factory (the fake installs an enum-only stand-in module to get past it).
- `stop()` blocks ~6 s (join plus two parking moves); the server parks in a thread. A non-blocking park
  with a done flag would let `stop()` report when the arm is safe.
- `holding` should be re-read from the live carriage position, not latched at the end of the close.
