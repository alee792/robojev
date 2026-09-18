# robojev

Drive a Trossen WidowX AI arm from natural language with TypeSafe's Jev model, Doom-style: code
renders a text situation report, asks a battery of small typed judgments ~10 times a second, and
composes the answers into persistent arm commands. Live standing orders change behaviour with no
code change. Research bundle: `context/`. API probes: `experiments/`.

**Read [docs/assessment.md](docs/assessment.md) first.** This was a one-day experiment. It works on
real hardware and it is also heavily fitted to one paper cup on one table; that file says which is
which, and what was never tested.

## What works (2026-09-17)
- **Real arm, Jev in the loop**: Jev sequences a full pick-move-set-down of a paper cup, repeatedly,
  from "keep moving the paper cup to a different spot". Run real13 did four grasp-to-release cycles
  in 234 s unattended; demo2 did one with the force-controlled grip (fingers stopping at 30 mm per
  side instead of crushing the cup). Earlier: scripted first contact passed (6 cm square at 3 cm/s,
  0.2 mm corner error, 3 mm lag). Safety layer: `docs/safety.md`.
- **Not tested on the real arm**: evade against an actual hand, the scene camera in the loop (never
  calibrated), the mat routine, the VLM naming tier, the segmenter. See `docs/assessment.md`.
- **Sim** (MuJoCo, same code, rendered wrist + overhead cameras): task "hover over the paper cup";
  a standing order "keep the gripper at least 15 cm away from the black flat object" makes the arm
  back away from a drifting object (first avoid command 0.2 s after injection; min distance 5 cm
  without the order vs 13 cm with it); "move very slowly" drops the speed cap; task text switches
  the target; a 6 s Jev outage walks the hold -> rise ladder and recovers.
- **Rates**: 10 Hz tick. Requests are event-driven: one is sent only when an entity moved > 2 cm,
  an entity appeared/disappeared, the arm's phase facts, orders, task or offered primitives changed,
  or 1 s of silence elapsed (`--clocked` restores one request per tick). A 30 s sim run sent 48
  requests in 300 ticks. ~2.8k tokens/call, p50 ~145 ms, p95 ~256 ms.

## Jev as the policy (v1, evening of 2026-09-17)
Direction: Jev picks every channel every tick in a scene that changes at the decision rate; code
resolves picks into setpoints and enforces limits. Built:
- **Primitive library** (`src/robojev/skills.py`): move_above, descend_to_grasp, close_gripper,
  lift, move_to_place, lower_to_place, set_down_here, open_gripper, retreat + hold/back_off/rise_away.
  Offered to Jev only when preconditions hold; verified by code; new verbs are sequences Jev
  composes at run time (no code per task).
- **Battery v1** (`questions/v1.py`): target, place, next (3 paraphrases, 2-of-3 majority), speed,
  evade (none/up/back/left/right/away_from X = Doom's DODGE, overrides everything), orders_violated,
  task_done. Default operator orders (hands, moving objects, never drop) sit above user orders.
- **Event-driven requests**: a request only when facts changed (entity moved > 2 cm, phase changed,
  orders/task changed, offer set changed) or 1 s of silence; ~85% fewer calls; the silence ladder
  distinguishes "chose not to ask" from "asked and got nothing".
- **Dynamic sim scenario** `--scenario intruder`: a hand slides in between gripper and cup mid-task
  and the cup is moved while the arm works. `scripts/analyze.py` measures pick changes per tick,
  intrusion-to-evade latency and task resumption.
- **Perception hygiene learned the hard way**: height-aware clustering only for top-down cameras,
  carried object masked and its track pinned to the gripper, partial/merged/occluded views never
  move a track, static objects remembered 3 min, moving ones 20 s, no new tracks from blobs at the
  gripper or wider than 16 cm.
- **VLM naming tier** (`--vlm claude|stub`, `perception/vlm.py`): off the control path, names each
  confirmed track once from a colour crop and drops phantoms; needs `ANTHROPIC_API_KEY`.

Results in sim (intruder scenario, pick-and-place "put the paper cup to the left of the phone"):
full sequence grasp -> carry -> lower -> release with a hand intrusion on the way (run intr11);
evade fires 0.6-7 s after the hand appears depending on where the arm is; the `next` pick changes
on 15-35% of answers versus 0-4% in the static scene. Remaining fragility is perception phantoms
from the shape-only labeller (the VLM tier is the fix) and override/timeout interplay.

## Layout
| Path | Role |
|---|---|
| `src/robojev/config.py` | every limit, band, threshold and timing constant |
| `src/robojev/questions/v0.py` | the battery (target, motion, hover_position, hover_height, speed, avoid, orders_violated) |
| `src/robojev/state.py` | World -> situation report (orders first, glossary, user text, facts with bands) |
| `src/robojev/brain.py` | answers -> commands: gates, hysteresis, avoid (dodge) override, silence ladder |
| `src/robojev/loop.py` | 10 Hz tick, in-flight/stale/out-of-order handling, perception thread, logging |
| `src/robojev/events.py` | change detector: what makes a tick worth a Jev request |
| `src/robojev/arm/` | one interface; `fake`, `sim` (MuJoCo), `real` (trossen driver, own thread, park-on-exit) |
| `src/robojev/perception/` | D405 camserver/client, camera->base geometry, depth tabletop detector, tracker with object permanence, fixed-camera calibration |
| `src/robojev/dashboard.py` | live judgments, confidence, latency, cameras, orders/task/STOP inputs |
| `src/robojev/replay.py` | re-ask recorded states offline, diff picks |
| `scripts/first_contact.py` | the scripted real-arm test |
| `runs/<name>/` | ticks / answers / commands / events JSONL + config.json per run |

## Real arm (night of 2026-09-17)
Jev-sequenced pick-move-set-down works on the real WidowX, repeatedly, 21-30 s per cycle
("keep moving the paper cup to a different spot", runs real8..demo2). What it took:
- **Side grasp** (`approach_side` / `advance_to_grasp`): the pads open 8.0 cm and a paper cup is
  7-9 cm, so the fingers come in level-ish (wrist pitch 0.5 rad on the approach, 0.35 / 0.25 on the
  advance) with the tips 2 cm above the table and 2.5 cm past the cup's centre. Geometry from the
  URDF: tool point = fingertips, pad faces 1.4-6.9 cm behind them, palm 6.9 cm back.
- **Wrist pitch channel** through the mover, sim IK and the real stream; a survey pose (pitch 0.5)
  at staging so the camera sees the table before the first pick; a `survey` primitive Jev can pick.
- **Force grip**: external-effort mode (6 N) with the Cartesian stream paused 1.1 s (streaming
  re-sends the stored gripper position and overrides the force), then the stop width is held in
  position mode. Position-mode closing crushed the cup to 2 cm; force stops at ~3 cm per side.
- **Perception fixes from real frames**: the pads sit inside the D405's minimum range and return
  garbage depth (lower 40% of the wrist image dropped, points under 11 cm dropped); a tool-axis
  cylinder masks fingers, carriages and wrist at any pitch; oblique views place objects by their
  near edge + across width; tracks are never moved by views taken within 15 cm of the gripper;
  a track squarely in view with nothing there is dropped (negative evidence); slivers < 2 cm and
  thin posts (table rail) are not targets; the "just appeared" grace counts from the arm going live.
- **Safety additions**: on a trip the arm is commanded to its actual pose (stop fighting); thermal
  governor (slow above 78 C, rest above 86 C: the shoulder rotor hit 96 C and the controller idled);
  `--no-evade` for pickup-only runs; safety picks need a streak to interrupt; gripper actions are
  never interrupted.
- **Places**: shift_left/right/away/closer from the pickup, somewhere_else (code picks a free spot
  in the survey view), where_it_was_set_down (memory only, within a run), and for a flat mat
  on:<mat> / off:<mat>. The mat is found by colour on the table plane (depth cannot see 3 mm),
  every object carries a `resting_on` fact, and the target rule says: pick the object that breaks
  the standing rule. `--scenario mat` exercises this in sim.
- **Cameras**: `scripts/camserver.sh <port>` (dyld shim so librealsense 2.56 does not segfault on the
  D455's IMU on macOS 26), `--camera name=url` for any number of display-only cameras,
  `scripts/webcam_server.py` for a USB webcam, `--segment` for FastSAM instance masks on the wrist
  camera (~50 ms on MPS; splits touching objects). Scene-camera calibration: `robojev calibrate
  --survey` (the read-only arm reports a stale pose, so calibrate from the survey pose); it refuses
  one-object or high-residual fits.

## Run
```bash
uv sync --extra sim --extra real --extra cameras
# sim, with the drifting-object scenario and the dashboard on :8080
.venv/bin/robojev run --arm sim --perception simcam --scenario drift --task "hover over the paper cup"
# bay: camera server needs root on macOS (librealsense cannot claim the UVC interface otherwise)
sudo tmux new -d -s cams "scripts/camserver.sh 8765" \; new-window "scripts/camserver.sh 8766"   # wrist D405 :8765, boom D455 :8766
.venv/bin/robojev calibrate --survey --out overhead_calib.json           # arm moves to the survey pose; 2 objects 8 cm+ tall, 10 cm apart, seen by both cameras
.venv/bin/robojev run --arm real --perception camera --i-am-at-the-estop --task "keep moving the paper cup to a different spot"   # the demo that works
.venv/bin/robojev run --arm sim --perception simcam --scenario mat --task "keep the cup on the mat and the box off the mat"
.venv/bin/robojev run --arm real-ro --perception camera --task "hover over the paper cup"   # no motion
.venv/bin/robojev run --arm real --perception camera --overhead http://127.0.0.1:8766 --overhead-calib overhead_calib.json --i-am-at-the-estop --task "hover over the paper cup"
.venv/bin/robojev replay runs/<name> --questions v0
```
Orders and task text: dashboard inputs, or `scripts/inject.sh <port> <delay_s> orders|task "<text>"`.
