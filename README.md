# robojev

Drive a Trossen WidowX AI arm from natural language with TypeSafe's Jev model, Doom-style: code
renders a text situation report, asks a battery of small typed judgments ~10 times a second, and
composes the answers into persistent arm commands. Live standing orders change behaviour with no
code change. Research bundle: `context/`. API probes: `experiments/`.

## What works (2026-09-17)
- **Real arm**: scripted first contact passed (stage, hover, 6 cm square at 3 cm/s, park; 0.2 mm
  corner error, 3 mm lag). Wrist D405 perception finds a paper cup on the bench; Jev picks it from
  "hover over the paper cup" with p = 1.0. Safety layer: `docs/safety.md`.
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

## Run
```bash
uv sync --extra sim --extra real --extra cameras
# sim, with the drifting-object scenario and the dashboard on :8080
.venv/bin/robojev run --arm sim --perception simcam --scenario drift --task "hover over the paper cup"
# bay: camera server needs root on macOS (librealsense cannot claim the UVC interface otherwise)
sudo .venv/bin/python -m robojev.perception.camserver                      # wrist D405, :8765
sudo .venv/bin/python -m robojev.perception.camserver --serial 408222301818 --port 8766   # boom D455
.venv/bin/robojev calibrate --out overhead_calib.json                      # arm read-only, 2 objects in both views
.venv/bin/robojev run --arm real-ro --perception camera --task "hover over the paper cup"   # no motion
.venv/bin/robojev run --arm real --perception camera --overhead http://127.0.0.1:8766 --overhead-calib overhead_calib.json --i-am-at-the-estop --task "hover over the paper cup"
.venv/bin/robojev replay runs/<name> --questions v0
```
Orders and task text: dashboard inputs, or `scripts/inject.sh <port> <delay_s> orders|task "<text>"`.
