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
