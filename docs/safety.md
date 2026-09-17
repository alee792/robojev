# Safety layer for the real arm (v0)

Everything here is enforced in code (`src/robojev/config.py`, `src/robojev/arm/`), not by Jev.
Jev picks among options; code decides whether and how the arm moves.

## What the arm may do
- **Workspace box** (base frame, m): x 0.15–0.42, y ±0.22, z 0.06–0.32. Every goal and every
  streamed setpoint is clamped into it. Nothing outside is ever sent.
- **Orientation is fixed**: gripper pointing down (`[0, 1.57, 0]`). Only xyz and gripper width move.
- **Speed cap**: setpoint advances at most `speed_levels[level]` m/s: 1 / 3 / 6 / 10 cm/s, hard cap
  10 cm/s. Each streamed command asks the controller for a point at most `cap × 0.1 s` away, so a
  controller overshoot is bounded by the same number.
- **Z floor while hovering**: never below the tallest tracked object + 5 cm.
- **Gripper stays open** in this task. No grasping.

## What stops it
- **Tracking-lag trip (primary)**: the EE lagging the streamed setpoint by > 2 cm for 3 ticks
  (150 ms) freezes the setpoint. Normal lag is 3 mm at 3 cm/s, so a blocked arm is unmistakable.
- **Effort watchdog (backstop)**: Cartesian external force is baselined for 1 s at the hover
  start, then tracked by a 3 s EMA while calm; a deviation > 40 N for 5 ticks freezes. The
  driver's estimate shifts ~25 N with motion direction, so this only catches hard pushes.
  Neither detector will notice a paper cup; the z floor and the box are what keep the arm off objects.
- **Jev silence ladder**: 0.5 s without a *fresh* tick → hold; 2 s → rise to 22 cm above the table
  at 1 cm/s. A tick is fresh if an answer was applied, or if the loop deliberately sent no request
  because nothing material had changed since the last one it did send (see `events.py`) *and* no
  request is outstanding or failed since the last applied answer. Any request error, timeout, 529,
  stale/out-of-order drop, in-flight skip or pause counts as silence, exactly as before — an
  intentional skip only counts as fresh while the channel is demonstrably healthy.
- **Confidence gates and hysteresis** (`Thresholds`): conservative picks (hold, back off, rise,
  slower, offset hover, high hover) latch on one answer; aggressive picks (approach, faster,
  directly above, low hover, switching target) need 3 consecutive answers.
- **Majority on `next`**: the primitive that actually moves the arm is asked three ways in the same
  request (`next`, `next_b`, `next_c` — same options, different wording; fan-out costs no latency).
  A pick counts only with 2 of 3 agreeing; its probability is the mean p_max of the agreeing
  variants. No majority → `disagree`: nothing changes and no hysteresis streak is counted. The
  `next_p_max` gate, `next_consecutive` and `next_interrupt_p` still apply on top.
- **`orders_violated` override**: P(yes) ≥ 0.7 holds the arm for that tick.
- **STOP** (dashboard or Ctrl-C): freeze the setpoint; Ctrl-C then parks.
- **Park on exit**: STAGED then SLEEP, 3 s each, on normal exit, on any exception in the arm
  thread, and on Ctrl-C. Same sequence as the bay's `ArmSession`.
- **The e-stop**: the operator is present with a hand on it for every real run in this phase.

## Start-up sequence
sleep → STAGED (joint move, 3 s) → gripper open → one blocking joint-space move to the hover start
(x 0.25, y 0, 22 cm above the table, pointing down, 4 s) → stream setpoints at 20 Hz.

## Rules of engagement
- Only one process holds the follower; the `mission_*` and `mcp__widowx__*` tools are not used
  while the loop runs.
- Bench cleared of anything the arm could hit inside the box except the task objects.
- First real run is the scripted, Jev-free `scripts/first_contact.py`: stage, hover start, a 6 cm
  square at 3 cm/s, park. It records pose vs setpoint lag and the force baseline.
- Real runs need `--i-am-at-the-estop`, and I ask before each one.

## Known gaps (v0)
- No collision checking against objects other than the z floor; no obstacle-aware paths.
- The controller's IK is the only reachability check; an unreachable setpoint's behaviour is
  measured by the first-contact test, not assumed.
- Watchdog threshold and baseline drift with arm configuration are untested.
