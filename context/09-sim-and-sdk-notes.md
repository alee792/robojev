# Base research: WidowX sim, bay, and the TypeSafe SDK

Compiled 2026-09-17 by two research sub-agents that read source on disk and in the SDK repos. Nothing
was run on hardware. **[src]** = read in code; **[inf]** = inference. File paths are on Anthony's
machine.

## TypeSafe Python SDK (`typesafe-sdk` 0.6.0) — mind the defaults
- HTTP client is `httpx2`. It is **HTTP/1.1 by default**; pass your own `http_client=` for HTTP/2 or
  custom connection limits. [src]
- **The default retry policy would break a control loop**: `max_retries=2`, backoff 0.5 s → 5 s,
  retries on 408/429/5xx/connection errors, honors `Retry-After`, **30 s total budget**, and a **10 s
  per-attempt timeout**. For a loop, use `RetryPolicy(max_retries=0)` with a timeout of a few hundred
  ms, or call the raw HTTP API as `experiments/common.py` does. [src]
- Exposes `request_id`; no streaming. Concurrent awaits on one async client are fine. [src/inf]
- Not deterministic. The docs' self-consistency cookbooks report a mean per-question std of about
  0.01 on Nouls, and Choice labels repeating 90.8% of the time (99.2% after gating on top probability
  ≥ 0.60). They put a fresh `uid` in the state for each repeat, which **[inf]** hints identical
  requests may be cached. [doc]
- No sampling parameters (temperature/seed) are documented. [doc]
- Confidence formula: see `08-experiment-results.md` §E4 (confirmed empirically for Choice).

## MuJoCo sim of the WidowX AI
- **Upstream:** `~/trossen_arm_mujoco/` (TrossenRobotics/trossen_arm_mujoco), already cloned. Plugin
  repo: `~/code/mission-robotics/plugins/widowx/` (skills, `src/widowx/`, `bin/mission`). The
  `widowx:sim` skill is in `~/.claude/plugins/cache/mission-robotics/widowx/0.1.0/skills/sim/`
  (`setup_sim.sh`, `run_demo.sh`, `headless_check.py`).
- **Runs on macOS.**
  - The viewer demos (`pick_place`, `follow_target`) use `launch_passive` and can't run headless.
  - Headless: load `assets/wxai/scene.xml` and step it yourself. Offscreen `mujoco.Renderer` works on
    macOS; Linux needs `MUJOCO_GL=egl|osmesa`.
  - Physics timestep is MuJoCo's default 2 ms, so a 5–10 Hz decision loop spans about 20–50 physics
    steps.
- **Ground truth available** via `MjModel`/`MjData`: `qpos`/`qvel`, end-effector pose from site
  `ee_site` (`Controller.get_ee_pose()`), object `xpos`/`xquat`, contacts (`data.contact`,
  `mj_contactForce`), and a wrist D405 camera `cam`.
- **Scenes** (`assets/wxai/`):
  - `scene.xml`: the arm alone.
  - `scene_wxai_pick_place.xml`: a 5 cm free cube.
  - `scene_wxai_follow_target.xml`: a mocap target cube, a natural fit for a reactive first task.
  - Objects can be repositioned through the freejoint `qpos`. Adding objects means composing XML.
- **Control:**
  - Position servos via `data.ctrl` (6 joints plus gripper; gripper ctrl range 0–0.044).
  - `controller.py` provides one-step damped-least-squares IK, `set_ee_pose(pos, quat_wxyz)`, and
    `open_gripper`/`close_gripper`. The demo iterates until error < 0.04 or 100 steps.
  - **No pre-motion collision checking**; contacts are only visible after the fact.
  - Servo gains are Trossen defaults and not tuned to the real arm.
- **Twin:** `mission twin build` builds a scene from `bay.yaml` (`src/widowx/twin/scene.py`), but most
  bay measurements are still null.

## Frames and conventions
- **Real arm:** `[x, y, z, roll, pitch, yaw]` of the end effector in the arm base frame, in m and rad;
  +x forward, +y left, +z up. A top-down grasp uses pitch = 1.57. Gripper 0 (closed) to 0.040 m.
  Joint order: base_yaw, shoulder, elbow, wrist_pitch, wrist_yaw, wrist_roll, gripper.
  (`skills/move/SKILL.md`, `references/grasping.md`, `src/widowx/cartesian.py`, `src/widowx/arms.py`)
- **Sim:** orientation is a wxyz quaternion (pointing down is `[0.707, 0, 0.707, 0]`) and gripper width
  is 0.022–0.044. **A sim↔real conversion layer is needed.**
- The experiments' scene frame (+x forward, +y left, positive bearing = left) matches this convention.

## Sim vs real arm control
- **Not drop-in.** Real motion goes through `ArmSession` (`src/widowx/arms.py`), which wraps
  `trossen_arm.TrossenArmDriver` 1.9.3 in position mode: `set_all_positions(goal, goal_time,
  blocking=False)`, `set_cartesian_positions(pose, interpolation, goal_time)`, `set_gripper_position`,
  and the `get_*` readers. IK runs on the arm's controller.
- **The existing CLI/MCP moves are unsuitable for a fast loop.** Each one connects, moves, sleeps
  `goal_time + 0.1`, then disconnects. A loop should hold one `ArmSession` open and send non-blocking
  setpoints. `driver_factory` allows swapping in a sim-backed fake.
- **Safety gating today:**
  - The `mission_*` tools and CLI have consent turned off (`src/mission/consent.py` `Unguarded`).
  - The older `mcp__widowx__*` server asks the user before every move (`ctx.elicit`), with an optional
    session-wide standing approval.
  - Trajectory runs must pass a MuJoCo gate: joint limits, velocity budget, no self or floor contact.
  - **Nothing gates Cartesian moves.**
- ⚠️ Upstream `replay_episode_real.py` defaults to 192.168.1.5, which is this bay's follower arm. Don't
  run it casually.

## Bay (`plugins/widowx/skills/twin/bay.yaml`, `src/widowx/bays/default-bay.json`)
- Bay `widowx-1`, leader/follower pair on 192.168.1.0/24: leader .3, **follower .5** (at the bay
  origin), firmware 1.9.3.
- Cameras:
  - Scene: D455F serial 408222301818, 1280×720, position not yet measured.
  - Wrist: D405 serial 230422271405, mounted on the follower.
- Table size, obstacles, fiducials (4 tag36h11 corner tags plus a 9×7 ChArUco board) and safety zones
  are all **null (unmeasured)**. Calibration commands are stubs.
