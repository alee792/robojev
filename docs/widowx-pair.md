# Two WidowX AI arms

Plan for running robojev on the bay's leader/follower WidowX AI pair: two arms sorting blocks
together, first under one controller, then each with its own brain. It replaces the Panda as the main
line; the Panda notes (`docs/panda-setup.md`) stay for reference.

## What we have

| | Follower | Leader |
|---|---|---|
| Address | 192.168.1.5 | 192.168.1.3 |
| End | Gripper; D405 wrist camera | Teleop handle with finger-driven jaws |
| Driver | `trossen_arm` (v1's `src/robojev/arm/real.py`) | Same driver, not used by robojev yet |

- Scene camera: D455F above the table.
- The MacBook runs the driver directly, as v1 did. No real-time box or kernel: the arm's own
  controller runs the fast loop, and we send setpoints. The Pi 5 isn't needed; it could host the
  scene camera later.
- Reach (from v1's sim map): top-down grasps out to ~0.44 m from the base, sideways grasps further.

**Unknown that shapes everything:** whether the leader's handle jaws can hold a block when the
motor drives them (spike W1). If not, the leader is a helper that pushes, sweeps and holds things
steady, and only the follower picks.

## Safety first

- **The teleop link must be off.** In the kit, the leader's motions drive the follower. Before
  robojev commands the leader, nothing may be mirroring it to the follower, or moving one arm moves
  both. Check no teleop process is running, every session.
- Each arm has its own workspace box and speed cap (v1's `Mover` and effort watchdog, per arm).
- A **shared-zone lock in code**: only one arm inside the middle zone at a time. No model can
  override it.
- STOP parks both arms.

## Layout

Bases facing each other across the table, about 0.6–0.7 m apart. Each arm owns the half nearest it,
and the middle strip, which both can reach, is the **shared zone**. The tray sits in one arm's half
and blocks start in the other's, so moving a block across means passing it over: put it down in the
shared zone for the other arm, or hand it gripper to gripper. Cooperation is required by the layout,
not added for show.

## Crawl, walk, run

### Crawl: one brain, two arms

- One harness, one planner. Each plan step names its arm: `move_object(arm=follower, block 5,
  shared_zone)`, then `move_object(arm=leader, block 5, slot 2)`.
- Code enforces the shared-zone lock and runs both arms' steps in parallel when they don't conflict.
- Every event gets one decision per arm. The right-now group treats the other arm as one more
  moving thing in the scene. Hand-in-the-way and corrections work as in e12v2.
- **Pass:** the e12v2 core and held-out scenarios in sim with two arms, then on the real pair. No
  arm-arm contact, no hand contacts, corrections change behaviour in under 0.5 s.

### Walk: two brains, one shared world

- Each arm runs its own harness and decision loop and reads one shared world state: where the blocks
  are, and which arm has **claimed** each block and the shared zone.
- One planner still writes the plan, but each arm runs its share and adapts locally. To each arm,
  the other is "a change I didn't cause": the right-now group holds when the other arm is in the
  shared zone, and the in-plan fix skips a block the other arm has already claimed.
- **Pass:** crawl's bar with no central controller in the loop, no deadlocks, no block claimed twice.

### Run: two brains that negotiate

- No shared plan. Each arm plans its own part; conflicts settle through claims and handovers. A
  correction reaches both, and they agree who redoes what.
- **Pass:** to be set after walk. This is the research question.

## Skills

The v2 skills gain an `arm` argument, plus:

- `place_in_shared_zone(block)` and `take_from_shared_zone(block)`: handing over by putting the block
  down, the default.
- `hand_over(block, to=arm)`: gripper to gripper, only if W6 passes.
- Leader-only fallbacks if W1 fails: `push(block, toward=point)`, `sweep(area, toward=point)`,
  `hold_steady(block)`.

## Discovery spikes

| # | Question | How | Pass when |
|---|---|---|---|
| W1 | Can the leader's handle jaws hold a block? | Drive the leader's gripper joint closed on a cube; lift and move it 20 times | 19 of 20 held; otherwise leader is push-only |
| W2 | Can one Mac drive both arms at once? | Two `trossen_arm` connections from one process; simultaneous moves; measure setpoint timing | Both arms track smoothly; no dropped connections over 10 min |
| W3 | Where is each base relative to the other? | Scene camera plus a fiducial, or touch both grippers to the same marked points | Both arms agree on a point within ~1 cm |
| W4 | Can we simulate the pair? | Two WidowX AI models from `trossen_arm_mujoco` in one MuJoCo scene, behind the same skill interface | e12v2 sorting runs in sim with two arms |
| W5 | How big is the shared zone? | Reach map for both arms at the chosen spacing | A strip both reach top-down, wide enough for two blocks |
| W6 | Can the arms hand a block gripper to gripper? | 20 handovers in the shared zone | 18 of 20; otherwise hand over by putting down |
| W7 | Does the leader behave like the follower under our motion code? | Run v1's `RealArm` against .3 with teleop off: workspace box, speed cap, effort trip, park | Same behaviour as the follower |

Order: W7 and W1 (is the leader usable, and how), W2, W4 in sim alongside, then W3, W5, W6.

## Demo

The spell-it-out demo (`docs/showcase-demo.md`) works with two arms: the letters start in the
leader's half and the word is spelled in the follower's tray, so every letter crosses the shared
zone. When the audience corrects OPENAI to PANDA, both arms re-plan, and the one holding a letter
that's no longer needed puts it back.

## Open questions

- Does the leader's handle have a gripper motor robojev can command, or are the jaws only moved by
  fingers? (W1, W7)
- Decisions: Jev or OpenAI's Decisions API? The decision interface takes either. The Decisions API is
  confirmed working from the office (~117 ms p50; `docs/panda-setup.md`).
