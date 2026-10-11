# Architecture

What we're building toward, on one or two WidowX AI arms. The design rationale is in `docs/v2.md`;
the spikes that de-risk it are in `docs/harness-spikes.md`.

```mermaid
flowchart TB
    user([You: text or voice]) --> bus
    subgraph mac [One Mac]
        subgraph perc [Perception process]
            cams[Scene + wrist cameras] --> ws[(World state:<br/>blocks, hands, arms)]
        end
        bus{{Event bus}}
        subgraph brain [Harness process: the brain]
            dec[Decision per event]
            plan[(Plan: skill steps<br/>+ done conditions)]
            rules[Code rules: gate on confidence,<br/>STOP, replan-loop limit]
            rec[(Recorder)]
        end
        subgraph motor [Skill server process: the body]
            skills[Skills: pick_and_place, stack_on,<br/>push, hand_over, survey, hold]
            safety[Safety filter: workspace box, speed cap,<br/>effort trip, shared-zone lock, heartbeat]
            be[Backend: MuJoCo sim or trossen_arm]
        end
        dash[Dashboard]
    end
    subgraph cloud [Cloud]
        llm[OpenAI planner<br/>seconds, async]
        dapi[Decisions API or Jev<br/>~120-150 ms]
    end
    ws -- "scene change" --> bus
    bus --> dec
    dec <-. "one request per event" .-> dapi
    dec --> rules --> plan
    rules -. "replan" .-> llm
    llm -. "new plan" .-> bus
    plan -- "skill call (arm, args)<br/>hold, pause, resume" --> skills
    skills -- "step done, failed" --> bus
    skills --> safety --> be
    be --> arms([WidowX arm 1, optional arm 2])
    stop([STOP]) --> safety
    brain --- dash
```

## Parts

Everything runs on one Mac; the WidowX needs no real-time box.

1. **Perception process.** Cameras to a world state: blocks, hands, arm poses. A change the robot
   didn't cause becomes an event. In sim, the world state comes from MuJoCo directly.
2. **Harness process (the brain).** The v2 design. Events (user text, scene change, step done or
   failed, new plan) go on a bus; each gets one decision request with three groups (right now,
   in-plan fix, who handles it). Code combines the answers, gates on confidence and applies the rules
   (STOP, models can only pause, replan-loop limit). An OpenAI model writes and fixes the plan
   asynchronously; a plan is skill calls plus done conditions. A recorder logs everything.
3. **Robot server process (the body).** Motor skills: `pick_and_place`, `stack_on`, `push`,
   `hand_over`, `survey`, `hold`. Under them a safety filter (workspace box, speed cap, effort trip,
   shared-zone lock, heartbeat), and under that a backend: MuJoCo sim or the real `trossen_arm`
   driver.
4. **Cloud.** The decision model (OpenAI's Decisions API or Jev, ~120-150 ms) and the planner (an
   OpenAI model, seconds).

## Hardware-agnostic: a protocol, not a library

The brain never knows which robot it drives. Each robot (sim WidowX, the bay's WidowX pair, a Panda,
a YAM) runs its own **robot server** speaking one protocol, shaped like MCP:

- **Manifest** (MCP `initialize` + `tools/list`): the arms with their workspace, gripper limits and
  base frame; the skills offered, each with a JSON schema for its arguments. The planner's output
  schema is generated from the manifest, so a robot that can't `hand_over` is never asked to.
- **Tools:** `start(arm, skill, args)`, `hold`, `pause`, `resume`, `retarget`, `stop`, `heartbeat`.
- **Resources:** the world state; each running skill's status with the literal phase text the
  decision loop reads.
- **Notifications:** skill done or failed, safety trip, scene change, heartbeat lost.

On top of the protocol sits a **catalog** (`src/robojev/catalog.py`): fixed names and
argument shapes for the standard skills (`pick_and_place`, `stack_on`, `push`, `hand_over`, `survey`,
`hold`), standard failure codes, and the control tools and resources every server must have. A
robot advertises the standard skills it can do, unchanged, and anything else under a prefix
(`widowx.wiggle_free`). Shared names are what let planner prompts, evaluations and recorded
training data transfer between robots. A conformance suite (K2's trials, D1 on hardware) proves a
server does the standard skills.

**Images.** The planner gets the camera frame with object ids drawn on it, next to the written
world state, so it can catch what a description misses and notice when the two disagree. The
decision model gets a frame only for questions about the physical scene (a hand near the gripper,
a grasp succeeded, the plan matches the table): a slower visual loop beside the ~120 ms text one.
Models judge from images; perception still measures: positions in metres come from perception.

Skills can be as fine-grained as the work needs. A precision task gets skills with explicit
tolerance arguments that verify their own result before reporting done (`insert` is done when the
part is seated and the force says so, not when the arm reaches a pose). A low-precision arm (a
hobby-grade WidowX: millimetre repeatability, backlash, a compliant gripper) needs closed-loop
skills: look, move, look again, correct. Verification is part of the skill contract.

Skill granularity is the protocol boundary. The policy inside a skill (10-100 Hz) and the motor loop
(~1 kHz) never cross it. Units are metres and seconds; frames are declared in the manifest.

The interface is `protocol.py` (in-process Python, each method mapped to its MCP counterpart), and
since K6 it is also a real MCP server: every manifest skill is an MCP tool with the catalog schema,
the control tools are `hold_arm`, `pause_arm`, `resume_arm`, `retarget_skill`, `stop`, `heartbeat`
and `precondition`, the world and skill statuses are resources, and events are a server-to-client
notification. The brain's tests pass unchanged over a stdio subprocess; event delivery is 2.5 ms
p95. The
planner LLM does not call tools directly: it emits a plan the harness executes, so the decision loop
and the safety rules always sit between the model and the robot.

## Where this sits: on top of the Model Hardware Standard

Anthropic's Model Hardware Standard (MHS, research preview Aug 2026) is a standard *driver*:
read/write primitives on a named device, discovery, a reference file (what it measures, what can be
adjusted, the safety limits the driver enforces, plain-language notes), reachable over MCP, a CLI
or code. For anything long-running the agent writes a script that chains driver commands. There
are no skills in it and no fast reactive layer. So:

| Layer | MHS | Here |
|---|---|---|
| Device: read a pose, write a setpoint, enforce limits | the driver | the arm backend |
| Describing the device to an agent | reference file + tags | the manifest |
| Skills with contracts | absent (agent-written scripts) | the catalog |
| Reacting in ~150 ms to a person | absent | the decision loop |

We build the two layers MHS lacks and shape our arm backend like an MHS driver (`read(pose)`,
`write(setpoint)`, `read(limits)`), so a vendor's or LeRobot's MHS driver can slot in underneath
later. One line: standard skills and a reactive loop, on top of MHS, exposed through MCP.

## Lessons from MCP friction, built in

| Pain with MCP servers | Here |
|---|---|
| Tool bloat eats context and gets mis-called | The catalog grows as the work needs (fine-grained skills like `align`, `insert`, `nudge` belong in it); what stays small is what the planner sees per call: the manifest narrows it to this robot, the brain to this task, and the planner reads a short description, never raw schemas |
| Long-running calls: progress and cancel bolted on | Nothing blocks: `start` returns an id, status is a resource, completion is an event; hold / pause / stop have defined semantics |
| Stateful servers, clients that assume otherwise, reconnects lose state | The server owns the state; a reconnecting brain reads world and status and carries on; control calls are idempotent |
| "Bad args", "failed" and "world changed" all come back as text | `precondition` is separate from `start`; failures carry a catalog code plus literal text; events carry the skill id |
| Tool descriptions as an injection surface | Manifest text is data; the brain renders its own planner description; code rules never read it |
| Schema drift, no versioning | The manifest carries a catalog version; a non-conforming server is refused at connect |
| Nested schemas the model mis-fills | Flat arguments, few fields, enums |
| Untestable without a model | Conformance trials and the stub server run with no model |
| No observability | The recorder logs every skill tick and event, keyed by skill id |

## Seams

- **Robot protocol** (above): the same surface fronts sim, the real arm, a second arm, another
  make of arm and, later, a learned policy behind a skill.
- **Arm backend (MHS-shaped):** write a setpoint (position, pitch, yaw, speed cap, gripper width);
  read the pose, forces and limits. Sim and real implement it; an MHS driver could.
- **World state:** objects with id, label, position, and who holds or has claimed them; hands; arm
  poses.
- **Decision backend:** questions in, answers with confidence out. Jev or the Decisions API.
- **Planner:** task plus world state in, a strict-JSON plan or plan diff out.

## Choices

1. Skills are scripted motor routines; reasoning stays in the LLM. A learned policy can replace a
   skill behind the same API later.
2. Harness and skill server are separate processes; safety lives in the skill server, out of reach
   of any model or harness bug.
3. One decision request per event with three question groups. The Decisions API is the default; Jev
   stays swappable.
4. The planner runs alongside; meanwhile the arm carries on or holds, as the right-now answer says.
5. Two arms: one harness drives both through one skill server, which owns the shared-zone lock. Two
   brains (two harnesses, one shared world state) come later.
6. Sim first, with ground-truth positions, before camera perception.
7. Code layout (done 2026-10-10, after L1 passed): the harness slice was promoted from
   `experiments/skills_sim/` into `src/robojev`, with e12v2's pure functions copied in as the brain's
   core; `experiments/e12v2/` stays whole as history. Where things live:
   - `robojev.protocol`, `robojev.catalog`: the robot protocol (MCP-shaped, in-process Python) and
     the standard skills, failure codes, control tools and resources.
   - `robojev.brain`: the brain, a client of the protocol on an asyncio clock (`loop`, `tracker`,
     `rules`, `arm`, `changes`, `connect`, `planner`, `schema`, `steps`, `adapt`, `text`,
     `messages`). `brain/core/` is e12v2's design logic copied verbatim (data, decision, combine,
     plan, planner, changes, safety, interfaces); `brain/eval/` is e12v2's evaluation oracle and mock
     models, and `brain/mocks.py` the offline stand-ins over a robot server; the brain never imports
     either. The brain imports nothing from `robojev.robots`, `robojev.arm` or `robojev.perception`
     (checked by an AST test in `tests/test_brain.py`).
   - `robojev.robots`: the robot servers. `stub` (physics-free), `widowx_sim/` (the MuJoCo WidowX:
     `scene`, `skills`, `server`), `mcp/` (`server`: any `RobotServer` as an MCP server; `client`:
     the brain's `MCPRobotServer` over stdio or in-process streams).
   - `robojev.recording`: the server-side JSONL recorder and the LeRobot-style export.
   - `robojev.conformance` (`robojev-conformance`): N trials of a standard skill judged from the
     world state; D1 points it at the real backend.
   - `robojev.arm`, `robojev.config`, `robojev.perception`: v1's motion code (`Mover`, `Workspace`,
     the effort watchdog; `RealArm` backs the real server at D1), limits, and perception, reused.
   - v1's per-tick system (`v1_brain`, `loop`, `skills`, `questions`, `state`, `world`, `events`,
     `jev`) is retiring but still behind the `robojev` CLI, the dashboard and `replay`; it goes when
     the CLI is rewritten around the brain.

## Skills and policies

A **policy** maps what the robot senses now to its next action, run in a loop. It can be hand-written
(inverse kinematics toward a target) or learned (a network trained from demonstrations or trial and
error; a VLA is a learned policy that reads images and an instruction).

A **skill** is a named, limited action with a contract: inputs (`pick_and_place(block 5, slot 2)`),
preconditions, done or failed with a reason, and safe interrupt points (hold, pause, resume,
re-target). Inside each skill is a policy; ours start hand-written and read positions from the world
state each tick. The planner chooses which skills run in what order; the decision loop adapts that
as things happen. Recorded runs of the hand-written skills are the training data for learned ones.

| Layer | Decides | How often | Ours |
|---|---|---|---|
| Planner | Which skills, in what order | On a task or correction (seconds) | OpenAI model |
| Decision loop | Carry on, hold, fix, replan | Every event (~150 ms) | Decisions API / Jev |
| Skill | The next arm position | 10-100 times a second | Hand-written, learned later |
| Motor controller | Motor currents | ~1 kHz | Inside the arm |

Not decided yet: the demo script, speech input, learned policies.
