# v2 diagrams

Companion to `docs/v2.md`. GitHub renders these.

## 1. The loop

Who wakes up on what, and where decisions go. Jev layers run only when their trigger fires.

```mermaid
flowchart TB
    user([User]) -- "task" --> planner
    user -- "correction" --> router
    stop([STOP button]) -. "always wins" .-> filter

    subgraph perception [Perception]
        cams[Cameras + depth] --> ws[(World state)]
        ws --> detect{"Change the robot<br/>didn't cause?"}
    end

    subgraph jev [Jev layers]
        spotter[Spotter<br/>classify the change,<br/>pick an intervention]
        seq[Sequencer<br/>check progress<br/>against the task]
        router[Router<br/>continue / adjust /<br/>fast LLM / capable LLM / stop]
    end

    subgraph llm [LLM]
        planner[Planner<br/>fast or capable model]
    end

    subgraph code [Code]
        plan[(Plan +<br/>parameters)]
        exec[Plan runner]
        skills[Skill]
        filter[Safety filter]
    end

    detect -- "yes" --> spotter
    spotter -- "intervention" --> exec
    spotter -- "can't place it" --> router
    skills -- "done / failed" --> seq
    seq -- "next step, retry, skip" --> exec
    seq -- "plan looks wrong" --> router
    router -- "adjust: set parameter" --> plan
    router -- "hand off" --> planner
    planner -- "new or repaired plan" --> plan
    plan --> exec --> skills --> filter --> arm([Arm])
    arm --> ws
```

## 2. Crawl: the normal cycle

No surprises: the plan runs and the Sequencer checks each step against the task.

```mermaid
sequenceDiagram
    actor U as User
    participant P as Planner (LLM)
    participant R as Plan runner (code)
    participant K as Skill (code)
    participant S as Sequencer (Jev)

    U->>P: "Sort the blocks by colour into the bins"
    P->>R: Plan: pick red_1 → red bin, pick blue_1 → blue bin, …
    loop each step
        R->>K: start(pick, red_1)
        K-->>R: done
        R->>S: world state + task + plan position
        S-->>R: carry on (0.93)
    end
    S-->>R: task done
```

## 3. Headline demo: "actually, reverse it"

The fast path acts within ~150 ms; the LLM handles only the ambiguous part, in parallel.

```mermaid
sequenceDiagram
    actor U as User
    participant Ro as Router (Jev)
    participant R as Plan runner (code)
    participant K as Skill (code)
    participant P as Fast LLM

    Note over R,K: Blocks 1, 2 in the tray, carrying 3
    U->>Ro: "actually, reverse it"
    Ro-->>R: adjust: sort order = descending, after this block (~150 ms)
    Ro-->>P: blocks already in the tray: undo them?
    R->>K: finish placing 3
    R->>K: next: pick 6 (code recomputed for descending)
    P-->>R: amend plan: take 2 and 1 back out first (~1-3 s)
    Note over R: Arm never waited for the LLM
```

## 4. Walk: someone moves the target

Perception notices a change the robot didn't cause; the Spotter decides what it means.

```mermaid
sequenceDiagram
    actor H as Person
    participant W as Perception (code)
    participant Sp as Spotter (Jev)
    participant R as Plan runner (code)
    participant K as Skill (code)

    R->>K: move_above(red_2)
    H->>W: slides red_2 15 cm left
    W->>W: not caused by the robot → flag it
    W->>Sp: change: red_2 moved while the arm was going for it
    Sp-->>R: intervention: re-target red_2 (0.91)
    R->>K: move_above(red_2) at its new position
    H->>W: hand reaches toward the arm
    W->>Sp: change: hand approaching the gripper
    Sp-->>R: intervention: pause (0.88)
    Note over R,K: The safety filter's own stop distance applies regardless
    H->>W: hand withdraws
    W->>Sp: change: hand gone
    Sp-->>R: intervention: resume
```

## 5. A wrong plan, caught by the Sequencer

Nothing outside changed: the fast model's plan was wrong, and checking against the task catches it.

```mermaid
sequenceDiagram
    actor U as User
    participant P1 as Fast LLM
    participant R as Plan runner (code)
    participant S as Sequencer (Jev)
    participant Ro as Router (Jev)
    participant P2 as Capable LLM

    U->>P1: "Put the blocks in the tray, biggest first"
    P1->>R: Plan: sorted by number, not by size
    R->>R: runs the first step
    R->>S: task says "biggest first", tray now holds a small block
    S-->>Ro: progress doesn't match the task (0.84)
    Ro-->>P2: re-plan: plan contradicts the task
    Note over R: Holds at a safe point
    P2->>R: Plan: sorted by size, put the small block back first
```

## 6. What the arm is doing

Only code moves the arm between these; models can request a pause but not a hard stop.

```mermaid
stateDiagram-v2
    [*] --> Running
    Running --> Paused: Spotter or Router pause
    Paused --> Running: Router or user resumes
    Running --> Holding: waiting for an LLM
    Holding --> Running: new or repaired plan
    Running --> Stopped: STOP button or safety filter
    Paused --> Stopped: STOP button
    Holding --> Stopped: STOP button
    Stopped --> Running: user restarts
    Running --> [*]: task done
```
