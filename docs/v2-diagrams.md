# v2 diagrams

Companion to `docs/v2.md`. GitHub renders these.

## 1. Architecture: the event bus

Perception, the user and skills put events on a bus. The harness takes each event, asks the right
layer (async), and applies the answer. Only the harness moves the arm, and perception sees the result.

```mermaid
flowchart TB
    subgraph perc [Perception, always running]
        cams[Cameras + depth] --> ws[(World state)]
    end
    user([User])
    bus{{Event bus}}
    subgraph harness [Harness, code]
        disp[Dispatcher]
        state[(Plan + world state)]
        skill[Current skill]
        filter[Safety filter]
    end
    subgraph jev [Jev, ~150 ms, async]
        spotter[Spotter<br/>changes from outside]
        seq[Sequencer<br/>progress vs the task]
        router[Router<br/>who handles it]
    end
    subgraph llm [LLM, 1 s or more, async]
        planner[Planner<br/>fast or capable model]
    end

    user -- "task, correction" --> bus
    ws -- "world updated,<br/>change the robot didn't cause" --> bus
    skill -- "done, failed" --> bus
    bus --> disp
    disp -. "change" .-> spotter
    disp -. "skill done or failed" .-> seq
    disp -. "correction, skill failed, not on track,<br/>can't place, low confidence" .-> router
    disp -. "no plan yet, hand off" .-> planner
    spotter -- "intervention" --> bus
    seq -- "next step, or not on track" --> bus
    router -- "route" --> bus
    planner -- "plan or patch" --> bus
    disp --> state --> skill --> filter --> arm([Arm])
    stop([STOP button]) -. "always wins" .-> filter
```

## 2. Workflow: start-up and execution

The same loop with the bus left out, read top to bottom. The main path runs one skill at a time; the
box on the side can fire at any moment, in parallel with it.

```mermaid
flowchart TB
    start([Start]) --> boot[Start perception and the harness]
    boot --> task[/User gives a task/]
    task --> plan[Fast LLM writes a plan]
    plan --> run[Run the next skill]
    run --> seq[Sequencer checks progress<br/>against the task]
    seq -- "next step, or retry" --> run
    seq -- "task done" --> done([Wait for the next task])
    seq -- "not on track, or unsure" --> r_in

    subgraph router [" "]
        r_in{{Router: pick one route}}
        r_in --> r_go[Continue]
        r_in --> r_adj[Adjust a<br/>parameter]
        r_in --> r_llm[Fast or<br/>capable LLM]
        r_in --> r_stop[Pause, or<br/>ask the user]
    end
    r_go --> run
    r_adj --> run
    r_llm --> patch[LLM patches the plan<br/>arm carries on or holds<br/>at a safe point]
    patch --> run
    r_stop --> paused([Paused until the user answers])

    subgraph anytime [At any moment, in parallel]
        change[/Perception: a change<br/>the robot didn't cause/] --> spotter[Spotter picks<br/>an intervention]
        corr[/User types a correction/]
    end
    spotter -- "re-target, re-queue, pause,<br/>back off, resume, ignore" --> run
    spotter -- "can't place it" --> r_in
    corr --> r_in

    classDef route fill:#fff4d6,stroke:#b8860b
    class r_go,r_adj,r_llm,r_stop route
    style router fill:#fffaf0,stroke:#b8860b
```

## 3. Start-up and the normal cycle, in time

Perception and the harness start first. With no plan, the harness asks the fast LLM for one; the
first skill runs as soon as it arrives, and everything after that is event-driven.

```mermaid
sequenceDiagram
    actor U as User
    participant W as Perception
    participant H as Harness
    participant P as Fast LLM
    participant K as Skill
    participant S as Sequencer (Jev)

    U->>H: "Put the red blocks in the left bin"
    par always running
        W-->>H: world state updates
    and no plan yet
        H->>P: task + world state
        P-->>H: plan: red_1, red_2, red_3 → left bin (~1 s)
    end
    loop each step
        H->>K: start(pick red_1 → left bin)
        K-->>H: done (event)
        H->>S: task + world state + plan so far
        S-->>H: on track, next step (event)
    end
```

## 4. Headline demo: "actually, reverse it"

The task is "line the numbered blocks up in the tray, lowest on the left". The correction changes the
final arrangement, so the sort order parameter flips. The fast path acts within ~150 ms; only the
ambiguous part goes to the LLM, in parallel.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant Ro as Router (Jev)
    participant K as Skill
    participant P as Fast LLM

    Note over H,K: Blocks 1, 2 in the tray, carrying 3
    U->>H: "actually, reverse it"
    H->>Ro: correction + task + parameters
    Ro-->>H: adjust: sort order = highest on the left, after this block (~150 ms)
    H-)P: blocks 1 and 2 sit in the slots 6 and 5 need: plan a fix
    H->>K: finish placing 3 (in its new slot, 4)
    H->>K: next: block 4 to slot 3 (code: the next block whose slot is free)
    P-->>H: patch: move 1 and 2 out, then place 6 and 5 (~1-3 s)
    Note over H: The arm never waited for the LLM
```

## 5. Walk: someone moves the target, then reaches in

Perception flags changes the robot didn't cause; the Spotter decides what each one means.

```mermaid
sequenceDiagram
    actor Pe as Person
    participant W as Perception
    participant H as Harness
    participant Sp as Spotter (Jev)
    participant K as Skill

    H->>K: move_above(red_2)
    Pe->>W: slides red_2 15 cm left
    W-->>H: change: red_2 moved, not by the robot
    H->>Sp: change + task + current skill
    Sp-->>H: intervention: re-target red_2 (0.91)
    H->>K: move_above(red_2) at its new position
    Pe->>W: reaches toward the arm
    W-->>H: change: hand approaching the gripper
    H->>Sp: change + current skill
    Sp-->>H: intervention: pause (0.88)
    Note over H,K: The safety filter's own stop distance applies regardless
    Pe->>W: hand withdraws
    W-->>H: change: hand gone
    H->>Sp: change
    Sp-->>H: intervention: resume
```

## 6. A plan that misses something

Nothing outside changed: the fast LLM's plan left out a block that was half hidden behind the bin.
The plan finishes, but checking against the task (not the plan) shows the task isn't done.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant P as Fast LLM
    participant S as Sequencer (Jev)
    participant Ro as Router (Jev)

    U->>H: "Put all the red blocks in the left bin"
    H->>P: task + world state
    P-->>H: plan: red_1, red_2 → left bin (missed red_3)
    Note over H: Both steps run and succeed
    H->>S: task + world state: red_3 still on the table
    S-->>H: plan finished, task not done (0.86)
    H->>Ro: plan finished, task not done
    Ro-->>H: route: fast LLM
    H->>P: task + world state + plan so far
    P-->>H: patch: add red_3 → left bin
```

## 7. What the arm is doing

Only the harness moves the arm between these states. A model can pause it; only the STOP button,
"stop" typed by the user, or the safety filter stops it.

```mermaid
stateDiagram-v2
    [*] --> Waiting: harness starts, no plan
    Waiting --> Running: first plan arrives
    Running --> Paused: Spotter or Router pause
    Paused --> Running: Spotter, Router or user resumes
    Running --> Holding: waiting on an LLM
    Holding --> Running: new plan or patch
    Running --> Stopped: STOP button, typed stop, or safety filter
    Paused --> Stopped: STOP button or typed stop
    Holding --> Stopped: STOP button or typed stop
    Stopped --> Waiting: user restarts
    Running --> Waiting: task done
```
