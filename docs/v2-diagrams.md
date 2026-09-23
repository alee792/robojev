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
        seq[Sequencer<br/>checks the plan,<br/>then progress]
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
    disp -. "new plan, skill<br/>done or failed" .-> seq
    disp -. "correction, skill failed, not on track,<br/>can't place, low confidence" .-> router
    disp -. "no plan yet, hand off" .-> planner
    spotter -- "intervention" --> bus
    seq -- "plan ok, next step,<br/>or not on track" --> bus
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
    plan --> check[Sequencer checks the plan<br/>against the task]
    check -- "ok" --> run[Run the next skill]
    check -- "something's wrong" --> replan[Capable LLM replans]
    replan --> check
    run --> seq[Sequencer checks progress,<br/>one question per object]
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
    patch --> check
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
    H->>S: plan + task + world state: anything wrong?
    S-->>H: plan ok (~150 ms)
    loop each step
        H->>K: start(pick red_1 → left bin)
        K-->>H: done (event)
        H->>S: per object: does the task want it in the left bin?
        S-->>H: answers (event), code compares with the world state
    end
```

## 4. Headline demo: "actually, reverse it"

The task is "line the numbered blocks up in the tray, lowest on the left". The correction changes the
final arrangement, so the sort order parameter flips. Jev maps the words onto the parameter within
~150 ms; code works out every move from there, including the blocks already placed. No LLM.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant Ro as Router (Jev)
    participant K as Skill

    Note over H,K: Blocks 1, 2 in the tray, carrying 3
    U->>H: "actually, reverse it"
    H->>Ro: correction + task + parameters
    Ro-->>H: adjust: sort order = highest on the left (stated: yes), after this block (~150 ms)
    Note over H: Code recomputes every goal slot. 1 and 2 are now "not at goal"
    H->>K: finish placing 3 (in its new slot, 4)
    H->>K: 4 to slot 3 (next block whose slot is free)
    H->>K: move 1 and 2 out of the slots 6 and 5 need
    H->>K: 6, 5, 2, 1 into their slots
    Note over H: The arm never waited, and no LLM was called
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
The Sequencer's check catches it before anything moves; the per-object check after each skill is
the backstop if it slips through.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant P1 as Fast LLM
    participant S as Sequencer (Jev)
    participant P2 as Capable LLM

    U->>H: "Put all the red blocks in the left bin"
    H->>P1: task + world state
    P1-->>H: plan: red_1, red_2 → left bin (missed red_3)
    H->>S: per object: does the task ask about it but the plan leave it out?
    S-->>H: red_3: yes (0.91), others: no
    H->>P2: task + world state + plan + "red_3 missing"
    Note over H: Arm still waiting for its first plan
    P2-->>H: plan: red_1, red_2, red_3 → left bin
    H->>S: check again
    S-->>H: plan ok
    Note over H: First skill runs
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
