# v2 diagrams

Companion to `docs/v2.md`. GitHub renders these.

## 1. Architecture

Perception and the user put events on a bus, and skills report back to the harness. For each event the harness sends one Jev
request (the decision) and applies the answers. LLM calls run alongside; only the harness moves the
arm.

```mermaid
flowchart TB
    subgraph perc [Perception, always running]
        cams[Cameras + depth] --> ws[(World state)]
    end
    user([User])
    bus{{Event bus}}
    subgraph harness [Harness, code]
        disp[Turn event into<br/>a decision]
        state[(Plan + world state)]
        skill[Current plan step]
        filter[Safety filter]
    end
    subgraph jev [Decision: one Jev request, ~150 ms]
        g1[Spotter group<br/>what to do right now]
        g2[Sequencer group<br/>fix within the plan?]
        g3[Router group<br/>who handles it]
    end
    subgraph llm [LLM, ~3 s, async]
        planner[Planner<br/>fast or capable model]
    end

    user -- "task, correction" --> bus
    ws -- "change the robot<br/>didn't cause" --> bus
    bus --> disp
    disp -. "event + plan position<br/>+ nearby objects" .-> jev
    jev -- "answers" --> disp
    disp -. "route: fast or<br/>capable LLM" .-> planner
    planner -- "new plan" --> state
    disp --> state --> skill --> filter --> arm([Arm])
    skill -. "step done, failed" .-> disp
    stop([STOP button]) -. "always wins" .-> filter
```

## 2. An episode

The same loop read top to bottom. Every event goes through the same decision; the route decides
whether it stays local or goes to an LLM.

```mermaid
flowchart TB
    start([Task arrives]) --> plan[Fast LLM writes a plan]
    plan --> run[Run the next plan step]
    plan -. "checked in parallel" .-> dec
    run --> ev{{Event: step done or failed,<br/>scene change, user text, new plan}}
    ev --> dec

    subgraph decision [" "]
        dec[Decision: one Jev request]
        dec --> now[Right now: carry on, hold,<br/>pause, back off, re-target, resume]
        dec --> r_local[Route: stay local]
        dec --> r_llm[Route: fast or<br/>capable LLM]
        dec --> r_user[Route: ask the user]
    end

    now -- "applied at once" --> run
    r_local -- "in-plan fix: retry, skip,<br/>re-queue, next step" --> run
    r_llm --> hold[LLM replans while the arm<br/>carries on or holds, per right now]
    hold --> run
    r_user --> paused([Paused until the user answers])
    dec -- "every done<br/>condition true" --> done([Episode over])

    classDef route fill:#fff4d6,stroke:#b8860b
    class r_local,r_llm,r_user route
    style decision fill:#fffaf0,stroke:#b8860b
```

## 3. An episode in time

"Line the numbered blocks up in the tray, lowest number on the left" (blocks 1, 3, 5, 7, 11, 12),
with a moved block, a correction and a hand.

```mermaid
sequenceDiagram
    actor U as User
    participant W as Perception
    participant H as Harness
    participant J as Jev (decision)
    participant L as Fast LLM
    participant K as Skill

    U->>H: "Line the numbered blocks up in the tray, lowest on the left"
    H->>L: task + world state
    L-->>H: plan: 1 → slot 1, 3 → slot 2, 5 → slot 3, … 12 → slot 6
    H->>K: step 1: block 1 → slot 1 (starts at once)
    H-)J: new plan: anything wrong?
    J-->>H: plan ok
    K-->>H: step done
    H->>J: step done
    J-->>H: block 1 in slot 1, next step
    W-->>H: block 5 slid aside, not by the robot
    H->>J: scene change
    J-->>H: right now: carry on, route: stay local
    H->>K: block 5 → slot 3, from its new position
    U->>H: "actually, highest on the left"
    H->>J: user text
    J-->>H: right now: hold (this move would be wrong), route: fast LLM
    H->>L: task + correction + world state + plan
    Note over H,K: Arm holds block 5 at a safe point
    L-->>H: new plan: 5 → slot 4, 12 → slot 1, … 1 and 3 move again
    H->>K: carry on with the new plan
    W-->>H: hand reaching toward the arm
    H->>J: scene change
    J-->>H: right now: pause
    Note over H,K: The code stop distance applies regardless
```

## 4. Headline demo: a correction mid-run

Jev reads the correction in ~150 ms and decides whether the arm carries on or holds; the LLM works
out the new order. Nothing about the correction was prepared in advance.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant J as Jev (decision)
    participant L as Fast LLM
    participant K as Skill

    Note over H,K: Blocks 1 and 3 in slots 1 and 2, carrying block 5 to slot 3
    U->>H: "actually, highest on the left"
    H->>J: user text + task + plan position
    J-->>H: right now: hold, in-plan fix: none, route: fast LLM (~150 ms)
    H->>L: task + correction + world state + plan
    Note over H,K: Arm holds block 5 at a safe point
    L-->>H: new plan: 5 → slot 4, move 1 and 3 aside, then 12, 11, 7 → slots 1-3, 3 → slot 5, 1 → slot 6 (a few seconds)
    H-)J: new plan: anything wrong?
    H->>K: block 5 → slot 4
```

## 5. Walk: someone moves a block, then reaches in

Changes from outside go through the same decision; the right-now answer does the work.

```mermaid
sequenceDiagram
    actor Pe as Person
    participant W as Perception
    participant H as Harness
    participant J as Jev (decision)
    participant K as Skill

    H->>K: move above block 5
    Pe->>W: slides block 5 15 cm left
    W-->>H: change: block 5 moved, not by the robot
    H->>J: scene change + plan position + nearby objects
    J-->>H: right now: re-target (0.91), route: stay local
    H->>K: move above block 5 at its new position
    Pe->>W: reaches toward the arm
    W-->>H: change: hand approaching the gripper
    H->>J: scene change
    J-->>H: right now: pause (0.88)
    Note over H,K: The safety filter's own stop distance applies regardless
    Pe->>W: hand withdraws
    W-->>H: change: hand gone
    H->>J: scene change
    J-->>H: right now: resume
```

## 6. A plan that misses something

The plan check runs in parallel with the first step. When it finds a problem, that is just another
decision routed to the LLM.

```mermaid
sequenceDiagram
    actor U as User
    participant H as Harness
    participant L as Fast LLM
    participant J as Jev (decision)
    participant K as Skill

    U->>H: "Line the numbered blocks up in the tray, lowest on the left"
    H->>L: task + world state
    L-->>H: plan: 1, 3, 5, 11, 12 (missed block 7, half hidden)
    H->>K: step 1: block 1 → slot 1 (starts at once)
    H-)J: new plan: per block, does the task ask about it but the plan leave it out?
    J-->>H: block 7: yes (0.91), route: fast LLM
    H->>L: task + world state + plan + "block 7 missing"
    L-->>H: plan patched: 7 → slot 4, 11 and 12 shift right
    Note over H,K: Step 1 was never interrupted
```

## 7. What the arm is doing

Only the harness moves the arm between these states. A model can pause it; only the STOP button,
"stop" typed by the user, or the safety filter stops it.

```mermaid
stateDiagram-v2
    [*] --> Waiting: no task yet
    Waiting --> Running: first plan arrives
    Running --> Paused: decision says pause
    Paused --> Running: decision or user says resume
    Running --> Holding: replan, and the decision says hold
    Holding --> Running: new plan
    Running --> Stopped: STOP button, typed stop, or safety filter
    Paused --> Stopped: STOP button or typed stop
    Holding --> Stopped: STOP button or typed stop
    Stopped --> Waiting: user restarts
    Running --> Waiting: episode over
```
