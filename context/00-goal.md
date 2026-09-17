# robojev — goal and decisions

**Goal:** Drive a WidowX AI arm from natural-language prompts, using Jev's fast typed judgments over
text built from observations (RealSense RGB-D and arm telemetry). Prompts can be tasks or behavior
changes, and they should take effect without code changes. How observations become state, which
questions get asked, how often, and how answers become motion are all open design (`05-design-space.md`).

**Non-goals (for now):** hardware and driver integration details; training a learned policy; putting
images into Jev (it is text-only).

**Success, first milestone (proposed):** the arm completes a simple tabletop task from an NL prompt,
driven by a closed Jev loop. A live standing order (e.g. "stay away from the blue block") visibly
changes its behavior mid-run, the way "Do not fire, simply dodge" did in the Doom demo.

## Decision: first approach = Doom-style composed loop (2026-09-17)
Chosen by Anthony as the **first** approach to try. It is a hypothesis to test, not the architecture.

The loop: a situation report, then a battery of small typed judgments several times a second, code
composing the answers into commands, and live standing orders. See `02`–`04` for how the demo does it.

**Why this first:**
- It's the only pattern with public evidence of Jev running at a high rate.
- Standing orders are what make the natural-language part distinctive. Picking one of a few
  primitives at about 1 Hz is already done (`06-prior-art.md`).
- Its core rule fits a robot anyway: Jev picks from options code has already validated, and code
  owns all geometry and side effects.

**Known adaptations it needs:**
- **Perception:** Doom reads perfect engine state; start with sim ground truth.
- **Motion:** smooth control and safety run in code at their own rates, with Jev as a slower layer
  on top.

**Revisit if:**
- Latency or rate limits can't sustain a useful decision rate.
- Answers flip between ticks enough to cause chatter.
- Event-driven or skill-level control proves simpler for the first task.

`05-design-space.md` keeps the alternatives open.

## Open decisions for Anthony (these shape the architecture)
1. **First task:** pick-and-place, or something reactive (follow/track an object, handover, avoid a
   moving obstacle)? Reactive tasks show off 10 Hz. Pick-and-place is easier to evaluate.
2. **NL scope:** task commands, standing-order behavior modifiers, or both?
3. **Perception to start:** sim ground truth → fiducials → detector+depth → VLM captions. Which rung
   first?
4. **Sim first or bay first?** A MuJoCo twin (`widowx:twin`, `widowx:sim`) exists in this environment.
5. **Target decision rate:** 10 Hz like Doom, event-driven, or skill-level at about 1 Hz like
   jev-askable-arm? The account limit is 20 req/s total.

## Status
Research only (2026-09-17). Handoff to Fable/Astra for architecture and implementation, with
creative license. See `README.md`.

## Decision: Jev as the policy, not a guard (2026-09-17, evening)
Anthony's direction after the first day of building. In Doom, Jev picks every channel every 100 ms
and the picks matter because the world changes between ticks. In a static tabletop scene the same
architecture degenerates into a constant ("approach, high, slow") and looks like a guard. The
pattern is not the difference; the dynamics are.

So the goal is oriented toward **Jev as a reactive policy over a factored, Doom-dense action space
in a scene that changes at the decision rate**:
- Channels re-judged every tick: TARGET (which object now), MOVE (toward target / hold / away /
  toward another entity), EVADE (none / up / left / right / back / away from X; overrides MOVE),
  GRIP (open / close / hold, offered only when code's preconditions allow), SPEED and HEIGHT as
  modifiers, LOOK (which entity to inspect) later.
- Code resolves every pick to a setpoint from the latest geometry, smooths, and enforces limits.
  Jev never emits a distance or a coordinate.
- Irreversible steps (descend, close, lift, place) stay sequenced through the primitive library with
  preconditions; the reactive channels sit on top and can interrupt any step.
- The demo that proves it: a hand enters between gripper and cup mid-task, the cup is moved while
  the arm approaches, and the picks change tick to tick with a visible, fast response, then the
  task resumes. Measures: fraction of ticks where a pick changed, intrusion-to-first-evade latency,
  task resumption after the intrusion clears.
- Standing orders remain the natural-language lever that changes the policy without code.
