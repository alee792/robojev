# robojev — goal (draft for alignment)

**Goal:** Do for a real robot arm what the TypeSafe Jev Doom demo does for a game. Observations
(RealSense RGB-D and arm telemetry) become a text **situation report** at a high rate. Jev answers a
battery of small **typed judgments** at about 100 ms per call. **Code composes** the answers into
safe, persistent arm commands. A **natural-language prompt** (a task, or a standing order that
changes behavior) steers every judgment without code changes.

**Non-goals (for now):** hardware and driver integration details; training a learned policy; putting
images into Jev (it is text-only).

**Success, first milestone (proposed):** the arm completes a simple tabletop task from an NL prompt,
driven by a closed Jev loop. A live standing order (e.g. "stay away from the blue block") visibly
changes its behavior mid-run, the way "Do not fire, simply dodge" did in the Doom demo.

## Decisions for Anthony (these shape the architecture)
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
