# Doom → arm: axes, tensions, and what code must own

This is **not an architecture**. It lists what changes when the Doom pattern moves onto a WidowX AI arm
with RealSense cameras, the options on each axis, and a few fixed constraints from the evidence.
The architect decides.

## Fixed constraints (from evidence, not preference)
- **Jev takes text only.** Something upstream has to turn pixels and depth into labeled entities with
  semantic attributes. (`01-jev-facts.md`)
- **Jev is bad at numbers.** Coordinates, joint angles, and mm distances can't be the decision medium.
  Code buckets them, Jev picks, and code turns the pick back into setpoints. Doom does exactly this with
  `"57 (contact)"` and `"-92° (right)"`.
- **Jev never emits a joint or Cartesian target.** It picks among options that code has generated and
  validated, and code resolves the pick into motion (Doom: "walk to stimpack A" → `(-768, 512)`).
- **Confidence is a gate, not a safety system.** Calibration only holds across many predictions. Code
  owns limits, collision checks, force thresholds, and e-stop. Thresholds should rise with how
  irreversible an action is (docs `confidence.md`).
- **About 20 requests/s account ceiling**, about 100 ms per call, about 64k tokens of context. The limits
  can change without notice.
- **Actuation has to persist and degrade safely between decisions.** Doom's controls "persist
  continuously between decisions". A robot needs a defined behavior when a call is late, fails, or
  comes back with low confidence (hold, slow, retreat).

## Axes to decide

### A. Perception → entity list (the biggest open axis)
| Option | Notes |
|---|---|
| Fiducials / AprilTags on objects | Trivial labels and poses. Least general, fastest to a working loop. |
| Detector + depth (e.g. open-vocab 2D detector → D405/D455 depth → 3D centroid) | Gives Doom-like `label / distance / bearing`. Needs tracking to keep IDs ("cup A") stable. |
| VLM captioning per frame or on change | Rich semantics ("the red mug is tipped over") but slow. Could run as a separate lower-rate composition. |
| Sim ground truth (MuJoCo twin) | Doom-equivalent "perfect perception" for developing questions before perception exists. `widowx:twin` / `widowx:sim` skills exist in this environment. |

Stable entity naming, `remembered[]`-style object permanence, and occlusion status belong to this layer.

### B. What the NL prompt becomes
- **Task command** ("put the red block in the bowl") → goal + subject selection (Doom GOAL + `which …`).
- **Standing order / behavior modifier** ("go slow near the laptop", "never pass over the mug") → free
  text injected into every battery (Doom ORDERS). Doom shows both at once: a default strategy guide
  appended to the end of state plus a user order.
- Where the prompt goes (state vs. question `instructions`) affects prompt-injection risk. The docs
  warn that adversarial *state* can steer answers.

### C. Decision vocabulary (the "actuator channels")
Doom has four persistent channels: FACE, MOVE, TRIGGER, WEAPON. Candidate arm analogues:
- **Where to go:** pick a target entity or a target relation (above X, beside X, 5 cm back from X). Code
  resolves it to a pose via IK or a planner.
- **How to move:** `approach / hold / back_off / carry_on`, speed band, approach axis.
- **Gripper:** `open / close / hold`, which is a 2-option Choice like FIRING.
- **Reflex / override** (the dodge analogue): contact or unexpected-effort response, retreat, pause.
  Could be pure code, pure Jev, or a Jev judgment that code can veto.
- **Grasp/phase status:** "is the object in the gripper?", "did the place succeed?" These are
  verification questions, and the docs' strongest use case.

Choice of level: **primitive-per-tick** (Doom: 10 Hz micro-decisions) vs. **skill selection**
(jev-askable-arm: about 30 primitives at about 1 Hz, with a PD controller in between). The latency
budget and how smooth the arm needs to be decide this.

### D. Rates and layering
Doom runs one brain at about 10 Hz plus a navigator at a lower rate. The third-party jev-drone split
is flight control 500 Hz, safety reflex 50 Hz (can veto), perception 15 Hz, Jev 2.5 Hz, and skip
calls when the scene hasn't changed. Open questions:
- Does the brain run on a fixed tick or on events (scene change, phase change, contact)?
- Do goal and subject run at a lower rate than motion judgments? This also settles the "objective in
  the question text" sequencing puzzle in `04-doom-questions.md`.
- Budget: 20 rps split across compositions.

### E. State rendering
Borrow from Doom: static glossary (`measurement_context`: units, bands relative to gripper width,
frame and bearing conventions, control rate, what persists) + dynamic facts + `remembered[]` + the
recent action trace. Open questions: one state per battery or filtered per question group (docs:
context rot), and whether facts get repeated in questions (`+ctx`) or the strategy guide goes at the
end of state (`+guide`). Both were A/B flags in their harness.

### F. Compositions
Doom uses two: moment-to-moment and exploration. Arm candidates are a task planner / phase tracker
(what step of the task are we in), a motion brain, and a verifier / safety judge. How many there are
and how they share the rate budget is open.

### G. Harness and evals
Their UI is a research harness as much as a demo: versioned question sets
(`research/v13_snowy_elephant`), feature flags, swappable judge model, repeated solo eval sessions,
live GRAPH of the composition, cost and latency strip. Offline replay (record state → re-run question
sets) and `system-one-adapter-python` (LLM baseline behind the same API) are natural complements.
Sim-first vs. real bay is open.

## Things Doom didn't have to handle
- Continuous contact dynamics, grasp failure, objects moving because *you* touched them.
- Calibration between camera frame and arm base (the wrist D405 avoids this for approach).
- Irreversible or unsafe states (hitting a person, dropping or breaking things).
- An open-world object ontology.
