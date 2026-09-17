# Doom demo — what it actually shows

Evidence: `media/video/`, `media/keyframes/INDEX.md`, blog `sources/blog/`. Companion files:
`03-doom-situation-report.json` (state), `04-doom-questions.md` (battery + graph).

## The pattern in one paragraph
Code reads engine state → renders a **situation report** (a JSON object mixing static glossary and
per-tick facts, numbers paired with semantic bands, entities given letter-suffixed names) → asks Jev a
**battery of ~5–12 narrow typed Choices** (some with options generated per tick from the entities
present) → code **composes** the answers (goal + subject → objective → destination/bearing) and applies
**overrides** (dodge beats movement) → writes **persistent** controls (FACE, MOVE, TRIGGER, WEAPON) that
hold until the next decision ~100 ms later. A free-text **standing order** ("Do not fire, simply dodge")
is injected as context and changes every judgment without code changes. A **second composition**
(navigator) picks exploration targets described in prose, at its own cadence.

## Observations worth carrying forward

1. **Two-layer state: glossary + facts.** `measurement_context` teaches units, distance bands relative
   to body size (32u body; contact <64), bearing sign convention, tick rate, and that controls persist.
   Weapons carry `one_liner` intuitions. This is how they fight Jev's weakness at numbers
   (see `sources/typesafe-docs/jaggedness-jev-1.13.md`).
2. **Numbers always travel with words.** `"57 (contact)"`, `"-92° (right)"`, `"274 map units/second"`,
   `"last_seen": "a few seconds ago"`. In-view items carry only the band (`"range": "medium"`).
3. **Object permanence in code.** `remembered[]` keeps out-of-view entities with `last_seen` and
   `status: "out of view"`. Jev isn't asked to remember; code does.
4. **Entities are nameable options.** "imp F", "stimpack A", "armor vest B" — stable IDs allow
   `which enemy` Choices over what's present, and the chosen label is templated into later
   instructions.
5. **Dynamic option sets.** "offers goals · 6 goals" vs "5 goals": code only offers feasible goals.
6. **Question templating with prior answers.** DODGE/MOVEMENT text embeds "current top priority is
   …". Sequencing is an open question (see `04-doom-questions.md`).
7. **Binary gated decisions are 2-option Choices** (FIRING shows `conf`; Nouls don't return confidence).
8. **Code owns geometry.** Jev picks "walk to stimpack A"; code resolves to `(-768, 512)` and a path.
   Jev picks "look at cacodemon D"; code resolves bearing 310°. Jev never outputs a coordinate.
9. **Override layering.** Dodge (magenta) preempts MOVE. Trigger and weapon are independent channels.
10. **Displayed bars may be smoothed/lagged relative to `conf`** (t062: fire ≈45% and hold_fire ≈60%
    bars — not summing to 1 — with conf 0.87). Don't reverse-engineer the confidence formula from UI;
    test it (see `07-experiments-to-run.md`).
11. **Harness has an experiment frame.** Session config (`media/keyframes/session_dialog_t097.jpg`): brain `helm`, question-set version
    `research/v13_snowy_elephant`, flags `+ctx` ("facts repeated in the question") and `+guide`
    ("strategy guide closes the state" — the strategy text is appended at the *end* of state), judge
    `TS Research` (model selector — cf. `sources/system-one-adapter-README.md` for swapping in LLMs),
    navigator `pathfinder`, `EVAL: one solo session per slot, repeated`, opponent bots, TTL, skill. Tabs
    MAP / GRAPH / EVALS. `DIRECTOR` spawns enemies. HUD stats `DLT` (damage dealt?) `TKN` (taken?)
    `KILL` `DTH` `COST`, `TIER 0` (unknown).

## Numbers from the header strip [seen]
| Frame | last / avg latency | calls | tokens | tok/call | cost shown |
|---|---|---|---|---|---|
| t002 (combat) | 110 / 117 ms | 231 | 1.3M | ≈5.6k | $0.0475 |
| t062 (combat) | 107 ms / — | 698 | 4.2M | ≈6.0k | $0.1589 |
| t108 (exploring, no enemies) | 92 / 103 ms | 91 | 228.4k | ≈2.5k | $0.0083 |

State size scales with entity count. At $0.042/Mtok, 6k tok × 10 Hz ≈ **$9/hr** (blog: "~$7/hour").
Latency sparkline across the top of the UI; call rate appears ~10/s.

## What Doom gives them for free (and a robot won't)
- **Perfect perception.** Engine gives labels, positions, velocities, projectile trajectories. The blog
  concedes "not on images (yet…)", and a third-party writeup notes it "sees through walls".
- **Reversible, cheap failure.** Death = restart. Arm collisions aren't.
- **Discrete, instant actuation.** MOVE/FACE are setpoints the engine executes perfectly.
- **Fixed ontology.** Finite enemy/item kinds; a kitchen table isn't.
