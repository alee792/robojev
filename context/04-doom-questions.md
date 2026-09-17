# The Doom question battery (reconstructed)

Tags: **[seen]** legible in a frame · **[truncated]** option label cut off in UI · **[inferred]** reasoning, not visible.
Frames: `media/keyframes/panel_t009.jpg`, `panel_t108.jpg`, `t029.jpg`, `t055.jpg`, `t062.jpg`, user screenshots.

## Per-tick "brain" battery (UI panel `JUDGMENTS`)

| ID | Instruction text [seen] | Options [seen / truncated] | Type |
|---|---|---|---|
| `FIRING` | "Should the player's trigger be held down right now?" | `fire`, `hold_fire` | **2-option Choice** — shows `conf`, and Nouls carry no confidence [inferred] |
| `GOAL` | "Considering \`player\`, \`enemies\`, and \`items\`, what is the player's highest-priority goal right now?" | `upgrade_w…` (upgrade_weapon), `scout`, `kill_enem…` (kill_enemies), `stock_ammo`, `add_armor`, `restore_h…` (restore_health) — option order varies between frames | Choice. Graph node says "offers goals → 6 goals" / "5 goals": **the option set is built per tick** from what's available [seen+inferred] |
| `DODGE` | "The player's current top priority is {objective}. What does this exact moment call for?" | `carry_on`, `dodge_left`, `dodge_rig…`, `dodge_bac…` ×2 (back_left/back_right) | Choice |
| `MOVEMENT` | "The player's current top priority is {objective}. Given the current situation, how should the player move right now?" | `hold_grou…`, `close_in:…`, `hold_rang…`, `back_off:…`, `walk to …` (several) | Choice; option labels embed a target ("walk to stimpack A") → **options are parameterized by entities** [seen/inferred] |
| `TURN` | "Where should the player point right now? Looking and aiming are one act; the gun goes where the eyes go." | `hold the …`, `look ahea…`, `look left`, … | Choice |

`{objective}` examples [seen]: "attacking enemies, focusing on cacodemon A", "trying to restore health with medikit C", "advancing through the level".

## Subject-selection questions (visible only in GRAPH tab)

Graph nodes (orange = Jev answer) [seen t055, t062, 1:33 screenshot]:
- `which enemy → imp F` (fed by `known enemies · 4`)
- `which health pickup → stimpack A`
- `which ammo → ammo clip C`
- `which weapon → super shotgun C`
- `which armor → armor vest B`
- `which key` (dimmed — no keys known)
- `which door switch` (dimmed), `map explore target` (dimmed; lit in level mode)
- `weapon → keep | pistol` (output WEAPON)

These are Choices whose options are the labeled entities from state ("imp F", "stimpack A"). Letter suffixes exist so entities are nameable options. [inferred]

## Composition (GRAPH legend) [seen, semantics inferred]

```
dashed box  = state input        (known enemies · 4, known health pickups · 3, known ammo · 10, …)
dashed box  = code-built option set (offers goals · 6 goals)
orange box  = Jev answer         (goal ▸ restore_health, which enemy ▸ imp F, facing ▸ cacodemon D, movement ▸ walk to medikit G, dodge ▸ dodge_right, trigger ▸ hold_fire, weapon ▸ keep)
oval        = code transform     (goal + subject → objective ▸ restore_health: medikit G
                                  pick/turn → bearing ▸ → 310°
                                  answer → destination ▸ → (-256, -768)
                                  open-door carry-out, "path leads to shut door")
green box   = actuator command   FACE ▸ 310° | keep heading
                                 MOVE ▸ (-1007, 375)
                                 TRIGGER ▸ fire | hold_fire
                                 WEAPON ▸ keep | pistol
magenta dashed = dodge override into MOVE
```

Note the MOVE target differs from the "destination" oval when dodge fires (t062: destination (-256,-768) but MOVE (-1007,375)) → dodge **overrides** movement in code. [inferred]

## Ordering puzzle (OPEN)
DODGE/MOVEMENT text contains the objective, which depends on GOAL + `which …`. Jev answers questions in one request independently. So either:
1. two requests per tick (stage 1: goal + which-X; stage 2: dodge/move/turn/fire), or
2. one request per tick using the **previous** tick's objective, or
3. goal re-evaluated at a lower rate.
Header shows ~100–120 ms "last/avg" per call, and "10 decisions/sec". Can't disambiguate from video. Relevant design choice for the arm.

## Level-mode variant (t108)
GOAL question removed; objective fixed to "advancing through the level" from the navigator. MOVEMENT → `walk to t…` (walk to target). Navigator ("pathfinder") candidates are prose frontier descriptions labeled A/B/D (`media/keyframes/nav_sidebar_t108.jpg`):
- "Room 3 lies to the north — seen from afar but never yet entered; about 12 paces away"
- "The unmapped edge on the southwest side of Room 1, where about eight paces of floor continue into unscanned territory; about three paces away"
