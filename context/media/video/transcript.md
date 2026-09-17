# Jev Doom demo — transcript & visual breakdown

Source video: `jev-doom-demo.mp4` (2:42, 1760x1080@60). Downloaded from the blog post's embedded
asset `https://framerusercontent.com/assets/rlL7ImEbISFoYt3IJEHHfvjpY.mp4`
(blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Transcript supplied by Anthony (user) at session start; corrections from frame inspection are marked **[frame-check]**.

## 0:00–0:28 | Intro & execution speed
Audio: "Jev is so fast, it can play Doom. Every decision—where to move, where to aim, whether to hold
down the trigger—Jev is making those decisions inside the software loop. Responding at about 100
milliseconds per battery of questions, it's able to make 10 decisions a second, which is fast enough
to track enemies and fire effectively, dodge projectiles."

Visual: ORDERS box (`> standing order…`), DIRECTOR (spawned/killed), JUDGMENTS panel:
- `FIRING` "Should the player's trigger be held down right now?"
- `GOAL` "Considering `player`, `enemies`, and `items`, what is the player's highest-priority goal right now?"
- `DODGE` "The player's current top priority is attacking enemies, focusing on imp B. What does this exact moment call for?"
- `MOVEMENT` "The player's current top priority is attacking enemies, focusing on cacodemon A. Given the current situation, how should the player move right now?"
- **[frame-check]** A fifth question `TURN` exists: "Where should the player point right now? Looking and aiming are one act; the gun goes where the eyes go." (panel_t108.jpg)

## 0:29–0:41 | The JSON Situation Report
Audio: "Our System 1 model is fed a structured description of the situation. Health, nearby enemies,
incoming projectiles, available pickups—everything a player needs to make informed decisions about
what to do next in the game."

Visual: collapsible `SITUATION REPORT` JSON. Reconstructed in `../../03-doom-situation-report.json`.

## 0:42–1:05 | Prompting & the state machine
Audio: "With that game state as input, the model is asked to make small, typed judgments, such as:
should the player's trigger be held down right now? What is the player's highest priority goal right
now? Should the player be dodging? And others. We compose their answers into the player's next
action and keep repeating that loop as the game changes. This is what we call a composition of AI
primitives."

Visual: GRAPH tab — dataflow from state inputs → Jev questions → code transforms → actuator outputs
(FACE / MOVE / TRIGGER / WEAPON). See `../../02-doom-demo-analysis.md`.

## 1:06–1:32 | Modifying behavior via text
Audio: "The model has a default game strategy generally telling it how to play the game somewhat
effectively. But of course, because that's fed in as text, we can change that at any time. Let's try
here: 'Do not fire, simply dodge.' We can see that immediately the player's behavior changes, because
the model is making the same judgments but now in the context of that new instruction. Not the best
strategy, but hey, that's on us."

Visual: ORDERS input: `> Do not fire, simply dodge.` Player strafes without firing.

## 1:33–2:42 | Two compositions cooperating (exploration)
Audio: "Now, this composition handles the moment-to-moment play. Add a second composition that
decides where to explore next, and they can work together to make it through a level."

Visual: NEW SESSION dialog (name `arena`, map mode `level`, IWAD The Ultimate Doom, E1M1, skill 3,
slot brain=`helm`, judge `TS Research`, context ✔ "facts repeated in the question", guide ✔ "strategy
guide closes the state", navigator `pathfinder`). Then MAP tab: auto-built top-down map, path trace,
frontier list on the right written in prose ("Room 3 lies to the north — seen from afar but never yet
entered; about 12 paces away"). Status: `GOAL ADVANCING THROUGH THE LEVEL`. GOAL question disappears
from the battery in this mode.
