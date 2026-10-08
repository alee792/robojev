# Showcase demo: spell it out

The hero demo for the OpenAI showcase, on the Panda. One run, about 90 seconds, built from the v2 skills
with nothing scripted inside the robot. Setup and spikes: `docs/panda-setup.md`.

## What it shows

People change their minds and get in the way, and the arm reacts while they're still talking, then
finishes the job. The fast loop (Decisions API, ~150 ms) decides what to do right now; the slow loop
(GPT planner, a few seconds) works out the new plan. Both are visible on screen.

## Props

- A tray with six marked slots, facing the audience.
- Wooden letter cubes, ~5 cm, letters on every face, high contrast: **O P E N A A I D**, plus spares.
  OPENAI and PANDA together need two A's and a D; the O and E are left over for the finale.
- The camera above the table; a screen next to the arm.

## The run

| Beat | What happens | What it shows |
|---|---|---|
| 1. Task | "Spell OPENAI." The arm starts on the O within ~3 s. | The planner, and that step 1 starts while the plan is checked |
| 2. Correction | After O and P are placed, someone says "actually, spell PANDA." The arm holds mid-carry within a beat; a new plan arrives in a few seconds; P moves to slot 1, O comes out, and A N D A follow. | The headline: right-now decision in ~150 ms, replan in seconds, no pre-written variants |
| 3. Theft | A volunteer takes the N off the table. The arm re-targets to the spare or, if there isn't one, asks for it back on screen. | Scene changes go through the same decision; asking the user is a route |
| 4. Hand in | Someone reaches toward the gripper. The arm pauses; it resumes when the hand leaves. | The right-now reflex; the code stop distance stays underneath |
| 5. Finale | "Hand me the O." The arm picks the leftover O and holds it out; it releases when a hand is open beneath it. | Handover, and a decision on a camera frame |

The volunteer lines are suggestions. Any phrasing works; that's the point. Rehearse with improvised
ones too.

## On screen

- Left: the tray as the camera sees it, with the current plan step.
- Right: a live strip of events and decisions, e.g. `user text → hold, fast LLM · 142 ms` and
  `new plan · 2.8 s`. The two timescales side by side are the story.
- The current word as the planner restated it ("PANDA").

## Input

Corrections are spoken (speech to text) or typed on a tablet by a volunteer. Typed is the fallback.

## Ready when

- 10 full runs in a row with no operator help, including improvised corrections.
- Correction to a visible hold: under 0.5 s.
- No hand contacts; every pause within the code stop distance.
- Every pick lands: 19 of 20 or better on the cubes (spike S7).

## Fallbacks

- **No internet or a slow planner:** a "rehearsed" mode that replays cached plans for the scripted
  lines, clearly labelled as such to the operator. Corrections still go through decisions.
- **Mic fails:** the tablet.
- **Arm faults:** stop button, recover in software (spike S11), restart from the current tray.

## Depends on

Spikes S7 (grasping the cubes), S8 (reading letters), S10 (seeing a hand), S12 (Decisions API with
images) and S14 (speech to text) in `docs/panda-setup.md`.
