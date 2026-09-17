# robojev context bundle

Research handoff (2026-09-17) for an architect or implementer (Fable/Astra). **Nothing is designed or
built yet, and that's intentional.** You have creative license on the architecture. The constraints
that are actually binding are called out as such.

## Read in this order
| File | What it is |
|---|---|
| `00-goal.md` | Draft goal, first milestone, and the open decisions for Anthony |
| `01-jev-facts.md` | Jev API facts that bind a control loop (limits, types, weaknesses), with doc pointers |
| `02-doom-demo-analysis.md` | How the Doom demo works, what it gets for free, and the cost/latency numbers |
| `03-doom-situation-report.json` | Their state JSON, rebuilt from video frames |
| `04-doom-questions.md` | Their question battery, the composition graph, and open sequencing questions |
| `05-robotics-translation-notes.md` | Doom vs. the arm: fixed constraints plus the design axes still open |
| `06-prior-art.md` | Third-party Jev arm/drone/Doom projects and language-to-robot literature |
| `07-experiments-to-run.md` | Questions only real API calls can answer |

## Raw material
| Path | Contents |
|---|---|
| `sources/jev-widowx-sources.md` | Anthony's original source list (Jev, Trossen, LeRobot, cameras, VLA background) |
| `sources/typesafe-docs/` | 38 TypeSafe doc pages as raw markdown, plus `llms.txt` (index) and `llms-full.txt` (the full ~800 KB corpus) |
| `sources/typesafe-skill/SKILL.md` | TypeSafe's official agent skill |
| `sources/system-one-adapter-README.md` | Same API backed by LLMs (for baselines and offline dev) |
| `sources/blog/` | Launch post: original HTML and extracted text. The FAQ answers are rendered by JavaScript and aren't in the static HTML, so only the questions were captured |
| `media/video/jev-doom-demo.mp4` | The Doom demo (2:42), from the blog's embedded asset |
| `media/video/transcript.md` | Narration and visual breakdown |
| `media/keyframes/` | Extracted frames, listed in `INDEX.md`. These replace the pasted screenshots, which only existed as chat attachments |

## Evidence tags used
**[verified]** = read in the primary docs · **[seen]** = legible in a video frame · **[inferred]** =
reasoning from evidence · **[read]/[snippet]/[known]** in `06-prior-art.md` = what the research
sub-agent actually opened, with details summarized by a small model, so re-check them.

## Out of scope this session, but available
- Hardware/driver research was intentionally skipped. `sources/jev-widowx-sources.md` §2–4 has the
  Trossen and RealSense pointers.
- The environment that created this bundle has WidowX tooling: skills `widowx:readiness`,
  `widowx:move`, `widowx:twin`, `widowx:sim`, `widowx:trajectory`, `widowx:teleop`, `widowx:policy`,
  plus `mcp__widowx__*` tools (movej/movel/gripper/trajectory). Read `widowx:readiness` and
  `widowx:move` before designing actuation. The MuJoCo twin is a likely place to develop against before
  touching the bay.
- No TypeSafe API key was used. All numbers are from the docs or the video.
