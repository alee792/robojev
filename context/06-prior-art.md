# Prior art

Collected by a research sub-agent on 2026-09-17. Tags: **[read]** means the agent fetched the page, but
through a summarizing fetch tool, so treat the details as summaries and re-verify before relying on
them. **[snippet]** means it was only seen in search results. **[known]** means it comes from model
knowledge and was not fetched.

## Direct Jev analogues (most relevant, read these first)
- **TarunTomar122/jev-askable-arm** [read]: https://github.com/TarunTomar122/jev-askable-arm
  - Franka Panda arm in the ManiSkill simulator (PickCube and button tasks). It does not use a camera:
    the state comes straight from the simulator (xyz positions, gripper state, distances, recent
    actions).
  - Each call, Jev picks one of about 30 closed primitives (`hover_over`, `descend_onto`,
    `close_gripper`, …) from an English goal. A PD controller then executes it.
  - Runs at about 1 s per pick.
  - This is the closest existing thing to robojev. Its perception is simulator ground truth.
- **RomanSlack/jev-drone** [read]: https://github.com/RomanSlack/jev-drone
  - A MuJoCo quadrotor. Classical computer vision turns the camera into a JSON scene (range sectors,
    obstacle height, target).
  - Three questions per call: `maneuver` (Choice), `risk` (Score), `target_truly_lost` (Noul).
  - Layered loops: control at 500 Hz, a safety reflex at 50 Hz that can veto, perception at 15 Hz,
    Jev at about 2.5 Hz.
  - Median call time is 0.11 s. It skips calls when nothing has changed, using about 110 calls in a
    65 s flight.
  - Good reference for the layering.
- **lukaske/jev-doom-agent** [read]: https://github.com/lukaske/jev-doom-agent
  - Chocolate Doom running as WebAssembly, with a C bridge that exposes the game state.
  - One Jev Choice picks a tactical macro, and a local motor controller carries it out.
  - When confidence is low it falls back to an offline policy.
- ViZDoom reimplementations with a distilled open model: mikesmullin/openjev [read, only an index],
  theoriclabs/letsusecompute#21 [read], kw2828/OpenJev, daseinlabs/open-jev, vinnylarouge/jevlike
  [snippet].
- awesome-jev list [read]: https://github.com/AnotiaWang/awesome-jev (Discord, playground, more game
  demos).

## Commentary on the launch
- The Register [read]: https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
  (a demo call took 0.114 s vs. 8.566 s for a GPT model).
- OrcaRouter explainer [read]: https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know
- HN/GeekNews critiques [snippet]: https://news.hada.io/topic?id=19453. Points raised: the weights are
  closed, the comparisons with LLMs aren't like-for-like, and the Doom bot "sees through walls" because
  it gets engine state.
- evals.typesafe.ai [read]: four business workflows only, nothing on control tasks.

## Language → robot, typed/discrete decisions over structured scenes [known]
1. SayCan: the language model scores each skill's usefulness, multiplied by a learned feasibility
   score. This is closest to typed Choice over skills. https://say-can.github.io
2. KnowNo / "Robots that ask for help": calibrated multiple-choice planning that asks for help when
   uncertain. This is the conceptual twin of confidence gating. https://robot-help.github.io
3. Code as Policies: the LLM writes code that calls perception and control APIs.
   https://code-as-policies.github.io
4. Inner Monologue: closed-loop replanning from text feedback. https://innermonologue.github.io
5. VoxPoser: the LLM composes 3D value maps, and a planner tracks them at a high rate.
   https://voxposer.github.io
6. ProgPrompt: program-style prompts over the available actions and objects.
   https://progprompt.github.io
7. Text2Motion: skill chaining with geometric feasibility checks.
   https://sites.google.com/stanford.edu/text2motion
8. Language to Rewards: the LLM sets reward parameters for a high-rate model-predictive controller.
   https://language-to-reward.github.io
9. LLM-BRAIn / behavior trees from LLMs: https://arxiv.org/abs/2305.19352
10. RT-2: a VLA with discretized action tokens (256 bins per dimension).
    https://robotics-transformer2.github.io

Background on action chunking, temporal ensembling and adaptive horizons is in
`sources/jev-widowx-sources.md` §5.

**Common thread:** code runs the fast loops. The model advises at a slower rate over a fixed menu of
options, and a safety layer can veto what it says.
