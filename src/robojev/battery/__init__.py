"""The evaluation battery for the spelling demo (docs/showcase-demo.md, docs/battery.md): generated
episodes, judged on outcomes only, with two controls beside the fast decision loop.

    phrasings.py  what the person says, per event type, train / held_out
    generate.py   the grammar: seeded episodes (letters, positions, 0-3 events, the latest word asked)
    scenes.py     the table on the stub and on the physics sim
    stage.py      the perception layer (hidden cubes, hands), the director that plays the events, the user
    spell.py      SpellLLM, the offline planner stand-in
    decider.py    SpellJev, the offline decision stand-in (judges from ground truth)
    controls.py   AlwaysLLM and RulesOnly
    judge.py      the record: row score, times, calls, drops, contacts, STOP
    models.py     deciders and planners by name (mock, controls, live)
    run.py        the runner, --compare, the table
    cli.py        robojev-battery
    paraphrase.py the --paraphrase hook (a no-op offline)
"""
