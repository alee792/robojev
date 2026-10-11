"""H1 + M1: the brain as a client of the robot protocol, on a real clock with async events.

Imports nothing hardware-specific (nothing from robojev.robots): only robojev.protocol,
robojev.catalog and core/, e12v2's pure functions.

    loop.py      Brain: the asyncio loop (one queue; models in threads; heartbeat)
    tracker.py   the plan, the queue, each step's status: e12v2's bookkeeping without its tick loop
    rules.py     the code rules: STOP, models only pause, the confidence gate, the replan-loop limit
    arm.py       every protocol call that moves the arm
    changes.py   e12v2's change detector, with object changes settled before they are events
    connect.py   the manifest check: a non-conforming server is refused
    planner.py   plan / replan requests built from the manifest; answers validated by e12v2's code
    schema.py    the planner's step schema and the robot's description, generated from the manifest
    steps.py     plan step <-> skill call: e12v2's names and slots <-> the catalog's
    adapt.py     protocol world (m) -> e12v2 world (cm)
    text.py      literal event text
    messages.py  what goes on the queue
    mocks.py     offline stand-ins (e12v2's mock Jev and mock LLM on the stub, a scripted user);
                 never imported by the brain
    core/        e12v2's design logic, copied from experiments/e12v2/core: data, decision, combine,
                 plan, planner, changes, safety, interfaces. Imports nothing outside itself.
    eval/        e12v2's evaluation oracle and mock models (the stand-ins mocks.py and the tests
                 use), copied from experiments/e12v2/eval; the brain never imports it.
"""
from .connect import ManifestRefused
from .loop import Brain, BrainConfig, EpisodeResult, UserChannel
from .planner import ManifestPlanner
from .schema import describe_robot

__all__ = ["Brain", "BrainConfig", "EpisodeResult", "ManifestPlanner", "ManifestRefused", "UserChannel", "describe_robot"]
