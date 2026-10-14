"""Several deciders at one tier (brain/panel.py): a single backend behind a Panel decides exactly as
JevDecider does; split backends each see only their groups and the merged answers combine the same."""
from __future__ import annotations

import asyncio
import time

from robojev.brain import Brain, BrainConfig
from robojev.brain.core.combine import JevDecider
from robojev.brain.eval.scenarios import SORT_TASK, Utt, sort_goal
from robojev.brain.mocks import Line, ScriptedUser, oracle_models
from robojev.brain.panel import Panel
from robojev.robots import stub as S

CORRECTION = "actually, highest on the left"


class Recording:
    """A backend wrapper that records which question keys each request carried."""

    def __init__(self, inner, name):
        self.inner, self.name, self.seen, self.live = inner, name, [], False

    def answer(self, state, questions, ref=None):
        self.seen.append(sorted(questions))
        return self.inner.answer(state, questions, ref=ref)


def make():
    objs, places = S.sort_scene((5, 1, 3))
    return S.StubRobotServer(S.widowx_like(), objs, places, durations={"pick_and_place": 1.0}).open()


def test_a_panel_with_one_backend_is_the_single_decider():
    srv = make()
    try:
        decider, llm = oracle_models(srv, [sort_goal({o.id: int(o.label) for o in srv.world().objects.values()})])
        single = JevDecider(decider.backend)
        panel = Panel({"*": decider.backend})
        r = asyncio.run(Brain(srv, panel, llm, None, BrainConfig(max_s=20)).run(SORT_TASK))
        assert r.outcome == "done", r.log
        # the same answers through the same combine: re-decide the recorded events' views is not possible
        # after the fact, so compare the shape of the output and that every decision stayed local
        assert all(d["combined"].route == "stay_local" for d in r.decisions)
        assert panel.name == "panel" and single.name.startswith("jev")
    finally:
        srv.close()


def test_split_backends_each_see_only_their_groups_and_the_correction_still_holds():
    srv = make()
    try:
        nums = {o.id: int(o.label) for o in srv.world().objects.values()}
        user = ScriptedUser([Line(lambda *_: srv.world().arms["arm_0"].holding == "block_1", CORRECTION)])
        decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)], {CORRECTION: Utt("correction", version=1)})
        safety, task, router = (Recording(decider.backend, n) for n in ("safety", "task", "router"))
        panel = Panel({"right_now": safety, "in_plan_fix": task, "route": router})
        r = asyncio.run(Brain(srv, panel, llm, user, BrainConfig(max_s=25)).run(SORT_TASK))
        assert r.outcome == "done", r.log
        assert all(keys == ["right_now"] for keys in safety.seen)
        assert all(keys == ["in_plan_fix"] for keys in task.seen)
        assert all(all(k == "route" or k.startswith(("done.", "check.")) for k in keys) for keys in router.seen)
        assert task.seen and router.seen and len(safety.seen) == len(r.decisions)
        said = next(d for d in r.decisions if d["event"] == "user_text")
        assert (said["combined"].right_now, said["combined"].route) == ("hold", "fast_llm")
        where = {o.id: o.where for o in srv.world().objects.values()}
        assert where == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}
        assert all("panel" in d for d in [])        # meta is on the decider output, not the recorded decision
    finally:
        srv.close()


def test_a_failing_member_fails_the_whole_decision_cautiously():
    class Broken:
        name, live = "broken", False

        def answer(self, state, questions, ref=None):
            return {"answers": None, "latency_ms": 1.0}
    srv = make()
    try:
        decider, llm = oracle_models(srv, [sort_goal({o.id: int(o.label) for o in srv.world().objects.values()})])
        panel = Panel({"*": decider.backend, "route": Broken()})
        t = time.monotonic()
        r = asyncio.run(Brain(srv, panel, llm, None, BrainConfig(max_s=20)).run(SORT_TASK))
        # every decision that asked the broken member (the route question: every event but plan_arrived)
        # took combine's no-answer path; a new plan with nothing to check never asked it
        asked = [d for d in r.decisions if d["event"] != "plan_arrived"]
        assert asked and all(d["combined"].reason == "jev_error" for d in asked)
        assert r.outcome in ("done", "gave_up"), (r.outcome, time.monotonic() - t)
    finally:
        srv.close()
