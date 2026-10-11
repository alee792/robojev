"""H2 (docs/widowx-pair.md "Crawl: one brain, two arms"): one brain and one planner drive the stub's
two-arm pair (robojev.robots.stub.widowx_pair, pair_scene), blocks in arm_0's half, the tray in
arm_1's, a shared zone between them whose lock the server owns. e12v2's mock decider (one oracle per
arm) and the hand-over mock planner (mocks.PairLLM); the stub only, on a real clock.

  (a) a 3-block sort across the table completes, both arms used, no step on an arm that can't reach;
  (b) both arms have a skill in flight at once;
  (c) a correction mid-run replans and both arms end consistent;
  (d) a hand near arm_1 pauses arm_1 only, arm_0 keeps working;
  (e) STOP parks both;
  (f) one arm: the planner's requests are byte-identical to before H2 (snapshot); test_brain.py
      runs unchanged;
  (g) M1: the same brain on the push-only one-arm manifest.

pytest-asyncio is not installed: each episode runs under asyncio.run in a plain test.
"""
import asyncio
import hashlib
import json
import threading
import time

import pytest

from robojev.brain import Brain, BrainConfig, describe_robot
from robojev.brain.adapt import core_state, geometry
from robojev.brain.arms import holding_errors, reach_errors, waits_for
from robojev.brain.eval.oracle import Goal
from robojev.brain.eval.scenarios import SORT_TASK, Utt, sort_goal
from robojev.brain.mocks import PRESS_STOP, Line, ScriptedLLM, ScriptedUser, give_arms, oracle_decider, pair_models
from robojev.brain.planner import ManifestPlanner
from robojev.robots import stub as S

CORRECTION = "actually, highest on the left"
ARMS = ("arm_0", "arm_1")


# ---------------------------------------------------------------- helpers


@pytest.fixture
def pair():
    """Make two-arm stub servers; every one opened here is closed after the test."""
    made = []

    def make(numbers=(5, 1, 3), pick_s=0.6, **kw) -> S.StubRobotServer:
        objs, places = S.pair_scene(numbers)
        srv = S.StubRobotServer(S.widowx_pair(), objs, places, durations={"pick_and_place": pick_s}, homes=S.PAIR_HOMES, **kw)
        made.append(srv)
        return srv.open()
    yield make
    for s in made:
        s.close()


def numbers(srv) -> dict:
    return {o.id: int(o.label) for o in srv.world().objects.values()}


def where(srv) -> dict:
    return {o.id: o.where for o in srv.world().objects.values()}


def starts(srv, arm=None) -> list[dict]:
    return [c[2][2] for c in srv.calls if c[1] == "start" and (arm is None or c[2][0] == arm)]


def brain(srv, user=None, desc_version=False, **cfg) -> Brain:
    nums = numbers(srv)
    versions = [sort_goal(nums)] + ([sort_goal(nums, desc=True)] if desc_version else [])
    decider, llm = pair_models(srv, versions, {CORRECTION: Utt("correction", version=1)})
    return Brain(srv, decider, llm, user, BrainConfig(max_s=25, **cfg))


def episode(b: Brain, task=SORT_TASK):
    return asyncio.run(b.run(task))


def in_tray(srv, desc=False) -> bool:
    w, nums = where(srv), numbers(srv)
    return all(w[oid] == f"tray_slot_{k}" for k, oid in enumerate(sorted(w, key=lambda o: nums[o], reverse=desc), 1))


class Watch:
    """Polls the server every 5 ms through the protocol (status() of each skill started, world()):
    which object each arm's skill in flight is on, and each arm's mode, at the server's time (the
    clock of its `history` and `script_log`)."""

    def __init__(self, srv):
        self.srv, self.samples = srv, []
        self._halt = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._t.join(timeout=1.0)

    def _poll(self):
        while not self._halt.wait(0.005):
            active, i = {}, 1
            while True:
                try:
                    st = self.srv.status(f"sk{i}")
                except KeyError:
                    break
                if st.state in ("running", "holding", "paused"):
                    active[st.arm] = st.args.get("object")
                i += 1
            modes = {a: x.mode for a, x in self.srv.world().arms.items()}
            self.samples.append((self.srv.now(), active, modes))

    def both_in_flight(self) -> bool:
        return any(len(active) == 2 for _, active, _ in self.samples)

    def double_claims(self) -> list:
        """Moments two arms' skills in flight were on the same object."""
        return [(t, a) for t, a, _ in self.samples if len(a) == 2 and len(set(a.values())) == 1]


def assert_one_decision_per_arm_event(r):
    """Every per-arm copy of an event got exactly one decision, on its own arm."""
    made = {e["id"]: e["arm"] for e in r.log if e["note"] == "event"}
    decided = [(d["event_id"], d["arm"]) for d in r.decisions]
    trusted = sum(1 for e in r.log if e["note"] == "plan_trusted")
    assert len(decided) == len({i for i, _ in decided})
    assert all(made[i] == arm for i, arm in decided)
    assert len(made) - len(decided) == trusted


# ---------------------------------------------------------------- (a) the sort across the table


def test_a_three_block_sort_across_the_table_uses_both_arms_each_within_reach(pair):
    srv = pair()
    r = episode(brain(srv))
    assert r.outcome == "done", r.log
    assert in_tray(srv)
    assert r.steps_done == 6 and r.steps_failed == 0 and r.failed_plans == 0
    # the hand-over pattern: arm_0 takes each block from its half to a zone slot, arm_1 from there to the tray
    assert [s["place"].rsplit("_", 1)[0] for s in starts(srv, "arm_0")] == ["zone_slot"] * 3
    assert [s["place"].rsplit("_", 1)[0] for s in starts(srv, "arm_1")] == ["tray_slot"] * 3
    # no step went to an arm that can't reach it: the plans pass the reach check, and the server
    # (which refuses an unreachable object or place) never refused a start
    applied = next(e["plan"] for e in r.log if e["note"] == "plan_applied")
    assert all(s["arm"] in ARMS for s in applied["steps"])
    assert not [e for e in r.log if e["note"] == "event" and e["event"] == "step_failed"]
    assert all(srv.status(f"sk{i}").state == "done" for i in range(1, 7))
    assert_one_decision_per_arm_event(r)
    assert not [c for c in srv.calls if c[1] == "stop"]


def test_reach_check_refuses_a_step_on_an_arm_that_cannot_reach():
    objs, places = S.pair_scene()
    srv = S.StubRobotServer(S.widowx_pair(), objs, places, homes=S.PAIR_HOMES)
    w, specs = srv.world(), {a.id: a for a in S.widowx_pair().arms}
    step = {"skill": "move_object", "object": "block_1", "target": "zone_slot_1", "direction": "none"}
    ok = [{**step, "id": "s1", "arm": "arm_0"}, {**step, "id": "s2", "arm": "arm_1", "target": "tray_slot_1"}]
    assert reach_errors(ok, w, specs) == []
    bad = [{**step, "id": "s1", "arm": "arm_1"}, {**step, "id": "s2", "arm": "arm_0", "target": "tray_slot_1"}]
    errs = reach_errors(bad, w, specs)
    assert len(errs) == 2
    assert "s1: arm_1 cannot reach block_1 where it will be then (x 0.18 m, y 0.00 m); arms that can: arm_0" in errs[0]
    assert "s2: arm_0 cannot reach tray_slot_1; arms that can: arm_1" in errs[1]   # block 1 is in the zone by then
    assert holding_errors(ok, {"arm_1": "block_3"}) == ["arm_1 is holding block_3: its first step must put it somewhere "
                                                        "(pick_and_place, stack_on or hand_over of block_3)"]


def test_planner_retries_a_plan_with_an_armless_or_unreachable_step(pair):
    srv = pair()
    good = {"reading": "r", "done_when": [{"id": "d1", "text": "block_1 is in tray_slot_1", "object": "block_1",
                                           "relation": "in", "target": "tray_slot_1"}], "constraints": []}
    step = {"skill": "pick_and_place", "object": "block_1", "direction": "none", "distance": None, "seconds": None}
    armless = {**good, "steps": [{**step, "id": "s1", "arm": None, "target": "tray_slot_1"}]}
    unreachable = {**good, "steps": [{**step, "id": "s1", "arm": "arm_0", "target": "tray_slot_1"}]}
    llm = ScriptedLLM([armless, unreachable], latency_ms=0)
    planner = ManifestPlanner({"fast_llm": llm}, srv.manifest(), ARMS)
    w = srv.world()
    res = planner.plan(SORT_TASK, [], w, core_state(w, "arm_0", geometry(srv.manifest().arms[0], 0.2), 0.2))
    assert not res.ok
    assert "arm must be one of arm_0, arm_1, not None" in res.attempts[0]["errors"][0]
    assert any("arm_0 cannot reach tray_slot_1" in e for e in res.attempts[1]["errors"])


def test_describe_robot_says_what_each_arm_reaches(pair):
    srv = pair()
    text = describe_robot(srv.manifest(), world=srv.world())
    assert "The robot has 2 arms (arm_0 and arm_1); you plan for all of them" in text
    assert "arm_0 reaches block_5, block_1, block_3 and shared_zone; arm_1 reaches tray and shared_zone" in text
    assert "Only one arm may be inside shared_zone at a time" in text
    assert len(text.splitlines()) <= 10
    # one arm of the pair, chosen: the one-arm description, nothing about the others
    assert describe_robot(srv.manifest(), "arm_1").startswith("The robot has 2 arms; you plan for arm_1.")


def test_mock_planner_makes_the_hand_over_pattern(pair):
    srv = pair()
    one_arm = [{"id": f"s{i}", "skill": "move_object", "object": b, "target": f"tray_slot_{i}", "direction": "none"}
               for i, b in enumerate(["block_1", "block_3", "block_5"], 1)]
    steps = give_arms(one_arm, srv.world(), srv.manifest())
    assert [(s["arm"], s["object"], s["target"]) for s in steps] == [
        ("arm_0", "block_1", "zone_slot_1"), ("arm_1", "block_1", "tray_slot_1"),
        ("arm_0", "block_3", "zone_slot_2"), ("arm_1", "block_3", "tray_slot_2"),
        ("arm_0", "block_5", "zone_slot_1"), ("arm_1", "block_5", "tray_slot_3")]
    assert reach_errors(steps, srv.world(), {a.id: a for a in srv.manifest().arms}) == []


# ---------------------------------------------------------------- (b) side by side


def test_b_both_arms_have_a_skill_in_flight_at_once_and_never_on_the_same_object(pair):
    srv = pair()
    with Watch(srv) as watch:
        r = episode(brain(srv))
    assert r.outcome == "done", r.log
    assert watch.both_in_flight()
    assert watch.double_claims() == []
    # arm_1's first step waited for arm_0's (same block) and said why
    waits = [e for e in r.log if e["note"] == "step_waits" and e["arm"] == "arm_1"]
    assert waits and "block 1" in waits[0]["why"]


def test_waits_for_another_arm_on_a_shared_object_or_slot(pair):
    srv = pair()
    w = srv.world()
    mk = lambda sid, arm, obj, tgt: {"id": sid, "arm": arm, "skill": "move_object", "object": obj, "target": tgt, "direction": "none"}  # noqa: E731
    steps = {s["id"]: s for s in [mk("s1", "arm_0", "block_1", "zone_slot_1"), mk("s2", "arm_1", "block_1", "tray_slot_1"),
                                  mk("s3", "arm_0", "block_3", "zone_slot_2"), mk("s4", "arm_0", "block_5", "zone_slot_1")]}
    q = list(steps)
    pending = {sid: "pending" for sid in q}
    assert waits_for("s1", q, steps, pending, w, "arm_0") is None
    assert "arm_0's step move_object(block_1 → zone_slot_1) (earlier in the plan) also uses block_1" == \
        waits_for("s2", q, steps, pending, w, "arm_0")
    # s1 done (block 1 now in zone slot 1): s2 may run; s4 (same arm_0 order aside) waits on s2 for zone slot 1
    srv.person_moves(0.0, "block_1", to="zone_slot_1")
    while where(srv)["block_1"] != "zone_slot_1":
        time.sleep(0.01)
    w = srv.world()
    status = {"s1": "done", "s2": "running", "s3": "pending", "s4": "pending"}
    q = ["s2", "s3", "s4"]
    assert waits_for("s3", q, steps, status, w, "arm_0") is None
    assert "(running) also uses zone_slot_1" in waits_for("s4", q, steps, status, w, "arm_0")


# ---------------------------------------------------------------- (c) a correction mid-run


def test_c_correction_mid_run_replans_and_both_arms_end_consistent(pair):
    srv = pair(pick_s=1.2)
    both_holding = lambda: all(a.holding for a in srv.world().arms.values())    # noqa: E731
    user = ScriptedUser([Line(both_holding, CORRECTION)])
    with Watch(srv) as watch:
        r = episode(brain(srv, user, desc_version=True))
    assert r.outcome == "done", r.log
    assert in_tray(srv, desc=True)                          # highest on the left now
    assert all(a.holding is None for a in srv.world().arms.values())
    assert watch.double_claims() == []
    said = [d for d in r.decisions if d["event"] == "user_text"]
    assert sorted(d["arm"] for d in said) == list(ARMS)     # one decision per arm
    assert all((d["combined"].right_now, d["combined"].route) == ("hold", "fast_llm") for d in said)
    replans = [x for x in r.llm if x["kind"] == "replan"]
    assert len(replans) == 1 and replans[0]["ok"]           # one replan for the event, not one per arm
    assert r.steps_failed == 0 and r.stale_plans == 0
    assert_one_decision_per_arm_event(r)


# ---------------------------------------------------------------- (d) a hand near arm_1


def test_d_hand_near_arm_1_pauses_arm_1_only_and_arm_0_keeps_working(pair):
    srv = pair(pick_s=1.2)
    srv.person_hand(lambda w: w.arms["arm_1"].holding is not None, for_s=1.0, arm="arm_1", xy=(0.66, 0.0))
    with Watch(srv) as watch:
        r = episode(brain(srv))
    assert r.outcome == "done", r.log
    assert in_tray(srv)
    paused = [(t, arm) for t, arm, mode in srv.history if mode == "paused"]
    assert [arm for _, arm in paused] == ["arm_1"]
    assert all(c[2] == ("arm_1",) for c in srv.calls if c[1] in ("pause", "resume"))
    # the hand was arm_1's decision only; while arm_1 was paused, arm_0 ran
    hand = [d for d in r.decisions if d["event"] == "scene_change"]
    assert hand and {d["arm"] for d in hand} == {"arm_1"}
    resumed = next(t for t, arm, mode in srv.history if arm == "arm_1" and t > paused[0][0] and mode == "running")
    assert any(m["arm_0"] == "running" and m["arm_1"] == "paused" for t, _, m in watch.samples if paused[0][0] < t < resumed)
    appeared = next(t for t, what in srv.script_log if what == "person's hand appears")
    left = next(t for t, what in srv.script_log if what == "person's hand leaves")
    assert appeared <= paused[0][0] < left <= resumed
    assert_one_decision_per_arm_event(r)


def test_the_other_arm_is_one_more_thing_nearby_in_an_arms_view(pair):
    srv = pair()
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner = inner

        def decide(self, ev, view):
            seen.append((ev, view))
            return self.inner.decide(ev, view)
    b = brain(srv)
    b.decider = Recording(b.decider)
    assert episode(b).outcome == "done"
    ev, view = next((e, v) for e, v in seen if e.kind == "step_done" and e.data["arm"] == "arm_1")
    assert "arm:arm_0" in view.state.objects and view.names["arm:arm_0"].startswith("arm_0, the other arm (")
    steps = {s["id"]: s for s in view.plan["steps"]}
    assert view.queue and all(steps[q]["arm"] == "arm_1" for q in view.queue)      # its own share of the plan
    ev, view = next((e, v) for e, v in seen if e.kind == "plan_arrived" and e.data["arm"] == "arm_1")
    assert "arm:arm_0" not in view.state.objects and len(view.queue) == 6          # the plan check sees it all


# ---------------------------------------------------------------- (e) STOP


@pytest.mark.parametrize("line", ["stop", PRESS_STOP])
def test_e_stop_parks_both_arms(pair, line):
    srv = pair(pick_s=1.2)
    user = ScriptedUser([Line(lambda: all(a.holding for a in srv.world().arms.values()), line)])
    r = episode(brain(srv, user))
    assert r.outcome == "stopped"
    stops = [c for c in srv.calls if c[1] == "stop"]
    assert len(stops) == 1
    assert not [c for c in srv.calls if c[0] > stops[0][0] and c[1] in ("start", "resume", "hold", "pause")]
    w = srv.world()
    for arm, home in S.PAIR_HOMES.items():
        a = w.arms[arm]
        assert a.mode == "stopped" and a.holding is None and (a.x, a.y) == home
    assert r.events["user_text"] == 0


# ---------------------------------------------------------------- the stub's shared-zone lock


def test_stub_shared_zone_lock_makes_the_other_arm_wait(pair):
    srv = pair(pick_s=1.0)
    srv.person_moves(0.0, "block_1", to="zone_slot_2")       # something for arm_1 to take from the zone
    while where(srv)["block_1"] != "zone_slot_2":
        time.sleep(0.01)
    a0 = srv.start("arm_0", "pick_and_place", {"object": "block_3", "place": "zone_slot_1"})   # carries into the zone
    while srv.status(a0).phase != "carry":
        time.sleep(0.005)
    a1 = srv.start("arm_1", "pick_and_place", {"object": "block_1", "place": "tray_slot_1"})   # reaches into the zone
    time.sleep(0.1)
    st = srv.status(a1)
    assert st.state == "running" and st.phase == "wait_zone"
    assert st.phase_text == "waiting outside the shared zone for arm_0 to leave it, not yet at block 1"
    assert srv.world().arms["arm_1"].mode == "running"
    while srv.status(a1).state == "running":
        time.sleep(0.01)
    assert srv.status(a0).state == srv.status(a1).state == "done"
    assert where(srv)["block_1"] == "tray_slot_1"
    log = [(arm, what) for _, arm, what in srv.zone_log]
    assert log == [("arm_0", "enter"), ("arm_1", "wait"), ("arm_0", "leave"), ("arm_1", "enter"), ("arm_1", "leave")]


def test_stub_refuses_a_grasp_on_the_other_arms_held_object(pair):
    srv = pair(pick_s=2.0)
    srv.start("arm_0", "pick_and_place", {"object": "block_1", "place": "zone_slot_1"})
    while srv.world().arms["arm_0"].holding != "block_1":
        time.sleep(0.01)
    assert srv.precondition("arm_1", "pick_and_place", {"object": "block_1", "place": "tray_slot_1"}) == \
        "precondition: block 1 is in arm_0's gripper"


# ---------------------------------------------------------------- (f) one arm: byte-identical requests

# sha256 of every planner request (instructions, input, schema; plan, its retry, replan, its retry) on
# the one-arm stub manifests, taken before H2 changed the brain.
ONE_ARM_REQUESTS = {"widowx-stub": "d202c1112219faaf5d6d7b147067be3ee01260391a04c65bb02396d08589388f",
                    "pusher-stub": "ead1d15fe675c8c8eb00c9bbdab104662b2d5fbc1b138260d7b4a2a7469fa4ff"}


def _one_arm_requests(manifest) -> list:
    objs, places = S.sort_scene((5, 1, 3))
    srv = S.StubRobotServer(manifest, objs, places)
    llm = ScriptedLLM([{"bad": 1}], latency_ms=0)
    b = Brain(srv, decider=None, planner={"fast_llm": llm})
    w = srv.world()
    core = core_state(w, "arm_0", geometry(manifest.arms[0], 0.2), 0.2)
    b.planner.plan(SORT_TASK, ["hi"], w, core)
    plan = {"reading": "r", "steps": [{"id": "s1", "skill": "move_object", "object": "block_1", "target": "tray_slot_1",
                                       "direction": "none"}],
            "done_when": [{"id": "d1", "text": "t", "object": "block_1", "relation": "in", "target": "tray_slot_1"}],
            "constraints": []}
    b.planner.replan("fast_llm", "task", ["hi"], w, core, plan, ["s1"], {"s1": "pending"},
                     {"current_step": "s1", "doing": "x", "while_you_plan": "carrying on"}, "why")
    return [(r.kind, r.name, r.instructions, r.input, json.dumps(r.schema)) for r in llm.requests]


@pytest.mark.parametrize("manifest", [S.widowx_like(), S.push_only()], ids=lambda m: m.robot)
def test_f_one_arm_planner_requests_are_byte_identical_to_before(manifest):
    reqs = _one_arm_requests(manifest)
    assert len(reqs) == 4
    assert hashlib.sha256(json.dumps(reqs).encode()).hexdigest() == ONE_ARM_REQUESTS[manifest.robot]
    assert all('"arm"' not in sch and "reached_by" not in inp for *_, inp, sch in reqs)


def test_f_two_arm_schema_adds_an_arm_enum_on_every_step(pair):
    srv = pair()
    b = brain(srv)
    ids = {"objects": ["block_1"], "places": ["tray_slot_1"]}
    for sch in (b.planner.plan_schema(ids)["properties"]["steps"], b.planner.diff_schema(ids)["properties"]["new_steps"]):
        item = sch["items"]
        assert list(item["properties"])[:3] == ["id", "arm", "skill"] and "arm" in item["required"]
        assert item["properties"]["arm"]["enum"] == list(ARMS)
    one = Brain(srv, decider=None, planner={"fast_llm": None}, cfg=BrainConfig(arm="arm_1"))   # one arm of the pair
    assert "arm" not in one.planner.step_schema(ids)["properties"]


# ---------------------------------------------------------------- (g) M1: the push-only arm


def test_g_m1_same_brain_runs_the_push_only_one_arm_manifest():
    objs, places = S.sort_scene((5, 1, 3))
    with S.StubRobotServer(S.push_only(), objs, places) as srv:
        srv.person_moves(0.0, "block_1", to="tray_slot_1")
        while where(srv)["block_1"] != "tray_slot_1":
            time.sleep(0.01)
        plan = {"reading": "push block 1 out of tray slot 1", "constraints": [],
                "steps": [{"id": "s1", "skill": "push", "object": "block_1", "target": "none", "direction": "away_from_robot",
                           "distance": 0.1, "seconds": None}],
                "done_when": [{"id": "d1", "text": "block_1 has been pushed out of tray_slot_1", "object": "block_1",
                               "relation": "other", "target": "none"}]}
        llm = ScriptedLLM([plan])
        b = Brain(srv, oracle_decider(srv, [Goal({})]), {"fast_llm": llm}, None, BrainConfig(max_s=20))
        assert list(b.arms) == ["arm_0"] and not b.multi
        r = episode(b, "Push block 1 out of tray slot 1, away from the robot.")
        assert r.outcome == "done", r.log
        assert [c[2][1] for c in srv.calls if c[1] == "start"] == ["push"]
        assert where(srv)["block_1"] == "table"
        body = json.loads(llm.requests[0].input)
        assert "arm" in body["world"] and "arms" not in body["world"]          # the one-arm world view
        assert {d["arm"] for d in r.decisions} == {"arm_0"}
