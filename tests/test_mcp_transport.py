"""K6 (experiments/skills_sim/mcp_server.py, mcp_client.py): a RobotServer as a real MCP server, and the
brain's MCPRobotServer client, over in-process memory streams and a stdio subprocess.

Skipped when the `mcp` extra is not installed (uv run --extra mcp). The brain scenario is
test_brain_loop's 3-block sort with the mock decider and planner, the brain untouched.
"""
import asyncio
import statistics
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
pytest.importorskip("mcp")

from e12v2.eval.scenarios import SORT_TASK, sort_goal  # noqa: E402
from skills_sim import catalog  # noqa: E402
from skills_sim import stub_server as S  # noqa: E402
from skills_sim.brain import Brain, BrainConfig  # noqa: E402
from skills_sim.brain.mocks import oracle_models  # noqa: E402
from skills_sim.mcp_client import MCPRobotServer  # noqa: E402
from skills_sim.mcp_server import (CATALOG_URI, CONTROL_TOOLS, MANIFEST_URI, PACKAGE_PARENT, SKILL_URI_TEMPLATE, WORLD_URI,  # noqa: E402
                                   stub_command)
from skills_sim.protocol import RobotServer  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402

PICK = {"object": "block_1", "place": "tray_slot_1"}


# ---------------------------------------------------------------- fixtures


def make_stub(pick_s=0.4, survey_s=0.5, **kw) -> S.StubRobotServer:
    objs, places = S.sort_scene((5, 1, 3))
    return S.StubRobotServer(S.widowx_like(), objs, places, durations={"pick_and_place": pick_s, "survey": survey_s}, **kw)


@pytest.fixture
def stub():
    with make_stub() as robot:
        yield robot


@pytest.fixture
def local(stub):
    """MCPRobotServer over memory streams in front of `stub`, the whole MCP stack and no pipe."""
    with MCPRobotServer.in_process(stub) as srv:
        yield srv


@pytest.fixture
def remote():
    """MCPRobotServer over stdio to a subprocess serving the stub."""
    with MCPRobotServer.stdio(stub_command(), cwd=PACKAGE_PARENT) as srv:
        yield srv


@pytest.fixture(params=["local", "remote"])
def srv(request):
    return request.getfixturevalue(request.param)


def wait_for(cond, timeout=5.0, step=0.01):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(step)


def where(srv) -> dict:
    return {o.id: o.where for o in srv.world().objects.values()}


# ---------------------------------------------------------------- discovery: tools and resources


def test_client_is_a_robot_server(local):
    assert isinstance(local, RobotServer)
    assert local.protocol_version == "2026-07-28"


def test_tools_are_the_manifest_skills_plus_the_control_tools_with_schemas(srv):
    tools = {t.name: t for t in srv.tools()}
    m = S.widowx_like()
    assert list(tools) == [s.name for s in m.skills] + list(CONTROL_TOOLS)
    suffixed = {"hold": "hold_arm", "pause": "pause_arm", "resume": "resume_arm", "retarget": "retarget_skill"}
    assert {suffixed.get(t, t) for t in catalog.CONTROL_TOOLS} <= set(CONTROL_TOOLS)    # every mandatory tool, no collision
    for s in m.skills:
        schema = tools[s.name].input_schema
        assert schema["properties"]["arm"] == {"type": "string", "enum": ["arm_0"], "description": "arm id from the manifest"}
        assert schema["required"][0] == "arm" and schema["required"][1:] == s.args_schema.get("required", [])
        without_arm = {**schema, "properties": {k: v for k, v in schema["properties"].items() if k != "arm"},
                       "required": schema["required"][1:]}
        if "required" not in s.args_schema:
            del without_arm["required"]
        assert without_arm == s.args_schema                      # the catalog's schema, untouched
        assert tools[s.name].description == s.description
    for name in CONTROL_TOOLS:
        schema = tools[name].input_schema
        assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert tools["hold_arm"].input_schema["required"] == ["arm"]
    assert tools["precondition"].input_schema["properties"]["skill"]["enum"] == [s.name for s in m.skills]
    assert tools["stop"].input_schema["properties"] == {}


def test_resources_are_manifest_world_catalog_and_the_skill_template(srv):
    resources, templates = srv.resources()
    assert [r.uri for r in resources] == [MANIFEST_URI, WORLD_URI, CATALOG_URI]
    assert [t.uri_template for t in templates] == [SKILL_URI_TEMPLATE]
    assert all(r.mime_type == "application/json" for r in resources)


def test_resources_read_back_equal_to_the_direct_servers(stub, local):
    assert local.manifest() == stub.manifest()
    assert local.manifest() is local.manifest()                  # read once: a Manifest is immutable
    assert local.catalog_version() == stub.manifest().catalog == catalog.CATALOG_VERSION
    assert replace(local.world(), t=0.0) == replace(stub.world(), t=0.0)
    sid = stub.start("arm_0", "pick_and_place", PICK)
    stub.pause("arm_0")                                           # frozen, so two reads agree
    assert local.status(sid) == stub.status(sid)
    assert local.status(sid).state == "paused" and local.world().arms["arm_0"].mode == "paused"
    with pytest.raises(KeyError, match="no skill sk99"):
        local.status("sk99")


# ---------------------------------------------------------------- tools: start, refusal, control


def test_a_skill_started_over_mcp_runs_to_a_skill_done_event_on_the_client_callback(srv):
    got, threads = [], set()

    def on_event(ev):
        got.append(ev)
        threads.add(threading.current_thread().name)
        srv.world()                                               # calling back in from the callback is allowed
    srv.subscribe(on_event)
    sid = srv.start("arm_0", "pick_and_place", PICK)
    assert sid == "sk1" and srv.status(sid).state == "running" and srv.status(sid).phase == "reach"
    wait_for(lambda: got)
    assert [(e.kind, e.skill_id, e.arm, e.object) for e in got] == [("skill_done", sid, "arm_0", "block_1")]
    assert threads == {"mcp-events"}
    st = srv.status(sid)
    assert (st.state, st.phase, st.reason) == ("done", "done", None) and "tray slot 1" in st.phase_text
    assert where(srv)["block_1"] == "tray_slot_1"
    srv.unsubscribe(on_event)
    srv.start("arm_0", "survey", {})
    time.sleep(0.7)
    assert len(got) == 1                                          # unsubscribed: no more deliveries


def test_a_refusal_is_a_value_error_with_the_literal_reason_and_no_event(srv):
    got = []
    srv.subscribe(got.append)
    bad = [("pick_and_place", {"object": "block_1"}, "precondition: pick_and_place needs place"),
           ("pick_and_place", {"object": "block_1", "place": "tray_slot_1", "speed": 2}, "takes no argument speed"),
           ("push", {"object": "block_1", "direction": "up", "distance": 0.1}, "direction must be one of"),
           ("pick_and_place", {"object": "block_9", "place": "tray_slot_1"}, "not_in_view: block_9 is not in view")]
    for skill, args, why in bad:
        assert why in srv.precondition("arm_0", skill, args)
        with pytest.raises(ValueError, match=why):
            srv.start("arm_0", skill, args)
    assert srv.precondition("arm_0", "pick_and_place", PICK) is None
    with pytest.raises(ValueError, match="arm_9 has no skill survey"):
        srv.start("arm_9", "survey", {})
    with pytest.raises(ValueError, match="precondition: arm_0 has no skill teleport"):
        srv.start("arm_0", "teleport", {})                        # not a tool: the server's reason, still a ValueError
    time.sleep(0.1)
    assert got == [] and srv.world().arms["arm_0"].mode == "idle"


def test_control_is_idempotent_over_the_wire(srv):
    srv.hold("arm_0")
    srv.resume("arm_0")
    srv.pause("arm_0")                                            # idle arm: no-ops
    assert srv.world().arms["arm_0"].mode == "idle"
    sid = srv.start("arm_0", "hand_over", {"object": "block_1"})
    srv.resume("arm_0")
    assert srv.status(sid).state == "running"
    srv.pause("arm_0")
    srv.hold("arm_0")                                             # a pause is the stronger state
    srv.pause("arm_0")
    assert srv.status(sid).state == "paused" and srv.world().arms["arm_0"].mode == "paused"
    srv.resume("arm_0")
    srv.resume("arm_0")
    assert srv.status(sid).state == "running"
    srv.retarget(sid)
    srv.retarget(sid)
    got = []
    srv.subscribe(got.append)
    srv.stop()
    srv.stop()
    wait_for(lambda: got)
    assert [(e.kind, e.skill_id) for e in got] == [("skill_failed", sid)]
    assert srv.status(sid).reason.startswith("stopped: ") and srv.world().arms["arm_0"].mode == "stopped"
    with pytest.raises(ValueError, match="STOP"):
        srv.start("arm_0", "survey", {})
    srv.heartbeat()


def test_a_failed_skill_reports_code_colon_text_in_status_and_event(stub, local):
    got = []
    local.subscribe(got.append)
    sid = local.start("arm_0", "pick_and_place", PICK)
    stub.person_moves(0.0, "block_1", xy=(0.30, -0.20))            # the person in the stub process
    wait_for(lambda: len(got) >= 2)
    st = local.status(sid)
    assert st.state == "failed" and st.reason.startswith("grasp_failed: ") and got[1].text == st.reason
    assert got[1].data == {"reason": st.reason}
    assert [(e.kind, e.skill_id) for e in got] == [("scene_change", None), ("skill_failed", sid)]


def test_scene_changes_and_safety_trips_cross_the_wire(stub, local):
    got = []
    local.subscribe(got.append)
    stub.person_moves(0.0, "block_3", to="tray_slot_2")
    stub.trip(0.0, "effort limit exceeded")
    wait_for(lambda: len(got) >= 2)
    kinds = {e.kind: e for e in got}
    assert "a person moved block 3" in kinds["scene_change"].text and kinds["scene_change"].object == "block_3"
    assert kinds["safety_trip"].arm == "arm_0" and kinds["safety_trip"].skill_id is None
    assert local.world().arms["arm_0"].mode == "tripped"
    local.resume("arm_0")
    assert local.world().arms["arm_0"].mode == "tripped"            # nothing to resume on an idle tripped arm (stub rule)


def test_legacy_handshake_era_works_too():
    with MCPRobotServer.stdio(stub_command(), cwd=PACKAGE_PARENT, mode="legacy") as srv:
        assert srv.protocol_version == "2025-11-25"
        got = []
        srv.subscribe(got.append)
        sid = srv.start("arm_0", "survey", {})
        wait_for(lambda: got)
        assert got[0].kind == "skill_done" and got[0].skill_id == sid


def test_a_server_that_will_not_start_is_reported_at_open():
    with pytest.raises(MCPError):
        MCPRobotServer.stdio([sys.executable, "-c", "import sys; sys.exit(3)"], timeout_s=2.0).open()


# ---------------------------------------------------------------- the brain, unchanged, over MCP


def numbers(srv) -> dict:
    return {o.id: int(o.label) for o in srv.world().objects.values()}


def sorted_ascending(srv) -> bool:
    w = where(srv)
    return all(w[oid] == f"tray_slot_{k}" for k, oid in enumerate(sorted(w, key=lambda o: numbers(srv)[o]), 1))


def test_brain_sorts_three_blocks_over_stdio(remote):
    srv = remote
    decider, llm = oracle_models(srv, [sort_goal(numbers(srv))])
    brain = Brain(srv, decider, llm, None, BrainConfig(max_s=20))
    t = time.monotonic()
    r = asyncio.run(brain.run(SORT_TASK))
    wall = time.monotonic() - t
    assert r.outcome == "done", r.log
    assert wall < 10.0
    assert sorted_ascending(srv)
    assert r.steps_done == 3 and r.steps_failed == 0 and r.failed_plans == 0
    decided = [d["event_id"] for d in r.decisions]
    assert len(decided) == len(set(decided))


def test_brain_sorts_three_blocks_in_process(stub, local):
    decider, llm = oracle_models(local, [sort_goal(numbers(local))])
    r = asyncio.run(Brain(local, decider, llm, None, BrainConfig(max_s=20)).run(SORT_TASK))
    assert r.outcome == "done", r.log
    assert sorted_ascending(local)
    assert [c[2][1] for c in stub.calls if c[1] == "start"] == ["pick_and_place"] * 3    # catalog names on the wire
    expected = r.wall_s / 0.2
    assert expected - 2 <= stub.heartbeats <= expected + 2


# ---------------------------------------------------------------- latency


def pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) >= 2 else xs[0]


def round_trips(fn, n=200) -> list[float]:
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def clock_offset(srv, n=20) -> float:
    """The server's t0 on this host's monotonic clock: world().t is monotonic - t0 in the server
    process, read at the midpoint of a round trip (error: half a round trip)."""
    est = []
    for _ in range(n):
        a = time.monotonic()
        t = srv.world().t
        b = time.monotonic()
        est.append((a + b) / 2 - t)
    return statistics.median(est)


def event_delays(srv, n=200, survey_s=0.02) -> list[float]:
    """Server emits (RobotEvent.t) -> client callback, one survey skill at a time."""
    t0 = clock_offset(srv)
    seen = []
    done = threading.Event()

    def on_event(ev):
        if ev.kind == "skill_done":
            seen.append(time.monotonic() - (t0 + ev.t))
            done.set()
    srv.subscribe(on_event)
    try:
        for _ in range(n):
            done.clear()
            srv.start("arm_0", "survey", {})
            assert done.wait(2.0)
    finally:
        srv.unsubscribe(on_event)
    return seen


def measure(srv) -> dict[str, list[float]]:
    """Round trips of heartbeat() and world(), and event delivery, with the brain's 200 ms heartbeat
    running beside them (the stub holds every arm when heartbeats stop for 1 s)."""
    stop = threading.Event()

    def beat():
        while not stop.wait(0.2):
            srv.heartbeat()
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        return {"heartbeat()": round_trips(srv.heartbeat), "world()": round_trips(srv.world), "event": event_delays(srv)}
    finally:
        stop.set()
        t.join()


def report(label: str, m: dict[str, list[float]]) -> str:
    rows = [f"  {k:12s} p50 {pct(v, 50)*1e3:6.2f} ms   p95 {pct(v, 95)*1e3:6.2f} ms   max {max(v)*1e3:6.2f} ms   n={len(v)}"
            for k, v in m.items()]
    return "\n".join([f"{label}:"] + rows)


@pytest.mark.parametrize("mode", ["2026-07-28", "legacy"])
def test_latency_over_stdio(mode):
    with MCPRobotServer.stdio(stub_command(survey_s=0.02, tick_s=0.005), cwd=PACKAGE_PARENT, mode=mode) as srv:
        m = measure(srv)
        print("\n" + report(f"stdio, protocol {srv.protocol_version}", m))
    assert pct(m["event"], 95) < 0.100, report("event delivery too slow", m)
    assert pct(m["heartbeat()"], 95) < 0.100 and pct(m["world()"], 95) < 0.100


def test_latency_in_process():
    with make_stub(survey_s=0.02, tick_s=0.005) as robot:
        with MCPRobotServer.in_process(robot) as srv:
            m = measure(srv)
            print("\n" + report("in-process memory streams", m))
    assert pct(m["event"], 95) < 0.100
