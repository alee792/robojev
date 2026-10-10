"""Any RobotServer as a real MCP server: spike K6 (docs/harness-spikes.md, docs/architecture.md).

protocol.RobotServer was shaped like MCP; this makes it MCP (the `mcp` 2.x SDK: protocol 2026-07-28,
or the 2025 handshake if the client opens with one), over stdio or in-process memory streams. The
brain drives it unchanged through mcp_client.MCPRobotServer, which shares this module's wire format.

The mapping, one MCP tool, resource or notification per protocol method:

  tools          Every skill in the manifest is a tool named by its catalog name, with the
                 manifest's args_schema plus `arm` (string, enum of the arm ids that can run it;
                 required). Calling it is start(arm, skill, args) and returns {"skill_id"} at once;
                 a refusal (ValueError) is a tool error (isError, the literal reason as its text) and
                 emits no event. Control tools: hold_arm, pause_arm, resume_arm, retarget_skill,
                 stop, heartbeat, precondition (-> {"why": reason or null}). The suffixes keep the
                 `hold` tool and the `hold` skill apart.
  resources      robot://manifest, robot://world, robot://catalog ({"catalog": version}) and the
                 template robot://skills/{id} (unknown id: INVALID_PARAMS, the client's KeyError).
                 JSON text bodies; dataclasses encoded by `to_json` (dataclasses.asdict, so tuples
                 arrive as lists) and decoded by the explicit *_from_json functions below.
  notifications  Every RobotEvent is one server->client notification, EVENT_METHOD, with params
                 {"event": <RobotEvent as JSON>}, pushed on the connection's standalone channel the
                 moment the robot emits it. Not the SDK's subscriptions/listen: at 2026-07-28 that
                 stream carries a closed vocabulary (resource-updated, list-changed), so an event
                 would cost a notification plus a resources/read; a method of our own is what the
                 spec's extension mechanism (SEP-2133) exists for, and the client binds it by name.

What the SDK lacks for a long-running, stateful, event-emitting server, and how this copes:
  - No "connection opened" hook, and sessions are per request: a server first learns of its client
    from the client's first request. A middleware remembers the latest ServerSession (its standalone
    channel is the connection's), and the event pump sends through it. Events before any request go
    nowhere, which the protocol allows: a connecting brain reads world() and status() itself.
  - Handlers are coroutines on one event loop; RobotServer calls are synchronous and brief, so they
    run inline. The robot emits events on its own thread; call_soon_threadsafe hands them to the
    loop (asyncio backend).
One connection at a time (stdio has one; the memory transport makes one).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams

from .protocol import (ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, ObjectObs, PlaceObs, RobotEvent, RobotServer,
                       SkillSpec, SkillStatus, WorldState)

# ------------------------------------------------------------------ the wire format (shared with mcp_client)

EVENT_METHOD = "notifications/robot/event"
MANIFEST_URI, WORLD_URI, CATALOG_URI = "robot://manifest", "robot://world", "robot://catalog"
SKILL_URI_TEMPLATE = "robot://skills/{id}"
SKILL_URI_PREFIX = "robot://skills/"
CONTROL_TOOLS = ("hold_arm", "pause_arm", "resume_arm", "retarget_skill", "stop", "heartbeat", "precondition")
JSON = "application/json"


def skill_uri(skill_id: str) -> str:
    return SKILL_URI_PREFIX + skill_id


def to_json(obj: Any) -> dict:
    """A protocol dataclass as JSON data (dataclasses.asdict: nested dataclasses become dicts, tuples
    survive until json.dumps makes them lists)."""
    return asdict(obj)


def manifest_from_json(d: dict) -> Manifest:
    arms = tuple(ArmSpec(a["id"], a["base_frame"], tuple(tuple(r) for r in a["workspace"]),
                         GripperSpec(**a["gripper"]) if a["gripper"] is not None else None, a["travel_z"], a["description"])
                 for a in d["arms"])
    skills = tuple(SkillSpec(s["name"], s["description"], s["args_schema"], tuple(s["arms"])) for s in d["skills"])
    return Manifest(d["robot"], arms, skills, d["units"], d["catalog"])


def world_from_json(d: dict) -> WorldState:
    return WorldState(d["t"], {k: ObjectObs(**o) for k, o in d["objects"].items()},
                      {k: PlaceObs(**{**p, "members": tuple(p["members"])}) for k, p in d["places"].items()},
                      [HandObs(**h) for h in d["hands"]], {k: ArmObs(**a) for k, a in d["arms"].items()})


def status_from_json(d: dict) -> SkillStatus:
    return SkillStatus(**{**d, "heading": tuple(d["heading"]) if d["heading"] is not None else None})


def event_from_json(d: dict) -> RobotEvent:
    return RobotEvent(**d)


class RobotEventParams(types.NotificationParams):
    """Params of EVENT_METHOD. The event sits in a plain dict so the SDK's camelCase aliasing never
    touches the protocol's field names."""
    event: dict[str, Any]


class RobotEventNotification(types.Notification[RobotEventParams, Literal["notifications/robot/event"]]):
    method: Literal["notifications/robot/event"] = EVENT_METHOD


# ------------------------------------------------------------------ tools


def _arm_property(arms: tuple[str, ...]) -> dict:
    return {"type": "string", "enum": list(arms), "description": "arm id from the manifest"}


def _object_schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def skill_tool(spec: SkillSpec, arm_ids: tuple[str, ...]) -> types.Tool:
    """The manifest's skill as an MCP tool: its args_schema with `arm` added (first, required), the
    enum limited to the arms that can run it."""
    schema = copy.deepcopy(spec.args_schema)
    schema["properties"] = {"arm": _arm_property(spec.arms or arm_ids), **schema.get("properties", {})}
    schema["required"] = ["arm", *schema.get("required", ())]
    return types.Tool(name=spec.name, description=spec.description, input_schema=schema)


def control_tools(m: Manifest) -> list[types.Tool]:
    """The mandatory control tools (catalog.CONTROL_TOOLS plus precondition), schemas bound to this
    manifest's arm and skill names. Names match CONTROL_TOOLS, in order."""
    arms = tuple(a.id for a in m.arms)
    arm_only = _object_schema({"arm": _arm_property(arms)}, ["arm"])
    none = _object_schema({}, [])
    return [
        types.Tool(name="hold_arm", input_schema=arm_only,
                   description="Stop the arm at a safe point, keeping the grip and the skill's state; resume_arm continues. "
                               "No-op on an idle arm."),
        types.Tool(name="pause_arm", input_schema=arm_only,
                   description="Like hold_arm, but the skill will not continue until resume_arm (a person is near). "
                               "No-op on an idle arm."),
        types.Tool(name="resume_arm", input_schema=arm_only,
                   description="Continue after hold_arm or pause_arm; the only way to clear a tripped arm."),
        types.Tool(name="retarget_skill", input_schema=_object_schema({"skill_id": {"type": "string"}}, ["skill_id"]),
                   description="Aim the running skill at where its object is now."),
        types.Tool(name="stop", input_schema=none,
                   description="STOP: every arm parks, every running skill fails with \"stopped: ...\", later starts "
                               "are refused. Terminal for this server."),
        types.Tool(name="heartbeat", input_schema=none,
                   description="The client is alive. A server that misses heartbeats for its timeout holds every arm."),
        types.Tool(name="precondition",
                   input_schema=_object_schema({"arm": _arm_property(arms),
                                                "skill": {"type": "string", "enum": [s.name for s in m.skills]},
                                                "args": {"type": "object", "description": "the skill's arguments"}},
                                               ["arm", "skill", "args"]),
                   description="Could the skill start now on this arm? Returns {\"why\": null} if so, else the literal "
                               "reason it could not."),
    ]


def _arg(args: dict, tool: str, key: str) -> Any:
    if key not in args:
        raise ValueError(f"{tool} needs {key}")
    return args[key]


# ------------------------------------------------------------------ the server


class RobotMCPServer:
    """`robot` behind MCP. `serve(read, write)` runs one connection to the end of its streams;
    `stdio()` serves the process's stdin/stdout; `memory_transport()` is an mcp client Transport
    that serves in a task beside the client, for tests. Tool and resource handlers call the robot
    inline; events reach the client through `_pump`."""

    def __init__(self, robot: RobotServer, name: str = "robot"):
        self.robot = robot
        m = robot.manifest()
        arm_ids = tuple(a.id for a in m.arms)
        self._skills = {s.name for s in m.skills}
        self._tools = [skill_tool(s, arm_ids) for s in m.skills] + control_tools(m)
        self._control: dict[str, Callable[[dict], dict | None]] = {
            "hold_arm": lambda a: robot.hold(_arg(a, "hold_arm", "arm")),
            "pause_arm": lambda a: robot.pause(_arg(a, "pause_arm", "arm")),
            "resume_arm": lambda a: robot.resume(_arg(a, "resume_arm", "arm")),
            "retarget_skill": lambda a: robot.retarget(_arg(a, "retarget_skill", "skill_id")),
            "stop": lambda a: robot.stop(),
            "heartbeat": lambda a: robot.heartbeat(),
            "precondition": lambda a: {"why": robot.precondition(_arg(a, "precondition", "arm"), _arg(a, "precondition", "skill"),
                                                                 _arg(a, "precondition", "args"))},
        }
        self._session = None                    # the latest request's ServerSession: the connection's channel
        self.server: Server = Server(
            name, version=m.catalog, lifespan=self._lifespan,
            instructions=f"{m.robot}: a robot server. Skills are tools (start at once, finish as {EVENT_METHOD}); "
                         f"state is {WORLD_URI} and {SKILL_URI_TEMPLATE}; the client sends heartbeat every 200 ms.",
            on_list_tools=self._list_tools, on_call_tool=self._call_tool,
            on_list_resources=self._list_resources, on_list_resource_templates=self._list_resource_templates,
            on_read_resource=self._read_resource)
        self.server.middleware.append(self._remember_session)

    # ---------------------------------------------------------------- serving
    async def serve(self, read_stream, write_stream) -> None:
        await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

    async def stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.serve(read_stream, write_stream)

    @asynccontextmanager
    async def memory_transport(self) -> AsyncIterator[tuple]:
        """Yields (read, write) client streams of an in-process connection served in a background
        task; closing them ends the server side (EOF, then a bounded wait before cancelling)."""
        async with create_client_server_memory_streams() as ((c_read, c_write), (s_read, s_write)):
            served = anyio.Event()

            async def run() -> None:
                try:
                    await self.serve(s_read, s_write)
                finally:
                    served.set()
            async with anyio.create_task_group() as tg:
                tg.start_soon(run)
                try:
                    yield c_read, c_write
                finally:
                    await c_write.aclose()
                    with anyio.move_on_after(2.0):
                        await served.wait()
                    tg.cancel_scope.cancel()

    # ---------------------------------------------------------------- events -> notifications
    @asynccontextmanager
    async def _lifespan(self, _server: Server) -> AsyncIterator[dict]:
        loop, queue = asyncio.get_running_loop(), asyncio.Queue()

        def on_event(ev: RobotEvent) -> None:       # the robot's thread; the loop may already be gone at shutdown
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            except RuntimeError:
                pass
        self.robot.subscribe(on_event)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._pump, queue)
                try:
                    yield {}
                finally:
                    tg.cancel_scope.cancel()
        finally:
            self.robot.unsubscribe(on_event)

    async def _pump(self, queue: asyncio.Queue) -> None:
        while True:
            ev = await queue.get()
            if self._session is not None:
                await self._session.send_notification(RobotEventNotification(params=RobotEventParams(event=to_json(ev))))

    async def _remember_session(self, ctx, call_next):
        self._session = ctx.session
        return await call_next(ctx)

    # ---------------------------------------------------------------- tools
    async def _list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self._tools)

    async def _call_tool(self, ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        name, args = params.name, dict(params.arguments or {})
        try:
            if name in self._skills:
                arm = _arg(args, name, "arm")
                del args["arm"]
                out = {"skill_id": self.robot.start(arm, name, args)}
            elif name in self._control:
                out = self._control[name](args) or {}
            else:
                raise MCPError(types.INVALID_PARAMS, f"no tool {name}")
        except ValueError as e:                     # a refusal: the literal reason, no event
            return types.CallToolResult(content=[types.TextContent(text=str(e))], is_error=True)
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(out))], structured_content=out)

    # ---------------------------------------------------------------- resources
    async def _list_resources(self, ctx, params) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=[
            types.Resource(uri=MANIFEST_URI, name="manifest", mime_type=JSON,
                           description="arms, skills (name, description, args_schema) and the catalog version"),
            types.Resource(uri=WORLD_URI, name="world", mime_type=JSON,
                           description="what perception reports now: objects, places, hands, arms"),
            types.Resource(uri=CATALOG_URI, name="catalog", mime_type=JSON, description="the catalog version")])

    async def _list_resource_templates(self, ctx, params) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resource_templates=[
            types.ResourceTemplate(uri_template=SKILL_URI_TEMPLATE, name="skill status", mime_type=JSON,
                                   description="where a skill is: state, literal phase text, reason if it failed")])

    async def _read_resource(self, ctx, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = str(params.uri)
        if uri == MANIFEST_URI:
            body = to_json(self.robot.manifest())
        elif uri == WORLD_URI:
            body = to_json(self.robot.world())
        elif uri == CATALOG_URI:
            body = {"catalog": self.robot.manifest().catalog}
        elif uri.startswith(SKILL_URI_PREFIX):
            try:
                body = to_json(self.robot.status(uri[len(SKILL_URI_PREFIX):]))
            except KeyError as e:
                raise MCPError(types.INVALID_PARAMS, str(e.args[0]) if e.args else uri) from None
        else:
            raise MCPError(types.INVALID_PARAMS, f"no resource {uri}")
        return types.ReadResourceResult(contents=[types.TextResourceContents(uri=uri, mime_type=JSON, text=json.dumps(body))])


# ------------------------------------------------------------------ the stub over stdio

PACKAGE_PARENT = Path(__file__).resolve().parents[1]      # experiments/: the cwd `stub_command()` needs


def stub_command(pick_s: float = 0.4, survey_s: float = 0.5, tick_s: float = 0.02) -> list[str]:
    """argv of a subprocess serving stub_server.widowx_like() on the 3-block sort scene over stdio:
    pick_and_place takes `pick_s`, survey `survey_s`, the clock ticks every `tick_s`. Run it with
    cwd=PACKAGE_PARENT so `skills_sim` imports."""
    return [sys.executable, "-m", "skills_sim.mcp_server", "--pick-s", str(pick_s), "--survey-s", str(survey_s),
            "--tick-s", str(tick_s)]


def main(argv: list[str] | None = None) -> None:
    from . import stub_server as S
    p = argparse.ArgumentParser(description="serve the physics-free stub robot over MCP on stdio")
    p.add_argument("--pick-s", type=float, default=0.4, help="seconds a pick_and_place takes")
    p.add_argument("--survey-s", type=float, default=0.5, help="seconds a survey takes")
    p.add_argument("--tick-s", type=float, default=0.02, help="the stub's clock period")
    p.add_argument("--numbers", default="5,1,3", help="block labels, left to right")
    a = p.parse_args(argv)
    objs, places = S.sort_scene(tuple(int(n) for n in a.numbers.split(",")))
    with S.StubRobotServer(S.widowx_like(), objs, places, durations={"pick_and_place": a.pick_s, "survey": a.survey_s},
                           tick_s=a.tick_s) as robot:
        asyncio.run(RobotMCPServer(robot).stdio())


if __name__ == "__main__":
    main()
