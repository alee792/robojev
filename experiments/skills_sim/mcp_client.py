"""The brain's end of K6: protocol.RobotServer over an MCP session, so Brain(MCPRobotServer(...)) runs
unchanged on any robot that mcp_server.RobotMCPServer fronts.

Threads. The MCP session (mcp 2.x ClientSession) lives on its own asyncio loop in an `mcp-io`
thread; every protocol method posts one coroutine there and waits for it, so calls are safe from
any thread and cost one round trip (`timeout_s` bounds each). Events arrive on the io thread as
mcp_server.EVENT_METHOD notifications and are queued to an `mcp-events` thread, where subscribers'
callbacks run in order: the protocol's "the server's thread; return fast". A callback may call
back into this object, because it is not the io thread.

Rules kept. start() refused -> ValueError with the server's literal reason (the tool error's text),
and no event follows. status() of an unknown id -> KeyError. hold/pause/resume/retarget/stop/
heartbeat return nothing and are idempotent on the server. manifest() is read once (a Manifest is
immutable). Every other call reads the server afresh: nothing is cached.

Transports. `stdio(argv, cwd)` starts a subprocess (the `mcp` stdio transport), `in_process(robot)`
serves a RobotMCPServer over memory streams on the io loop (fast tests), and the constructor takes
any mcp client Transport. `mode` is the protocol era: a modern version ("2026-07-28", the default:
server/discover, then every request carries its envelope) or "legacy" (the 2025 initialize handshake).
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mcp_types as types
from mcp.client import ClientSession, NotificationBinding, Transport
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_types.version import LATEST_MODERN_VERSION
from mcp.shared.exceptions import MCPError

from .mcp_server import (CATALOG_URI, EVENT_METHOD, MANIFEST_URI, WORLD_URI, RobotEventParams, RobotMCPServer,
                         event_from_json, manifest_from_json, skill_uri, status_from_json, world_from_json)
from .protocol import Manifest, RobotEvent, RobotServer, SkillStatus, WorldState

_CLOSE = object()


class ResourceRefused(Exception):
    """The server refused a resources/read (an unknown skill id, an unknown URI)."""


class MCPRobotServer:
    """Implements protocol.RobotServer over MCP. open()/close() (or `with`) bracket the session;
    `tools()`, `resources()` and `read(uri)` expose the raw MCP surface for tests and tooling."""

    def __init__(self, transport: Transport, *, mode: str = LATEST_MODERN_VERSION, timeout_s: float = 5.0):
        self.transport, self.mode, self.timeout_s = transport, mode, timeout_s
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done: asyncio.Event | None = None
        self._up = threading.Event()
        self._error: BaseException | None = None
        self._manifest: Manifest | None = None
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._io = threading.Thread(target=self._io_main, name="mcp-io", daemon=True)
        self._deliverer = threading.Thread(target=self._deliver, name="mcp-events", daemon=True)

    @classmethod
    def stdio(cls, argv: Sequence[str], cwd: str | Path | None = None, env: dict[str, str] | None = None, **kw) -> MCPRobotServer:
        """A server subprocess `argv` talking on its stdin/stdout (its stderr is inherited)."""
        params = StdioServerParameters(command=argv[0], args=list(argv[1:]), cwd=cwd, env=env)
        return cls(stdio_client(params), **kw)

    @classmethod
    def in_process(cls, robot: RobotServer, **kw) -> MCPRobotServer:
        """`robot` served over memory streams in this process: the whole MCP stack, no pipe."""
        return cls(RobotMCPServer(robot).memory_transport(), **kw)

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> MCPRobotServer:
        """Connect; raises what the connection raised (a subprocess that would not start, a refused
        handshake) instead of returning a dead server."""
        self._deliverer.start()
        self._io.start()
        self._up.wait()
        if self._error is not None:
            self._io.join(timeout=self.timeout_s)
            raise _leaf(self._error)
        return self

    def close(self) -> None:
        if self._loop is not None and self._done is not None and self._io.is_alive():
            self._loop.call_soon_threadsafe(self._done.set)
            self._io.join(timeout=self.timeout_s + 5.0)
        if self._deliverer.is_alive():
            self._events.put(_CLOSE)
            self._deliverer.join(timeout=self.timeout_s)

    def __enter__(self) -> MCPRobotServer:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def protocol_version(self) -> str | None:
        return self._session.protocol_version if self._session is not None else None

    # ---------------------------------------------------------------- protocol: resources
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = manifest_from_json(self.read(MANIFEST_URI))
        return self._manifest

    def world(self) -> WorldState:
        return world_from_json(self.read(WORLD_URI))

    def status(self, skill_id: str) -> SkillStatus:
        try:
            return status_from_json(self.read(skill_uri(skill_id)))
        except ResourceRefused as e:
            raise KeyError(str(e)) from None

    # ---------------------------------------------------------------- protocol: tools
    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        return self._tool("precondition", {"arm": arm, "skill": skill, "args": dict(args)})["why"]

    def start(self, arm: str, skill: str, args: dict) -> str:
        if self.manifest().skill(skill) is None:     # not a tool on this server: the server still words the refusal
            raise ValueError(self.precondition(arm, skill, args) or f"precondition: {arm} has no skill {skill}")
        return self._tool(skill, {"arm": arm, **args})["skill_id"]

    def hold(self, arm: str) -> None:
        self._tool("hold_arm", {"arm": arm})

    def pause(self, arm: str) -> None:
        self._tool("pause_arm", {"arm": arm})

    def resume(self, arm: str) -> None:
        self._tool("resume_arm", {"arm": arm})

    def retarget(self, skill_id: str) -> None:
        self._tool("retarget_skill", {"skill_id": skill_id})

    def stop(self) -> None:
        self._tool("stop", {})

    def heartbeat(self) -> None:
        self._tool("heartbeat", {})

    # ---------------------------------------------------------------- protocol: notifications
    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        self._subs.append(callback)

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        if callback in self._subs:
            self._subs.remove(callback)

    # ---------------------------------------------------------------- the raw MCP surface
    def tools(self) -> list[types.Tool]:
        return self._run(self._session.list_tools()).tools

    def resources(self) -> tuple[list[types.Resource], list[types.ResourceTemplate]]:
        return (self._run(self._session.list_resources()).resources,
                self._run(self._session.list_resource_templates()).resource_templates)

    def read(self, uri: str) -> Any:
        """A resource's JSON body. Raises ResourceRefused if the server refuses the URI."""
        return self._run(self._read(uri))

    def catalog_version(self) -> str:
        return self.read(CATALOG_URI)["catalog"]

    # ---------------------------------------------------------------- internals
    def _tool(self, name: str, args: dict) -> dict:
        """Call a tool; a tool error (isError) is the server's refusal: ValueError(its text)."""
        return self._run(self._call(name, args))

    async def _call(self, name: str, args: dict) -> dict:
        r = await self._session.call_tool(name, args)
        if r.is_error:
            raise ValueError("".join(c.text for c in r.content if isinstance(c, types.TextContent)))
        return r.structured_content or {}

    async def _read(self, uri: str) -> Any:
        try:
            r = await self._session.read_resource(uri)
        except MCPError as e:
            raise ResourceRefused(e.message) from None
        return json.loads(r.contents[0].text)

    def _run(self, coro):
        """Run `coro` on the io loop from any thread and wait for its result."""
        if self._loop is None or not self._io.is_alive():
            coro.close()
            raise RuntimeError("MCPRobotServer is not open")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout_s)

    def _io_main(self) -> None:
        try:
            asyncio.run(self._session_main())
        except BaseException as e:      # noqa: BLE001 - reported to open(), or lost with the connection
            self._error = e
        finally:
            self._up.set()

    async def _session_main(self) -> None:
        self._loop, self._done = asyncio.get_running_loop(), asyncio.Event()
        binding = NotificationBinding(method=EVENT_METHOD, params_type=RobotEventParams, handler=self._on_event)
        async with self.transport as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=self.timeout_s,
                                     notification_bindings=[binding]) as session:
                if self.mode == "legacy":
                    await session.initialize()
                else:
                    session.adopt(types.DiscoverResult.model_validate(await session.send_discover(self.mode)))
                self._session = session
                self._up.set()
                await self._done.wait()

    async def _on_event(self, params: RobotEventParams) -> None:
        self._events.put(event_from_json(params.event))

    def _deliver(self) -> None:
        while (ev := self._events.get()) is not _CLOSE:
            for cb in list(self._subs):
                cb(ev)


def _leaf(exc: BaseException) -> BaseException:
    """The one exception inside anyio's task-group wrapping, when there is exactly one."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc

