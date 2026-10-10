"""H1: the brain on a real clock, as a client of the robot protocol (docs/harness-spikes.md "Order").

One asyncio queue carries everything that can happen: server notifications (bridged from the server's
thread with call_soon_threadsafe), user text, decision answers, planner results, and a 200 ms tick
that also sends the server's heartbeat. The loop handles one item at a time and never waits on a
model: every decision and planner call runs in a thread and comes back as a queue item, so the arm
carries on or holds (per the right-now answer) while a plan is in flight, and several decisions can
be in flight at once. What happens on each item is e12v2's harness logic (docs/v2.md), re-expressed
for seconds and server calls instead of ticks and per-tick motor commands; the pure parts (the
decision request, combine(), the plan format, validator and diffs, the change detector) are imported.

Rules enforced here, in code:
  - STOP (the button, or a bare typed "stop") calls server.stop() on the caller's thread before
    anything queues, and ends the episode. Nothing else calls stop().
  - Models can only pause: a decision reaches the arm only through _arm_to (hold / pause / resume)
    and retarget. A paused arm resumes on a decision's "resume" or the user's word.
  - One decision per event (a trusted plan in a replan loop is the one event that skips it).
  - The in-plan fix applies only when the Router says stay_local and the weakest confidence of the
    route and the fix clears 0.8 (combine.Gates.stay_local); otherwise an LLM replans.
  - Replan loop: new plans sent back to the LLM `max_plan_rejections` (3) times since the user last
    spoke; later plans run as they come until the user says something new. The same two steps
    alternating, or one step failing 3 times, escalates to the capable LLM.
  - safety_trip: hold in code at once, then a decision about it; heartbeat every `heartbeat_s`.
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Protocol

from e12v2.core import plan as P
from e12v2.core.changes import ChangeDetector, hand_facts
from e12v2.core.data import Event
from e12v2.core.decision import View
from e12v2.core.harness import LoopDetector, is_stop

from .. import catalog
from ..protocol import RobotEvent, RobotServer
from .adapt import core_state, geometry, names as world_names, xy_cm
from .schema import ManifestPlanner, arm_skills, step_args

END = ("done", "stopped", "gave_up", "timeout")
_BAND = {"very close": "very close to", "near": "near", "mid-range": "mid-range from", "far": "far from"}


class UserChannel(Protocol):
    def attach(self, say: Callable[[str], None], stop_button: Callable[[], None]) -> None:
        """Called once from inside the running loop; `say` and `stop_button` are safe from any thread."""

    def ask(self, question: str) -> None: ...


@dataclass
class BrainConfig:
    max_s: float = 120.0
    heartbeat_s: float = 0.2
    hold_s: float = 1.0               # a hold with no replan pending lasts this long
    paused_idle_s: float = 5.0        # a pause with nothing happening becomes a "paused_idle" event
    max_llm_requests: int = 12        # per episode; past this the episode is given up
    loop_repeats: int = 3
    max_step_failures: int = 3
    max_plan_rejections: int = 3
    arm: str | None = None            # the arm to drive; None = the manifest's first (H2 widens this)


@dataclass
class EpisodeResult:
    outcome: str = "running"          # done | stopped | gave_up | timeout
    wall_s: float = 0.0
    decisions: list = field(default_factory=list)
    llm: list = field(default_factory=list)
    events: Counter = field(default_factory=Counter)
    steps_started: int = 0
    steps_done: int = 0
    steps_failed: int = 0
    loops: int = 0
    stale_plans: int = 0
    failed_plans: int = 0
    log: list = field(default_factory=list)


class Brain:
    def __init__(self, server: RobotServer, decider, planner, user: UserChannel | None = None, cfg: BrainConfig | None = None):
        self.server, self.decider, self.user = server, decider, user
        self.cfg = cfg or BrainConfig()
        self.m = server.manifest()
        problems = catalog.check_manifest_skills(self.m.skills)
        if problems:        # a standard name with another shape would be planned, prompted and recorded wrongly
            raise ValueError(f"{self.m.robot}: manifest refused: " + "; ".join(problems))
        self.arm = self.cfg.arm or self.m.arms[0].id
        self.skills = {s.name: s for s in arm_skills(self.m, self.arm)}
        # the brain owns the schema: whatever planner client is passed, its requests are rebuilt from this manifest
        self.planner = ManifestPlanner(planner.backends, self.m, self.arm, getattr(planner, "retry", True))
        self.geom = geometry(next(a for a in self.m.arms if a.id == self.arm), self.cfg.heartbeat_s)
        self.detector = ChangeDetector()
        self.loops = LoopDetector(self.cfg.loop_repeats)
        self.r = EpisodeResult()
        self.task, self.user_messages = "", []
        self.plan, self.queue, self.status = None, [], {}
        self.done_at, self.tries, self.done_answers = {}, Counter(), {}
        self.cur, self.skill_id, self.last_step = None, None, None
        self.mode, self.pause_reason, self.hold_until = "waiting", None, None
        self.awaiting = None              # event id whose decision gates the next step
        self.ev_id, self.new_events, self.in_flight = 0, [], 0
        self.replan_seq, self.replan_pending, self.llm_requests = 0, False, 0
        self.plan_rejections, self.trust_next_plan, self.unmet_raised, self.misfits = 0, False, False, 0
        self.last_event_t, self.touched = 0.0, set()
        self.world = self.state = None
        self.names_: dict = {}
        self._loop = self._q = None
        self._closed = self._stopped = False
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="brain")
        self._t0 = time.monotonic()

    # ---------------------------------------------------------------- inputs (any thread)
    def say(self, text: str):
        """Typed text. A bare "stop" is STOP, handled here in code before anything queues."""
        if is_stop(text):
            self._hard_stop(f'the user typed "{text}"')
        else:
            self._post("user", text)

    def press_stop(self):
        self._hard_stop("the STOP button")

    def _hard_stop(self, why: str):
        self._stopped = True
        self.server.stop()
        self._post("stop", why)

    def _post(self, kind: str, payload=None):
        if self._closed or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._q.put_nowait, (kind, payload))
        except RuntimeError:              # the loop closed under us: the episode is over
            pass

    def _submit(self, kind: str, fn, *args):
        """Run a slow call (a model) in a thread; its result comes back as a queue item."""
        def work():
            try:
                out = fn(*args)
            except BaseException as e:    # a decider or planner bug fails the episode loudly, not as a hang
                self._post("error", e)
                return
            self._post(kind, out)
        self._pool.submit(work)

    # ---------------------------------------------------------------- the episode
    async def run(self, task: str) -> EpisodeResult:
        self._loop, self._q = asyncio.get_running_loop(), asyncio.Queue()
        self._t0, self.task = time.monotonic(), task
        self.server.subscribe(lambda ev: self._post("robot", ev))
        if self.user is not None:
            self.user.attach(self.say, self.press_stop)
        self._observe()
        self.detector.update(self.state)
        self._request_plan()
        hb = asyncio.create_task(self._heartbeat())
        try:
            while self.mode not in END and not self._stopped:
                try:
                    kind, payload = await asyncio.wait_for(self._q.get(), max(0.0, self.cfg.max_s - self.now()))
                except TimeoutError:
                    self.mode = "timeout"
                    break
                self._handle(kind, payload)
                if self.mode in END or self._stopped:
                    break
                self._advance()
                self._dispatch()
        finally:
            hb.cancel()
            self._closed = True
            self._pool.shutdown(wait=False, cancel_futures=True)
            if not self._stopped and self.mode not in ("done",):
                self.server.hold(self.arm)      # timeout, given up or a bug: leave the arm still, grip kept
        self.r.outcome = "stopped" if self._stopped else self.mode
        self.r.wall_s = self.now()
        self._note("end", outcome=self.r.outcome)
        return self.r

    async def _heartbeat(self):
        while True:
            self.server.heartbeat()
            self._q.put_nowait(("tick", None))
            await asyncio.sleep(self.cfg.heartbeat_s)

    def _handle(self, kind: str, payload):
        if kind == "stop":
            self.mode = "stopped"
            self._note("stop", why=payload)
        elif kind == "user":
            self._user_text(payload)
        elif kind == "robot":
            self._robot(payload)
        elif kind == "tick":
            self._observe_changes()
            self._timers()
        elif kind == "decision":
            self._apply_decision(payload)
        elif kind == "plan":
            self._apply_plan(payload)
        elif kind == "error":
            raise payload

    def _advance(self):
        if self.mode == "running" and self.skill_id is None and self.awaiting is None and self.plan is not None:
            self._start_next()

    def _dispatch(self):
        evs, self.new_events = self.new_events, []
        for ev in evs:
            if ev.kind == "plan_arrived" and self.trust_next_plan:
                self._note("plan_trusted", why="replan loop: running the new plan without routing it back")
                continue
            self.in_flight += 1
            self._submit("decision", self._decide, ev, self._view(ev))

    def _decide(self, ev: Event, view: View) -> dict:
        out = self.decider.decide(ev, view)
        out["event"] = ev
        return out

    # ---------------------------------------------------------------- helpers
    def now(self) -> float:
        return time.monotonic() - self._t0

    def _note(self, kind: str, **rec):
        self.r.log.append({"t": round(self.now(), 3), "kind": kind, **rec})

    def name(self, oid) -> str:
        return self.names_.get(oid, oid)

    def _observe(self):
        self.world = self.server.world()
        self.state = core_state(self.world, self.arm, self.geom)
        self.names_ = world_names(self.world)

    def _why_not(self, s: dict) -> str | None:
        sk = self.skills.get(s["skill"])
        if sk is None:
            return f"{self.m.robot} has no skill {s['skill']}"
        return self.server.precondition(self.arm, s["skill"], step_args(s, sk))

    def path_to(self) -> tuple | None:
        if self.skill_id is None or self.mode == "paused":
            return None
        return xy_cm(self.server.status(self.skill_id).heading)

    def _arm_to(self, mode: str):
        """The brain's mode and the server call that makes the arm match. Models reach the arm only
        through here and retarget(): hold, pause, resume. Never stop()."""
        before, self.mode = self.mode, mode
        if self._stopped or self.skill_id is None or before == mode:
            return
        {"holding": self.server.hold, "paused": self.server.pause, "running": self.server.resume}[mode](self.arm)

    def _drop_skill(self):
        """The protocol has no cancel: hold the arm so the dropped skill stops at a safe point; the next
        start() replaces it on the server."""
        if self.skill_id is not None and not self._stopped:
            self.server.hold(self.arm)
        self.skill_id, self.cur = None, None

    def _view(self, ev: Event | None = None) -> View:
        st, cur = self.state, self.cur
        if cur is None and ev is not None and ev.step_id and self.plan:
            cur = P.steps_by_id(self.plan).get(ev.step_id)
        if self.skill_id is not None:
            phase = self.server.status(self.skill_id).phase_text
        elif self.last_step is not None:
            held = f"holding {self.name(st.arm.holding)}" if st.arm.holding else "the gripper is empty"
            phase = (f"between steps: {P.step_text(self.last_step, self.names_)} just "
                     f"{'finished' if self.status.get(self.last_step['id']) == 'done' else 'stopped'}; {held}")
        else:
            phase = "not started on any step yet" if self.plan else "waiting for the first plan"
        path, other = self.path_to(), self._runnable_other(ev)
        return View(t=self.now(), task=self.task, user_messages=list(self.user_messages), state=st, plan=self.plan,
                    queue=list(self.queue), status=dict(self.status), current=cur, phase=phase, mode=self.mode,
                    pause_reason=self.pause_reason, replan_pending=self.replan_pending, names=dict(self.names_),
                    path_to=path, hand=hand_facts(st, path),
                    runnable_other=P.step_text(P.steps_by_id(self.plan)[other], self.names_) if other else None,
                    tries=dict(self.tries), grasped=bool(cur and st.arm.holding and st.arm.holding == cur.get("object")))

    def _runnable_other(self, ev: Event | None) -> str | None:
        """A pending step other than the current one whose precondition holds now (its id)."""
        if not self.plan:
            return None
        byid = P.steps_by_id(self.plan)
        skip = ({ev.step_id} if ev and ev.step_id else set()) | ({self.cur["id"]} if self.cur else set())
        return next((sid for sid in self.queue if sid not in skip and self.status.get(sid) == "pending"
                     and self._why_not(byid[sid]) is None), None)

    # ---------------------------------------------------------------- events
    def event(self, kind: str, text: str, obj=None, step_id=None, **data) -> Event:
        self.ev_id += 1
        ev = Event(self.ev_id, self.now(), kind, text, obj, step_id, data)
        self.new_events.append(ev)
        self.r.events[kind] += 1
        self.last_event_t = self.now()
        self._note("event", event=kind, text=text)
        return ev

    def _user_text(self, text: str):
        self.user_messages.append(text)
        self.plan_rejections, self.trust_next_plan = 0, False      # the user spoke: judge new plans again
        if self.mode == "paused" and (self.pause_reason or "").startswith("waiting for the user's answer"):
            self._arm_to("holding")                                  # the answer ends that pause; hold while it is decided
            self.pause_reason, self.hold_until = None, self.now() + self.cfg.hold_s
        self.event("user_text", f'the user typed: "{text}"', text=text)

    def _robot(self, ev: RobotEvent):
        if ev.kind in ("skill_done", "skill_failed"):
            if ev.skill_id is None or ev.skill_id != self.skill_id:
                self._note("skill_event_ignored", event=ev.kind, skill_id=ev.skill_id)   # a replaced or dropped skill
                return
            self._observe()
            self._step_ended(ev.kind == "skill_done", ev.data.get("reason") or ev.text)
        elif ev.kind == "scene_change":
            self._observe_changes()
        elif ev.kind == "safety_trip":
            # the safety filter outranks every answer: hold in code now, then decide what it means
            if not self._stopped:
                self.server.hold(self.arm)
            self._arm_to("paused")
            self.pause_reason = "the robot's safety filter tripped"
            self.event("scene_change", f"the robot's safety filter tripped ({ev.text}); the arm stopped where it was")
        elif ev.kind == "heartbeat_lost":
            self._note("heartbeat_lost", text=ev.text)
            if self.mode == "running":       # the server held the arm; carry on after a normal hold
                self._arm_to("holding")
                self.hold_until = self.now() + self.cfg.hold_s

    def _observe_changes(self):
        """Changes the robot didn't cause (e12v2 core.changes) from a fresh world, as scene_change events.
        The server's scene_change is a wake-up; the 200 ms tick also catches hand transitions (closer,
        into the arm's path) that only the brain can judge, since only it knows where the step heads."""
        self._observe()
        touched, self.touched = self.touched, set()
        for ch in self.detector.update(self.state, self.path_to(), touched):
            self._scene_event(ch)

    def _where(self, where: str) -> str:
        if where.startswith("on:"):
            return f"on top of {self.name(where[3:])}"
        return {"table": "on the table", "gripper": "in the gripper", "person": "held by the person"}.get(where, f"in {self.name(where)}")

    def _scene_event(self, ch: dict):
        w = ch["what"]
        if w == "hand":
            now, before = ch["now"], ch["before"]
            if now is None:
                txt = "the person's hand has gone out of view"
            else:
                band, appr, path, out = now
                bits = [_BAND[band] + " the gripper", "coming closer" if appr else "not coming closer",
                        "in the arm's path" if path else "not in the arm's path"]
                if out:
                    bits.append("held out still, palm up, as if to take something")
                txt = ("a person's hand appeared: " if before is None else "the person's hand is now ") + ", ".join(bits)
            self.event("scene_change", txt, None, change={"what": "hand", "now": now, "before": before})
            return
        o = ch["object"]
        if w == "object_moved":
            fr, to = self._where(ch["from"]), self._where(ch["to"])
            txt = (f"{self.name(o)} was moved by someone else (not the robot): " +
                   (f"it was {fr}, now it is {to}" if ch["from"] != ch["to"] else f"still {to}, about {ch['moved_cm']} cm from where it was"))
        elif w == "object_gone":
            txt = f"{self.name(o)} is no longer in view (it was {self._where(ch['from'])})"
        else:
            txt = f"{self.name(o)} appeared, {self._where(ch['to'])}"
        self.event("scene_change", txt, o, change=ch)

    def _timers(self):
        t = self.now()
        if self.mode == "holding" and not self.replan_pending and self.hold_until is not None and t >= self.hold_until:
            self.hold_until = None
            self._arm_to("running")
        if self.mode == "paused" and t - self.last_event_t >= self.cfg.paused_idle_s and not self.in_flight:
            self.event("paused_idle", f"the arm has been paused for {self.cfg.paused_idle_s:.0f} s ({self.pause_reason}) and nothing has changed")

    # ---------------------------------------------------------------- steps
    def _obj(self, s: dict):
        return s["object"] if s["object"] != P.NONE else None

    def _start_next(self):
        if self._stopped:
            return
        nxt = next((sid for sid in self.queue if self.status.get(sid) == "pending"), None)
        if nxt is None:
            self._check_done()
            return
        s = P.steps_by_id(self.plan)[nxt]
        why = self._why_not(s)
        if why:
            self.queue.remove(nxt)
            self.status[nxt], self.last_step = "failed", s
            self.tries[nxt] += 1
            self.r.steps_failed += 1
            ev = self.event("step_failed", f"step could not start: {P.step_text(s, self.names_)}: {why}", self._obj(s), nxt,
                            reason=f"could not start: {why}")
            self.awaiting = ev.id
            self._loop_check(s, nxt)
            return
        self.skill_id = self.server.start(self.arm, s["skill"], step_args(s, self.skills[s["skill"]]))
        self.cur, self.status[nxt] = s, "running"
        self.r.steps_started += 1
        self._note("step_start", step=P.step_text(s, self.names_), skill_id=self.skill_id)
        self._loop_check(s, nxt)

    def _loop_check(self, s: dict, sid: str):
        if self.loops.record((s["skill"], s["object"], s["target"], s["direction"])) or self.tries[sid] >= self.cfg.max_step_failures:
            self.r.loops += 1
            self.tries[sid] = 0
            self._note("loop", step=P.step_text(s, self.names_))
            self._arm_to("holding")
            if not self.replan_pending:
                ev = Event(0, self.now(), "loop", f"loop: {P.step_text(s, self.names_)} keeps coming back", self._obj(s), sid)
                self._start_replan("capable_llm", f"a loop: the plan keeps returning to {P.step_text(s, self.names_)} without progress", ev)

    def _step_ended(self, ok: bool, reason: str | None):
        s = self.cur
        sid = s["id"]
        self.skill_id, self.cur, self.last_step = None, None, s
        if self._obj(s):
            self.touched.add(s["object"])      # the robot's own doing: re-anchor it in the change detector
        if sid in self.queue:
            self.queue.remove(sid)
        txt = P.step_text(s, self.names_)
        if ok:
            self.status[sid], self.done_at[sid] = "done", self.now()
            self.r.steps_done += 1
            g = P.step_goal(s)
            after = f"; {self.name(g[0])} is now {self._where(self.state.objects[g[0]].where)}" if g and g[0] in self.state.objects else ""
            ev = self.event("step_done", f"step finished: {txt}{after}", self._obj(s), sid)
        else:
            self.status[sid] = "failed"
            self.tries[sid] += 1
            self.r.steps_failed += 1
            ev = self.event("step_failed", f"step failed: {txt}: {reason}", self._obj(s), sid, reason=reason)
        self._note("step_end", step=txt, ok=ok, reason=reason)
        self.awaiting = ev.id

    def _check_done(self):
        if self.replan_pending:
            return              # a plan is on its way: not done until it arrives and says so
        self._observe()
        unmet = []
        for c in self.plan["done_when"]:
            v = P.measure(c, self.state)
            if not (self.done_answers.get(c["id"], True) if v is None else v):   # unmeasurable: Jev's answer on step_done
                unmet.append(c)
        if not unmet:
            self.mode = "done"
            self._note("done")
        elif not self.unmet_raised:
            self.unmet_raised = True
            obj = unmet[0]["object"] if unmet[0]["object"] in self.state.objects else None
            ev = self.event("plan_done_unmet", "every planned step has run, but these done conditions are false: " +
                            "; ".join(c["text"] for c in unmet), obj, unmet=[c["text"] for c in unmet])
            self.awaiting = ev.id

    # ---------------------------------------------------------------- decisions
    def _apply_decision(self, out: dict):
        ev, c = out["event"], out["combined"]
        self.in_flight -= 1
        self.r.decisions.append({"t": self.now(), "t_event": ev.t, "event": ev.kind, "text": ev.text, "combined": c,
                                 "latency_ms": out.get("latency_ms", 0.0)})
        self._note("decision", event=ev.kind, right_now=c.right_now, fix=c.fix, route=c.route, reason=c.reason)
        if ev.id == self.awaiting:
            self.awaiting = None
        if self.mode in END:
            return
        self.done_answers.update(c.done)
        self._right_now(c.right_now, ev)
        if c.route == "stay_local":
            self._fix(c.fix, ev)
        elif c.route in ("fast_llm", "capable_llm"):
            self._start_replan(c.route, ev.text + ("; plan check: " + "; ".join(c.problems) if c.problems else ""), ev)
        elif c.route == "ask_user":
            self._arm_to("paused")
            self.pause_reason = "waiting for the user's answer to a question"
            if self.user is not None:
                self.user.ask(f"About \"{ev.data.get('text', ev.text)}\": what should the end result be?")
        if self.mode == "holding" and not self.replan_pending:
            self.hold_until = self.now() + self.cfg.hold_s

    def _right_now(self, rn: str, ev: Event):
        before = self.mode
        if self.mode == "waiting" and rn not in ("pause", "back_off"):
            return
        if rn == "hold" and self.mode in ("running", "waiting"):
            self._arm_to("holding")
        elif rn in ("pause", "back_off"):
            # the protocol has no back-off motion: back_off pauses (the server's own stop distance still applies)
            if self.mode != "paused":
                self._arm_to("paused")
                self.pause_reason = (f'the user said "{ev.data.get("text", "")}"' if ev.kind == "user_text" else
                                     "a person's hand came near" if "hand" in ev.text else f"paused after: {ev.text}")
        elif rn == "re_target":
            if self.skill_id is not None and not self._stopped:
                self.server.retarget(self.skill_id)
        elif rn == "resume" and self.mode in ("paused", "holding"):
            self.hold_until = None
            self._arm_to("running")
        if self.mode != before:
            self._note("mode", before=before, after=self.mode, because=rn)

    def _fix(self, fix: str, ev: Event):
        if fix == "none" or not self.plan:
            return
        at = 1 if self.cur is not None else 0
        if fix == "retry":
            if ev.step_id and self.status.get(ev.step_id) == "failed":
                self.status[ev.step_id] = "pending"
                self.queue.insert(at, ev.step_id)
        elif fix == "skip":
            sid = self.cur["id"] if self.cur is not None else ev.step_id
            if self.cur is not None and sid == self.cur["id"]:
                if self.state.arm.holding:
                    return          # never drop a step with its object in the gripper
                self._drop_skill()
                if sid in self.queue:
                    self.queue.remove(sid)
            if sid:
                self.status[sid] = "skipped"
        elif fix == "re_queue":
            cands = [s for s in self.plan["steps"] if s["object"] == ev.object and P.step_goal(s)
                     and self.status.get(s["id"]) in ("done", "failed", "skipped")]
            if cands:
                sid = cands[-1]["id"]
                self.status[sid] = "pending"
                if sid in self.queue:
                    self.queue.remove(sid)
                self.queue.insert(at, sid)
                self.unmet_raised = False
        elif fix == "another_step_first":
            other = self._runnable_other(ev)
            if other:
                self.queue.remove(other)
                self.queue.insert(at, other)
                if ev.step_id and self.status.get(ev.step_id) == "failed":
                    self.status[ev.step_id] = "pending"
                    self.queue.insert(at + 1, ev.step_id)
        self._note("fix", fix=fix, queue=list(self.queue))

    # ---------------------------------------------------------------- the planner
    def _ref(self, ev):
        # e12v2's mock LLM reads the plan it is diffing against from here; a snapshot, since it runs in a thread
        return {"episode": SimpleNamespace(plan=copy.deepcopy(self.plan), queue=list(self.queue)), "event": ev}

    def _planned(self, seq: int, t_request: float, ev, call):
        res = call()
        res.seq, res.t_request, res.event = seq, t_request, ev
        return res

    def _new_request(self) -> int:
        self.replan_seq += 1              # the newest request wins; older results are stale on arrival
        self.replan_pending = True
        self.llm_requests += 1
        return self.replan_seq

    def _request_plan(self):
        args = (self.task, list(self.user_messages), self.world, self.state)
        seq = self._new_request()
        self._note("llm_request", kind="plan", route="fast_llm")
        self._submit("plan", self._planned, seq, self.now(), None, lambda: self.planner.plan(*args, ref=self._ref(None)))

    def _start_replan(self, route: str, why: str, ev: Event | None):
        if self.llm_requests >= self.cfg.max_llm_requests:
            self.mode = "gave_up"
            self._note("gave_up", why=f"{self.cfg.max_llm_requests} LLM requests")
            return
        if not self.plan:
            return
        if ev is not None and ev.kind == "plan_arrived":
            self.plan_rejections += 1
            if self.plan_rejections >= self.cfg.max_plan_rejections and not self.trust_next_plan:
                # a replan loop: from here until the user says something new, new plans run as they come
                self.trust_next_plan = True
                self.r.loops += 1
                self._note("loop", loop="replans", rejections=self.plan_rejections)
        arm_now = {"current_step": self.cur["id"] if self.cur else "none",
                   "doing": self.server.status(self.skill_id).phase_text if self.skill_id else "between steps",
                   "while_you_plan": {"holding": "holding still", "paused": "paused"}.get(self.mode, "carrying on")}
        args = (route, self.task, list(self.user_messages), self.world, self.state, copy.deepcopy(self.plan), list(self.queue),
                dict(self.status), arm_now, why)
        ref = self._ref(ev)
        seq = self._new_request()
        self._note("llm_request", kind="replan", route=route, why=why)
        self._submit("plan", self._planned, seq, self.now(), ev, lambda: self.planner.replan(*args, ref=ref))

    def _apply_plan(self, res):
        self.r.llm.append({"t_request": res.t_request, "t": self.now(), "kind": res.kind, "route": res.route, "ok": res.ok,
                           "attempts": res.attempts, "event": res.event.kind if res.event else None})
        if res.seq != self.replan_seq:
            self.r.stale_plans += 1
            self._note("plan_stale", seq=res.seq)
            return
        self.replan_pending = False
        self._observe()
        if not res.ok:
            self.r.failed_plans += 1
            self._note("plan_failed", kind=res.kind, errors=[a["errors"] for a in res.attempts])
            if res.kind == "plan":
                if self.r.failed_plans >= 2:
                    self.mode = "gave_up"
                else:
                    self._request_plan()
            else:
                self.unmet_raised = False
                if self.mode == "holding":
                    self._arm_to("running")
            return
        if res.kind == "plan":
            self.plan = res.plan
            self.queue = [s["id"] for s in self.plan["steps"]]
            self.status = {sid: "pending" for sid in self.queue}
            if self.mode == "waiting":
                self.mode = "running"
            self._note("plan_applied", plan=self.plan)
            self._advance()
            self.event("plan_arrived", "the first plan arrived: " + self.plan.get("reading", ""))
            return
        newp, q = P.apply_diff(self.plan, res.diff)
        q = [sid for sid in q if not (self.status.get(sid) == "done" and self.done_at.get(sid, -1) >= res.t_request)]
        held, byid = self.state.arm.holding, P.steps_by_id(newp)
        if held and self.cur is not None and self.cur.get("object") == held and (not q or byid[q[0]]["object"] != held):
            # never drop a step with its object in the gripper: the plan was written before the grasp
            if self.cur["id"] in q:
                q.remove(self.cur["id"])
            q.insert(0, self.cur["id"])
            self._note("kept_running_step", step=self.cur["id"])
        errs = P.validate(newp, self.state, held, [sid for sid in q if self.status.get(sid) != "done" or sid in res.diff["pending_order"]])
        if errs and self.misfits < 2:
            # the world moved on while the LLM worked: ask again from where things stand now
            self.misfits += 1
            self._note("plan_misfit", errors=errs)
            txt = "the new plan no longer fits the world: " + "; ".join(errs[:3])
            self._start_replan(res.route, txt, Event(0, self.now(), "misfit", txt))
            return
        self.misfits = 0
        if self.cur is not None and (not q or q[0] != self.cur["id"]):
            sid = self.cur["id"]
            self._drop_skill()
            self.status[sid] = "pending" if sid in q else "dropped"
        for sid in list(self.status):
            if self.status[sid] == "pending" and sid not in q:
                self.status[sid] = "dropped"
        for sid in q:
            if self.status.get(sid) != "running":
                self.status[sid] = "pending"
        self.plan, self.queue, self.unmet_raised = newp, q, False
        if self.mode == "holding":
            self.hold_until = None
            self._arm_to("running")
        self._note("plan_applied", diff=res.diff, queue=q)
        self.event("plan_arrived", "a new plan arrived: " + (res.diff.get("reading") or ""))
