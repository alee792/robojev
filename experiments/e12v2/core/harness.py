"""The harness: one episode, closed loop (docs/v2.md "An episode", v2-diagrams.md 2 and 7).

    plan = fast_llm(task, world)              # the one planned wait
    start plan step 1                          # the plan check runs alongside
    decide("new plan")
    until done:
        run the current plan step              # control loop, safety filter on every command
        on any event: decide -> apply right-now at once; in-plan fix if stay_local and confident;
                      else LLM replans while the arm carries on or holds, per right-now; decide("new plan")

Time is simulation ticks. A decision or LLM call is made when its event happens and its result is
applied `latency` later (measured latency for live backends, sampled for mocks), so the world keeps
moving while models think. Several decisions can be in flight; replans supersede each other (the
newest request wins). STOP (button or typed "stop") is code and ends the episode at once.

The harness imports nothing from sim/ or eval/: the world, the user, the skills, the decider and the
planner are passed in; `observer` (optional) gets read-only hooks for evaluation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from . import plan as P
from .changes import ChangeDetector, hand_facts
from .data import Combined, Command, Event, WorldState, dxy
from .decision import View
from .planner import PlannerClient
from .safety import SafetyFilter

STOP_RE = re.compile(r"^\s*(stop|stop!|stop now|stop it|stop stop)\W*$", re.I)


def is_stop(text: str) -> bool:
    """Typed "stop" is handled in code (docs/v2.md). Only a bare stop: "don't stop" or "stop at the tray" are text."""
    return bool(STOP_RE.match(text))


@dataclass
class HarnessConfig:
    max_s: float = 240.0
    hold_s: float = 1.0            # a hold with no replan pending lasts this long
    paused_idle_s: float = 5.0     # a pause with nothing happening becomes a "paused_idle" event
    max_llm_calls: int = 16        # per episode; past this the episode is given up
    loop_repeats: int = 3          # A,B alternating this many times = a loop
    max_step_failures: int = 3     # the same step failing this many times = a loop
    decide_plan_arrived: bool = True


class LoopDetector:
    """A loop is a failure: the same two plan steps alternating more than twice escalates."""

    def __init__(self, repeats: int = 3):
        self.repeats, self.hist = repeats, []

    def record(self, sig) -> bool:
        self.hist.append(sig)
        n = 2 * self.repeats
        h = self.hist[-n:]
        if len(h) == n and h[0] != h[1] and all(h[i] == h[i % 2] for i in range(n)):
            self.hist.clear()
            return True
        return False


@dataclass
class EpisodeResult:
    outcome: str = "running"          # done | stopped | gave_up | timeout
    ticks: int = 0
    done_t: int | None = None
    decisions: list = field(default_factory=list)
    llm: list = field(default_factory=list)     # one record per LLM request (with its attempts)
    mode_ticks: Counter = field(default_factory=Counter)
    steps_started: int = 0
    steps_done: int = 0
    steps_failed: int = 0
    loops: int = 0
    refusals: int = 0
    floor_stops: int = 0
    events: Counter = field(default_factory=Counter)
    stale_plans: int = 0
    failed_plans: int = 0


class Episode:
    def __init__(self, world, user, skills: dict, decider, planner: PlannerClient, cfg: HarnessConfig | None = None,
                 log=None, observer=None):
        self.world, self.user, self.skills, self.decider, self.planner = world, user, skills, decider, planner
        self.cfg = cfg or HarnessConfig()
        self.g = world.geometry
        self.log, self.obs = log, observer
        self.safety = SafetyFilter()
        self.detector = ChangeDetector()
        self.loops = LoopDetector(self.cfg.loop_repeats)
        self.r = EpisodeResult()
        self.t = 0
        self.task = ""
        self.user_messages: list[str] = []
        self.plan: dict | None = None
        self.queue: list[str] = []
        self.status: dict[str, str] = {}
        self.done_at: dict[str, int] = {}
        self.tries: Counter = Counter()
        self.skill = None
        self.cur: dict | None = None            # the running step
        self.last_step: dict | None = None      # the step that last finished or failed
        self.mode = "waiting"
        self.pause_reason: str | None = None
        self.backoff_to: tuple | None = None
        self.hold_until: int | None = None
        self.awaiting: int | None = None        # event id whose decision gates the next step
        self.pending: list = []                 # (due, seq, kind, payload)
        self.seq = 0
        self.ev_id = 0
        self.new_events: list[Event] = []
        self.replan_seq = 0                     # the newest replan request; older ones are stale
        self.replan_pending = False
        self.unmet_raised = False
        self.done_answers: dict = {}
        self.last_event_t = 0
        self.touched: set = set()
        self.state: WorldState | None = None
        self.misfits = 0

    # ---------------------------------------------------------------- helpers
    def ticks(self, ms: float) -> int:
        return max(1, math.ceil(ms / (1000.0 / self.g.ticks_per_s) - 1e-9))

    def names(self) -> dict:
        return self.world.names()

    def name(self, oid) -> str:
        return self.names().get(oid, oid)

    def _log(self, kind, **rec):
        if self.log:
            self.log(kind=kind, t=self.t, **rec)

    def path_to(self) -> tuple | None:
        return self.skill.heading(self.state) if self.skill is not None and self.mode != "paused" else None

    def view(self, ev: Event | None = None) -> View:
        st = self.state
        cur = self.cur
        if cur is None and ev is not None and ev.step_id and self.plan:
            cur = P.steps_by_id(self.plan).get(ev.step_id)
        if self.skill is not None:
            phase = self.skill.phase_text(st)
        elif self.last_step is not None:
            held = f"holding {self.name(st.arm.holding)}" if st.arm.holding else "the gripper is empty"
            phase = (f"between steps: {P.step_text(self.last_step, self.names())} just "
                     f"{'finished' if self.status.get(self.last_step['id']) == 'done' else 'stopped'}; {held}")
        else:
            phase = "not started on any step yet" if self.plan else "waiting for the first plan"
        return View(t=self.t, task=self.task, user_messages=list(self.user_messages), state=st, plan=self.plan,
                    queue=list(self.queue), status=dict(self.status), current=cur, phase=phase, mode=self.mode,
                    pause_reason=self.pause_reason, replan_pending=self.replan_pending, names=self.names(),
                    path_to=self.path_to(), hand=hand_facts(st, self.path_to()), runnable_other=self._runnable_other(ev),
                    tries=dict(self.tries), grasped=bool(cur and st.arm.holding and st.arm.holding == cur.get("object")))

    def _runnable_other(self, ev: Event | None) -> str | None:
        if not self.plan:
            return None
        byid = P.steps_by_id(self.plan)
        skip = {ev.step_id} if ev and ev.step_id else set()
        if self.cur:
            skip.add(self.cur["id"])
        for sid in self.queue:
            if sid in skip or self.status.get(sid) != "pending":
                continue
            s = byid[sid]
            if self.skills[s["skill"]].precondition(self.state, s) is None:
                return P.step_text(s, self.names())
        return None

    def _runnable_other_id(self, ev: Event) -> str | None:
        txt = self._runnable_other(ev)
        if txt is None:
            return None
        byid = P.steps_by_id(self.plan)
        return next(sid for sid in self.queue if P.step_text(byid[sid], self.names()) == txt and sid != ev.step_id)

    # ---------------------------------------------------------------- events
    def event(self, kind: str, desc: str, obj=None, step_id=None, **data) -> Event:
        self.ev_id += 1
        ev = Event(self.ev_id, self.t, kind, desc, obj, step_id, data)
        self.new_events.append(ev)
        self.r.events[kind] += 1
        self.last_event_t = self.t
        self._log("event", event=ev.__dict__)
        if self.obs:
            self.obs.on_event(self, ev)
        return ev

    def _scene_event(self, ch: dict):
        n = self.name
        w = ch["what"]
        if w == "object_moved":
            o = ch["object"]
            fr, to = self._where(ch["from"]), self._where(ch["to"])
            if ch["from"] != ch["to"]:
                txt = f"{n(o)} was moved by someone else (not the robot): it was {fr}, now it is {to}"
            else:
                txt = f"{n(o)} was moved by someone else (not the robot): still {to}, about {ch['moved_cm']} cm from where it was"
            self.event("scene_change", txt, o, change=ch)
        elif w == "object_gone":
            self.event("scene_change", f"{n(ch['object'])} is no longer in view (it was {self._where(ch['from'])})", ch["object"], change=ch)
        elif w == "object_appeared":
            self.event("scene_change", f"{n(ch['object'])} appeared, {self._where(ch['to'])}", ch["object"], change=ch)
        elif w == "hand":
            now, before = ch["now"], ch["before"]
            if now is None:
                txt = "the person's hand has gone out of view"
            else:
                band, appr, path, out = now
                bits = [{"very close": "very close to", "near": "near", "mid-range": "mid-range from", "far": "far from"}[band] + " the gripper",
                        "coming closer" if appr else "not coming closer",
                        "in the arm's path" if path else "not in the arm's path"]
                if out:
                    bits.append("held out still, palm up, as if to take something")
                txt = ("a person's hand appeared: " if before is None else "the person's hand is now ") + ", ".join(bits)
            self.event("scene_change", txt, None, change={"what": "hand", "now": now, "before": before})

    def _where(self, where: str) -> str:
        if where == "table":
            return "on the table"
        if where == "gripper":
            return "in the gripper"
        if where == "person":
            return "held by the person"
        if where.startswith("on:"):
            return f"on top of {self.name(where[3:])}"
        return f"in {self.name(where)}"

    # ---------------------------------------------------------------- the episode
    def run(self, task: str) -> EpisodeResult:
        self.task = task
        self.state = self.world.state()
        self.detector.update(self.state)
        self._request_plan()
        max_t = int(self.cfg.max_s * self.g.ticks_per_s)
        for t in range(max_t):
            self.t = t
            self.world.advance(t)
            if self.user.stop_pressed(t):
                self._stop("the STOP button")
                break
            for text in self.user.poll(t):
                self.user_messages.append(text)
                if is_stop(text):
                    self._stop('the user typed "stop"')
                    break
                if self.mode == "paused" and (self.pause_reason or "").startswith("waiting for the user's answer"):
                    # "paused until the user answers": the answer ends the pause; hold while it is decided
                    self.mode, self.pause_reason = "holding", None
                    self.hold_until = t + int(self.cfg.hold_s * self.g.ticks_per_s) + 3
                self.event("user_text", f'the user typed: "{text}"', text=text)
            if self.mode == "stopped":
                break
            self.state = self.world.state()
            for ch in self.detector.update(self.state, self.path_to(), self.touched):
                self._scene_event(ch)
            self.touched = set()
            for item in sorted([p for p in self.pending if p[0] <= t], key=lambda p: (p[0], p[1])):
                self.pending.remove(item)
                self._deliver(item)
                if self.mode in ("stopped", "gave_up"):
                    break
            if self.mode in ("stopped", "gave_up"):
                break
            self._arm_tick()
            self._timers()
            evs, self.new_events = self.new_events, []
            for ev in evs:
                self._decide(ev)
            self.r.mode_ticks[self.mode] += 1
            if self.obs:
                self.obs.on_tick(self)
            if self.mode in ("done", "gave_up"):
                break
        self.r.ticks = self.t + 1
        if self.r.outcome == "running":
            self.r.outcome = {"done": "done", "stopped": "stopped", "gave_up": "gave_up"}.get(self.mode, "timeout")
        self.r.refusals, self.r.floor_stops = self.safety.refusals, self.safety.floor_stops
        self._log("end", outcome=self.r.outcome, ticks=self.r.ticks)
        return self.r

    def _stop(self, why: str):
        self.mode = "stopped"
        self.skill = None
        self.r.outcome = "stopped"
        self._log("stop", why=why)

    # ---------------------------------------------------------------- the arm (control loop)
    def _exec(self, cmd: Command | None) -> tuple[bool, str | None]:
        st = self.state
        if cmd is None or cmd.kind == "wait":
            return True, None
        out, why = self.safety.clip(st, cmd, 2.0)
        if out is None:
            return False, why
        if out.kind == "release" and st.arm.holding:
            self.touched.add(st.arm.holding)
        if out.drag:
            self.touched.add(out.drag)
        self.world.execute(out)
        self.state = self.world.state()
        return True, None

    def _arm_tick(self):
        if self.mode == "running":
            if self.skill is None and self.awaiting is None and self.plan is not None:
                self._start_next()
            if self.skill is not None:
                cmd = self.skill.step(self.state)
                ok, why = self._exec(cmd)
                self.skill.feedback(ok, why)
                st, reason = self.skill.status()
                if st == "done":
                    self._step_ended(True, None)
                elif st == "failed":
                    self._step_ended(False, reason)
        elif self.mode == "holding" and self.skill is not None:
            self._exec(self.skill.safe_point(self.state))
        elif self.mode == "paused" and self.backoff_to is not None:
            a = self.state.arm
            if math.dist((a.x, a.y, a.z), self.backoff_to) < 0.5:
                self.backoff_to = None
            else:
                ok, _ = self._exec(Command("move", self.backoff_to, 2.0))
                if not ok:
                    self.backoff_to = None

    def _start_next(self):
        byid = P.steps_by_id(self.plan)
        nxt = next((sid for sid in self.queue if self.status.get(sid) == "pending"), None)
        if nxt is None:
            self._check_done()
            return
        s = byid[nxt]
        sk = self.skills[s["skill"]]
        why = sk.precondition(self.state, s)
        if why:
            self.queue.remove(nxt)
            self.status[nxt] = "failed"
            self.tries[nxt] += 1
            self.r.steps_failed += 1
            self.last_step = s
            ev = self.event("step_failed", f"step could not start: {P.step_text(s, self.names())}: {why}", s["object"] if s["object"] != P.NONE else None,
                            nxt, reason=f"could not start: {why}")
            self.awaiting = ev.id
            self._loop_check(s, nxt)
            return
        sk.start(self.state, s)
        self.skill, self.cur = sk, s
        self.status[nxt] = "running"
        self.r.steps_started += 1
        self._log("step_start", step=s)
        self._loop_check(s, nxt)

    def _loop_check(self, s: dict, sid: str):
        sig = (s["skill"], s["object"], s["target"], s["direction"])
        if self.loops.record(sig) or self.tries[sid] >= self.cfg.max_step_failures:
            self.r.loops += 1
            self.tries[sid] = 0
            self._log("loop", step=s)
            self.mode = "holding"
            if self.replan_pending:
                return
            ev = Event(0, self.t, "loop", f"loop: {P.step_text(s, self.names())} keeps coming back", s["object"], sid)
            self._start_replan("capable_llm", "replan", f"a loop: the plan keeps returning to {P.step_text(s, self.names())} without progress", ev)

    def _step_ended(self, ok: bool, reason: str | None):
        s = self.cur
        sid = s["id"]
        self.skill, self.cur, self.last_step = None, None, s
        if sid in self.queue:
            self.queue.remove(sid)
        if ok:
            self.status[sid] = "done"
            self.done_at[sid] = self.t
            self.r.steps_done += 1
            g = P.step_goal(s)
            after = f"; {self.name(g[0])} is now {self._where(self.state.objects[g[0]].where)}" if g and g[0] in self.state.objects else ""
            ev = self.event("step_done", f"step finished: {P.step_text(s, self.names())}{after}", s["object"] if s["object"] != P.NONE else None, sid)
        else:
            self.status[sid] = "failed"
            self.tries[sid] += 1
            self.r.steps_failed += 1
            ev = self.event("step_failed", f"step failed: {P.step_text(s, self.names())}: {reason}", s["object"] if s["object"] != P.NONE else None,
                            sid, reason=reason)
        self._log("step_end", step=s, ok=ok, reason=reason)
        self.awaiting = ev.id

    def _check_done(self):
        if self.replan_pending:
            return          # a plan is on its way: not done until it arrives and says so
        unmet = []
        for c in self.plan["done_when"]:
            v = P.measure(c, self.state)
            if v is None:
                v = self.done_answers.get(c["id"], True)   # asked of Jev on step_done; controls assume true
            if not v:
                unmet.append(c)
        if not unmet:
            self.mode = "done"
            self.r.done_t = self.t
            self._log("done")
            return
        if not self.unmet_raised:
            self.unmet_raised = True
            obj = unmet[0]["object"] if unmet[0]["object"] in self.state.objects else None
            ev = self.event("plan_done_unmet", "every planned step has run, but these done conditions are false: " +
                            "; ".join(c["text"] for c in unmet), obj, unmet=[c["text"] for c in unmet])
            self.awaiting = ev.id

    def _timers(self):
        if self.mode == "holding" and not self.replan_pending and self.hold_until is not None and self.t >= self.hold_until:
            self.mode, self.hold_until = "running", None
        if self.mode == "paused" and self.t - self.last_event_t >= self.cfg.paused_idle_s * self.g.ticks_per_s \
                and not any(p[2] == "decision" for p in self.pending):
            self.event("paused_idle", f"the arm has been paused for {self.cfg.paused_idle_s:.0f} s ({self.pause_reason}) and nothing has changed")

    # ---------------------------------------------------------------- decisions
    def _decide(self, ev: Event):
        if ev.kind == "plan_arrived" and not self.cfg.decide_plan_arrived:
            return
        v = self.view(ev)
        out = self.decider.decide(ev, v)
        out["event"], out["t"] = ev, self.t
        if self.obs:
            self.obs.on_decision(self, ev, out, v)
        self.seq += 1
        self.pending.append((self.t + self.ticks(out["latency_ms"]), self.seq, "decision", out))

    def _deliver(self, item):
        _, _, kind, payload = item
        if kind == "decision":
            self._apply_decision(payload)
        else:
            self._apply_plan(payload)

    def _apply_decision(self, out: dict):
        ev, c = out["event"], out["combined"]
        self.r.decisions.append({"t": self.t, "event": ev.kind, "combined": c, "latency_ms": out["latency_ms"],
                                 "in_tok": out.get("in_tok", 0), "jev": out.get("jev", False), "truth": out.get("truth")})
        self._log("decision", event=ev.__dict__, t_event=out["t"], latency_ms=out["latency_ms"], in_tok=out.get("in_tok", 0),
                  request=out.get("request"), answers=out.get("answers"), combined=c.__dict__, truth=out.get("truth"),
                  decider=self.decider.name)
        if ev.id == self.awaiting:
            self.awaiting = None
        if self.mode in ("done", "stopped"):
            return
        self.done_answers.update(c.done)
        if c.react:
            self._start_replan("fast_llm", "react", ev.text, ev)
            return
        self._right_now(c.right_now, ev)
        if c.route == "stay_local":
            self._fix(c.fix, ev)
        elif c.route in ("fast_llm", "capable_llm"):
            why = ev.text + ("" if not c.problems else "; plan check: " + "; ".join(c.problems))
            self._start_replan(c.route, "replan", why, ev)
        elif c.route == "ask_user":
            self.mode, self.pause_reason = "paused", "waiting for the user's answer to a question"
            self.user.ask(self.t, f"About \"{ev.data.get('text', ev.text)}\": what should the end result be?")
            self._log("ask_user", event=ev.text)
        if self.mode == "holding" and not self.replan_pending:
            self.hold_until = self.t + int(self.cfg.hold_s * self.g.ticks_per_s)

    def _right_now(self, rn: str, ev: Event):
        before = self.mode
        if self.mode == "waiting" and rn not in ("pause", "back_off"):
            return
        if rn == "hold" and self.mode in ("running", "waiting"):
            self.mode = "holding"
        elif rn == "pause":
            if self.mode != "paused":
                self.mode = "paused"
                self.pause_reason = ("the user asked the robot to wait" if ev.kind == "user_text" else
                                     "a person's hand came near" if "hand" in ev.text else f"paused after: {ev.text}")
        elif rn == "back_off":
            h, a = self.state.hand, self.state.arm
            if h is not None:
                vx, vy = a.x - h.x, a.y - h.y
                n = math.hypot(vx, vy) or 1.0
                x0, x1, y0, y1, _, z1 = self.g.workspace
                self.backoff_to = (min(x1, max(x0, a.x + 10 * vx / n)), min(y1, max(y0, a.y + 10 * vy / n)), min(z1, a.z + 5))
            self.mode, self.pause_reason = "paused", "backed off from a person's hand that was very close"
        elif rn == "re_target":
            if self.skill is not None:
                self.skill.retarget(self.state)
        elif rn == "resume":
            if self.mode in ("paused", "holding"):
                self.mode, self.backoff_to, self.hold_until = "running", None, None
        if self.mode != before:
            self._log("mode", before=before, after=self.mode, because=rn)

    def _fix(self, fix: str, ev: Event):
        if fix == "none" or not self.plan:
            return
        byid = P.steps_by_id(self.plan)
        at = 1 if self.cur is not None else 0
        if fix == "retry":
            sid = ev.step_id
            if sid and self.status.get(sid) == "failed":
                self.status[sid] = "pending"
                self.queue.insert(at, sid)
        elif fix == "skip":
            sid = self.cur["id"] if self.cur is not None else ev.step_id
            if sid and self.cur is not None and sid == self.cur["id"]:
                if self.state.arm.holding:
                    return   # never drop a step with its object in the gripper
                self.skill, self.cur = None, None
                if sid in self.queue:
                    self.queue.remove(sid)
            if sid:
                self.status[sid] = "skipped"
        elif fix == "re_queue":
            oid = ev.object
            cands = [s for s in self.plan["steps"] if s["object"] == oid and P.step_goal(s) and self.status.get(s["id"]) in ("done", "failed", "skipped")]
            if cands:
                sid = cands[-1]["id"]
                self.status[sid] = "pending"
                if sid in self.queue:
                    self.queue.remove(sid)
                self.queue.insert(at, sid)
                self.unmet_raised = False
        elif fix == "another_step_first":
            other = self._runnable_other_id(ev)
            if other:
                self.queue.remove(other)
                self.queue.insert(at, other)
                if ev.step_id and self.status.get(ev.step_id) == "failed":
                    self.status[ev.step_id] = "pending"
                    self.queue.insert(at + 1, ev.step_id)
        self._log("fix", fix=fix, queue=list(self.queue))
        _ = byid

    # ---------------------------------------------------------------- the planner
    def _llm_calls(self) -> int:
        return sum(len(x["attempts"]) for x in self.r.llm)

    def _request_plan(self):
        res = self.planner.plan(self.task, list(self.user_messages), self.state, self.names(), ref=self._ref(None))
        self._planned(res, None)

    def _ref(self, ev):
        return {"episode": self, "event": ev}

    def _start_replan(self, route: str, kind: str, why: str, ev: Event | None):
        if self._llm_calls() >= self.cfg.max_llm_calls:
            self.mode = "gave_up"
            self.r.outcome = "gave_up"
            self._log("gave_up", why=f"more than {self.cfg.max_llm_calls} LLM calls")
            return
        if not self.plan:
            return
        arm_now = {"current_step": self.cur["id"] if self.cur else "none",
                   "doing": self.skill.phase_text(self.state) if self.skill else "between steps",
                   "while_you_plan": "holding still" if self.mode == "holding" else ("paused" if self.mode == "paused" else "carrying on")}
        res = self.planner.replan(kind, route, self.task, list(self.user_messages), self.state, self.names(), self.plan,
                                  list(self.queue), dict(self.status), arm_now, why, ref=self._ref(ev))
        self._planned(res, ev)

    def _planned(self, res, ev):
        self.replan_seq += 1
        res.seq = self.replan_seq
        res.t_request = self.t
        self.replan_pending = True
        self.r.llm.append({"t": self.t, "kind": res.kind, "route": res.route, "ok": res.ok, "attempts": res.attempts,
                           "event": ev.kind if ev else None})
        self._log("llm", request=res.kind, route=res.route, ok=res.ok, attempts=res.attempts, why=ev.text if ev else self.task,
                  out=res.plan if res.kind == "plan" else res.diff)
        if self.obs:
            self.obs.on_llm(self, res, ev)
        self.seq += 1
        self.pending.append((self.t + self.ticks(res.latency_ms), self.seq, "plan", res))

    def _apply_plan(self, res):
        if res.seq != self.replan_seq:
            self.r.stale_plans += 1
            self._log("plan_stale", seq=res.seq)
            return
        self.replan_pending = False
        if not res.ok:
            self.r.failed_plans += 1
            self._log("plan_failed", request=res.kind)
            if res.kind == "plan":
                if self.r.failed_plans >= 2:
                    self.mode, self.r.outcome = "gave_up", "gave_up"
                else:
                    self._request_plan()
                return
            if self.mode == "holding":
                self.mode = "running"
            self.unmet_raised = False
            return
        if res.kind == "plan":
            self.plan = res.plan
            self.queue = [s["id"] for s in self.plan["steps"]]
            self.status = {sid: "pending" for sid in self.queue}
            self.safety.set_constraints(self.plan)
            if self.mode == "waiting":
                self.mode = "running"
            self._log("plan_applied", plan=self.plan)
            self._start_next_now()
            self.event("plan_arrived", "the first plan arrived: " + self.plan.get("reading", ""))
            return
        if res.kind == "react" and res.right_now:
            ev = Event(0, self.t, "react", "", None)
            self._right_now(res.right_now, ev)
        if res.kind == "react" and not res.change_plan:
            if self.mode == "holding":
                self.hold_until = self.t + int(self.cfg.hold_s * self.g.ticks_per_s)
            return
        newp, q = P.apply_diff(self.plan, res.diff)
        q = [sid for sid in q if not (self.status.get(sid) == "done" and self.done_at.get(sid, -1) >= res.t_request)]
        held = self.state.arm.holding
        byid = P.steps_by_id(newp)
        if held and self.cur is not None and self.cur.get("object") == held and (not q or byid[q[0]]["object"] != held):
            # never drop a step with its object in the gripper: the plan was written before the grasp
            if self.cur["id"] in q:
                q.remove(self.cur["id"])
            q.insert(0, self.cur["id"])
            self._log("kept_running_step", step=self.cur["id"])
        errs = P.validate(newp, self.state, held, [sid for sid in q if self.status.get(sid) != "done" or sid in res.diff.get("pending_order", [])])
        if errs and self.misfits < 2:
            # the world moved on while the LLM worked: ask again from where things stand now
            self.misfits += 1
            self._log("plan_misfit", errors=errs)
            ev = Event(0, self.t, "misfit", "the new plan no longer fits the world: " + "; ".join(errs[:3]))
            self._start_replan(res.route, res.kind if res.kind != "react" else "replan", ev.text, ev)
            return
        self.misfits = 0
        if self.cur is not None and (not q or q[0] != self.cur["id"]):
            sid = self.cur["id"]
            self.skill, self.cur = None, None
            self.status[sid] = "pending" if sid in q else "dropped"
        for sid in list(self.status):
            if self.status[sid] == "pending" and sid not in q:
                self.status[sid] = "dropped"
        for sid in q:
            if self.status.get(sid) != "running":
                self.status[sid] = "pending"
        self.plan, self.queue = newp, q
        self.safety.set_constraints(self.plan)
        self.unmet_raised = False
        if self.mode == "holding":
            self.mode, self.hold_until = "running", None
        self._log("plan_applied", diff=res.diff, queue=q)
        self.event("plan_arrived", "a new plan arrived: " + (res.diff.get("reading") or ""))

    def _start_next_now(self):
        if self.mode == "running" and self.skill is None and self.awaiting is None:
            self._start_next()


def near_xy(a, b, cm: float) -> bool:
    return dxy(a, b) <= cm


__all__ = ["Episode", "EpisodeResult", "HarnessConfig", "LoopDetector", "is_stop", "Combined"]
