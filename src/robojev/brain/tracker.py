"""Where the episode is in its plan: the plan, the queue of steps still to run, each step's status.

Pure bookkeeping, no I/O: the loop decides when, this module says what changes. The rules are
e12v2's harness rules (docs/v2.md), lifted out of its tick loop:
  - statuses: pending | running | done | failed | skipped | dropped;
  - in-plan fixes: retry, skip, re_queue, another_step_first, each touching only the queue; and
    swap_target, which rewrites the running step's object (and the done conditions that named it at
    that destination) and leaves the queue alone (`swapped` writes the plan, the loop validates it,
    `adopt_swap` installs it);
  - a diff from a replan: steps done since the request are not re-run, and a step whose object is
    in the gripper is never dropped (the plan was written before the grasp);
  - done: every measurable done condition true (code measures), every other one true by the
    decision's answer on step_done (assumed true until answered otherwise).

Arms. Each step runs on one arm (its "arm" field, else `default_arm`: a one-arm plan has none), and
each arm has at most one running step. The queue is one list in the planner's order, all arms
interleaved; an arm's next step is its first pending one (arms.py says when it may start). Every
rule above holds per arm: "the running step", "the gripper" and "first" mean the event's arm's.
A fix puts a step back at the first pending position: with one arm that is just after the running
step (e12v2's position); with several it keeps the step ahead of any later step that depends on it.
"""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping

from .core import plan as P
from .core.data import Event, WorldState

from .arms import arm_of


class PlanTracker:
    def __init__(self, default_arm: str = "arm_0") -> None:
        self.default_arm = default_arm
        self.plan: dict | None = None
        self.queue: list[str] = []
        self.status: dict[str, str] = {}
        self.done_at: dict[str, float] = {}
        self.tries: Counter = Counter()          # step id -> failures so far
        self.running: dict[str, dict] = {}       # arm -> its running step
        self.last: dict[str, dict] = {}          # arm -> the step that last finished or failed on it
        self.done_answers: dict[str, bool] = {}  # done-condition id -> the decision's answer
        self.unmet_raised = False                # a plan_done_unmet event is out for this plan

    # ---------------------------------------------------------------- reading
    def steps(self) -> dict[str, dict]:
        return P.steps_by_id(self.plan) if self.plan else {}

    def arm_of(self, step: dict) -> str:
        return arm_of(step, self.default_arm)

    def current(self, arm: str) -> dict | None:
        """The step running on `arm`."""
        return self.running.get(arm)

    def next_for(self, arm: str) -> str | None:
        """`arm`'s next step: its first pending one in queue order."""
        steps = self.steps()
        return next((sid for sid in self.queue if self.status.get(sid) == "pending" and self.arm_of(steps[sid]) == arm), None)

    def pending(self, exclude: set[str]) -> list[str]:
        """Pending step ids in queue order, other than `exclude`."""
        return [sid for sid in self.queue if sid not in exclude and self.status.get(sid) == "pending"]

    def unmet(self, state: WorldState) -> list[dict]:
        """Done conditions not yet true."""
        out = []
        for c in self.plan["done_when"]:
            v = P.measure(c, state)
            if not (self.done_answers.get(c["id"], True) if v is None else v):
                out.append(c)
        return out

    # ---------------------------------------------------------------- a full plan
    def load(self, plan: dict) -> None:
        self.plan = plan
        self.queue = [s["id"] for s in plan["steps"]]
        self.status = {sid: "pending" for sid in self.queue}
        self.unmet_raised = False

    # ---------------------------------------------------------------- steps
    def begin(self, sid: str) -> dict:
        s = self.steps()[sid]
        self.running[self.arm_of(s)] = s
        self.status[sid] = "running"
        return s

    def could_not_start(self, sid: str) -> dict:
        s = self.steps()[sid]
        self.queue.remove(sid)
        self.status[sid] = "failed"
        self.tries[sid] += 1
        self.last[self.arm_of(s)] = s
        return s

    def end(self, arm: str, ok: bool, t: float) -> dict:
        """`arm`'s running step finished (ok) or failed."""
        s = self.running.pop(arm)
        sid = s["id"]
        self.last[arm] = s
        if sid in self.queue:
            self.queue.remove(sid)
        if ok:
            self.status[sid], self.done_at[sid] = "done", t
        else:
            self.status[sid] = "failed"
            self.tries[sid] += 1
        return s

    # ---------------------------------------------------------------- in-plan fixes
    def _at(self) -> int:
        """Where a fix puts a step back: the first pending position (module docstring)."""
        return next((i for i, sid in enumerate(self.queue) if self.status.get(sid) == "pending"), len(self.queue))

    def apply_fix(self, fix: str, ev: Event, arm: str, holding: str | None, other: str | None) -> bool:
        """Apply one in-plan fix for `arm`'s event. `holding`: what that arm holds; `other`: a pending
        step of that arm that could start now (another_step_first).
        -> True if the arm's running step was dropped (the loop then releases its skill)."""
        if fix == "none" or not self.plan:
            return False
        at = self._at()
        cur = self.running.get(arm)
        dropped = False
        if fix == "retry":
            if ev.step_id and self.status.get(ev.step_id) == "failed":
                self.status[ev.step_id] = "pending"
                self.queue.insert(at, ev.step_id)
        elif fix == "skip":
            sid = cur["id"] if cur is not None else ev.step_id
            if cur is not None:
                if holding:
                    return False                 # never drop a step with its object in the gripper
                del self.running[arm]
                dropped = True
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
        elif fix == "another_step_first" and other is not None:
            self.queue.remove(other)
            self.queue.insert(at, other)
            if ev.step_id and self.status.get(ev.step_id) == "failed":
                self.status[ev.step_id] = "pending"
                self.queue.insert(at + 1, ev.step_id)
        return dropped

    # ---------------------------------------------------------------- swap_target
    def swapped(self, arm: str, oid: str) -> dict | None:
        """The plan with `arm`'s running grasping step going for `oid` instead of its object, same
        destination, and every done condition that put the old object there now naming `oid` (the
        condition is about the destination; the object was the means). Not applied: the loop
        validates it first. None when no grasping step runs on `arm` or `oid` is its object already."""
        cur = self.running.get(arm)
        if self.plan is None or cur is None or cur["skill"] not in P.GRASPING or oid == cur["object"]:
            return None
        old, goal = cur["object"], P.step_goal(cur)
        newp = copy.deepcopy(self.plan)
        for s in newp["steps"]:
            if s["id"] == cur["id"]:
                s["object"] = oid
        for c in newp["done_when"]:
            if c["object"] == old and goal is not None and c["target"] in (goal[1], goal[1][3:] if goal[1].startswith("on:") else None):
                c["object"] = oid
                c["text"] = c["text"].replace(old, oid)
        return newp

    def adopt_swap(self, arm: str, newp: dict) -> dict:
        """Install a plan from `swapped` once it validated and the rewritten step has started. -> the step."""
        cur = self.running[arm]
        self.plan = newp
        self.running[arm] = step = P.steps_by_id(newp)[cur["id"]]
        return step

    # ---------------------------------------------------------------- a replan's diff
    def merge(self, diff: dict, t_request: float, holdings: Mapping[str, str | None]) -> tuple[dict, list[str], list[str]]:
        """-> (new plan, new queue, the queue to validate). Not applied yet: the loop validates first.
        `holdings`: arm -> what it holds now. An arm's running step on the object it holds is kept
        first in the queue when the diff would put another of its steps first."""
        newp, q = P.apply_diff(self.plan, diff)
        q = [sid for sid in q if not (self.status.get(sid) == "done" and self.done_at.get(sid, -1.0) >= t_request)]
        byid = P.steps_by_id(newp)
        for arm, cur in self.running.items():
            held = holdings.get(arm)
            if not held or cur.get("object") != held:
                continue
            mine = [sid for sid in q if self.arm_of(byid[sid]) == arm]
            if not mine or byid[mine[0]]["object"] != held:
                if cur["id"] in q:
                    q.remove(cur["id"])
                q.insert(0, cur["id"])
        check = [sid for sid in q if self.status.get(sid) != "done" or sid in diff["pending_order"]]
        return newp, q, check

    def install(self, newp: dict, q: list[str]) -> list[str]:
        """Adopt a merged plan. -> the arms whose running step is no longer their first (dropped)."""
        byid = P.steps_by_id(newp)
        dropped = []
        for arm, cur in list(self.running.items()):
            mine = [sid for sid in q if self.arm_of(byid[sid]) == arm]
            if not mine or mine[0] != cur["id"]:
                del self.running[arm]
                dropped.append(arm)
                self.status[cur["id"]] = "pending" if cur["id"] in q else "dropped"
        for sid in list(self.status):
            if self.status[sid] == "pending" and sid not in q:
                self.status[sid] = "dropped"
        for sid in q:
            if self.status.get(sid) != "running":
                self.status[sid] = "pending"
        self.plan, self.queue, self.unmet_raised = newp, q, False
        return dropped
