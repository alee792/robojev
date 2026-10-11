"""Where the episode is in its plan: the plan, the queue of steps still to run, each step's status.

Pure bookkeeping, no I/O: the loop decides when, this module says what changes. The rules are
e12v2's harness rules (docs/v2.md), lifted out of its tick loop:
  - statuses: pending | running | done | failed | skipped | dropped;
  - in-plan fixes: retry, skip, re_queue, another_step_first, each touching only the queue;
  - a diff from a replan: steps done since the request are not re-run, and a step whose object is
    in the gripper is never dropped (the plan was written before the grasp);
  - done: every measurable done condition true (code measures), every other one true by the
    decision's answer on step_done (assumed true until answered otherwise).
"""
from __future__ import annotations

from collections import Counter

from .core import plan as P
from .core.data import Event, WorldState


class PlanTracker:
    def __init__(self) -> None:
        self.plan: dict | None = None
        self.queue: list[str] = []
        self.status: dict[str, str] = {}
        self.done_at: dict[str, float] = {}
        self.tries: Counter = Counter()          # step id -> failures so far
        self.cur: dict | None = None             # the running step
        self.last_step: dict | None = None       # the step that last finished or failed
        self.done_answers: dict[str, bool] = {}  # done-condition id -> the decision's answer
        self.unmet_raised = False                # a plan_done_unmet event is out for this plan

    # ---------------------------------------------------------------- reading
    def steps(self) -> dict[str, dict]:
        return P.steps_by_id(self.plan) if self.plan else {}

    def next_pending(self) -> str | None:
        return next((sid for sid in self.queue if self.status.get(sid) == "pending"), None)

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
        self.cur = self.steps()[sid]
        self.status[sid] = "running"
        return self.cur

    def could_not_start(self, sid: str) -> dict:
        s = self.steps()[sid]
        self.queue.remove(sid)
        self.status[sid] = "failed"
        self.tries[sid] += 1
        self.last_step = s
        return s

    def end(self, ok: bool, t: float) -> dict:
        """The running step finished (ok) or failed."""
        s, sid = self.cur, self.cur["id"]
        self.cur, self.last_step = None, s
        if sid in self.queue:
            self.queue.remove(sid)
        if ok:
            self.status[sid], self.done_at[sid] = "done", t
        else:
            self.status[sid] = "failed"
            self.tries[sid] += 1
        return s

    # ---------------------------------------------------------------- in-plan fixes
    def apply_fix(self, fix: str, ev: Event, holding: str | None, other: str | None) -> bool:
        """Apply one in-plan fix. `other` is a pending step that could start now (another_step_first).
        -> True if the running step was dropped (the loop then releases the arm's skill)."""
        if fix == "none" or not self.plan:
            return False
        at = 1 if self.cur is not None else 0
        dropped = False
        if fix == "retry":
            if ev.step_id and self.status.get(ev.step_id) == "failed":
                self.status[ev.step_id] = "pending"
                self.queue.insert(at, ev.step_id)
        elif fix == "skip":
            sid = self.cur["id"] if self.cur is not None else ev.step_id
            if self.cur is not None:
                if holding:
                    return False                 # never drop a step with its object in the gripper
                self.cur, dropped = None, True
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

    # ---------------------------------------------------------------- a replan's diff
    def merge(self, diff: dict, t_request: float, holding: str | None) -> tuple[dict, list[str], list[str]]:
        """-> (new plan, new queue, the queue to validate). Not applied yet: the loop validates first."""
        newp, q = P.apply_diff(self.plan, diff)
        q = [sid for sid in q if not (self.status.get(sid) == "done" and self.done_at.get(sid, -1.0) >= t_request)]
        byid = P.steps_by_id(newp)
        cur = self.cur
        if holding and cur is not None and cur.get("object") == holding and (not q or byid[q[0]]["object"] != holding):
            if cur["id"] in q:
                q.remove(cur["id"])
            q.insert(0, cur["id"])
        check = [sid for sid in q if self.status.get(sid) != "done" or sid in diff["pending_order"]]
        return newp, q, check

    def install(self, newp: dict, q: list[str]) -> bool:
        """Adopt a merged plan. -> True if the running step is no longer first (it was dropped)."""
        dropped = False
        if self.cur is not None and (not q or q[0] != self.cur["id"]):
            sid = self.cur["id"]
            self.cur, dropped = None, True
            self.status[sid] = "pending" if sid in q else "dropped"
        for sid in list(self.status):
            if self.status[sid] == "pending" and sid not in q:
                self.status[sid] = "dropped"
        for sid in q:
            if self.status.get(sid) != "running":
                self.status[sid] = "pending"
        self.plan, self.queue, self.unmet_raised = newp, q, False
        return dropped
