"""Two or more arms under one brain (docs/widowx-pair.md, "Crawl: one brain, two arms"): which arm
reaches what, which steps may run side by side, and the holding rule per arm.

Pure functions over the plan (e12v2 step dicts, each naming its arm in "arm"), the manifest's arms
and the protocol world (metres). The loop, the planner's checks and the offline mock planner use
them; nothing here calls the server.

- Reach. An arm reaches a point when it lies inside the arm's workspace box (x and y; the manifest's
  numbers, in the frame the arms share). As a plan runs, an object is where its last step put it:
  a slot or bin is a point; a free spot on the table, a push, or a gripper is somewhere only that
  arm is sure to reach; the person is not reach's question (the server refuses what it must).
- Side by side. One skill per arm, and an arm runs its own steps in queue order. A step waits while
  its object is in another arm's gripper or claimed by another arm (ObjectObs.claimed_by), and
  while another arm's running step, or an earlier pending step on another arm, uses the same object
  or one-object place: its object, its target, or the place its object leaves. The earliest pending
  step therefore never waits on a pending step, only on running ones, so the queue cannot deadlock.
- Holding. An arm holding an object must put it somewhere with its first step (e12v2's rule, which
  core/plan.validate applies to its one arm, applied to each).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .core import plan as P

from ..protocol import ArmSpec
from ..protocol import WorldState as RobotWorld

GRASPING = P.GRASPING
NO_OBJECT = ("survey", "hold")
POINT_PLACES = ("slot", "bin", "holder")      # places that are one point an arm either reaches or not


def reaches(spec: ArmSpec, x: float, y: float) -> bool:
    (x0, x1), (y0, y1), _ = spec.workspace
    return x0 <= x <= x1 and y0 <= y <= y1


def arm_of(step: dict, default: str) -> str:
    """The arm a step runs on: its "arm" field, or `default` (a one-arm plan has no such field)."""
    return step.get("arm") or default


class Reach:
    """Where each object will be as a plan runs, for reach: an (x, y) in metres, "arm:<id>" (only
    that arm is sure to reach it) or "person". Start from the world now; `apply` each step in order."""

    def __init__(self, world: RobotWorld, specs: Mapping[str, ArmSpec]):
        self.specs, self.places = dict(specs), world.places
        self.at: dict[str, tuple[float, float] | str] = {}
        for o in world.objects.values():
            p = world.places.get(o.where)
            if o.where.startswith("gripper:"):
                self.at[o.id] = "arm:" + o.where[8:]
            elif o.where == "person":
                self.at[o.id] = "person"
            elif p is not None and p.kind in POINT_PLACES:
                self.at[o.id] = (p.x, p.y)
            else:
                self.at[o.id] = (o.x, o.y)

    def _arms_at(self, at: tuple[float, float] | str | None) -> list[str]:
        if at is None or at == "person":
            return list(self.specs)
        if isinstance(at, str):
            return [at[4:]] if at[4:] in self.specs else []
        return [a for a, s in self.specs.items() if reaches(s, *at)]

    def object_arms(self, oid: str) -> list[str]:
        """Arms that reach object `oid` where it is now in the replay."""
        return self._arms_at(self.at.get(oid))

    def place_arms(self, pid: str) -> list[str]:
        """Arms that can put something at place `pid`: every arm for the table (each has a spot in
        its own half) and the person; else the arms whose box holds the place."""
        p = self.places.get(pid)
        if p is None or p.kind in ("table", "person"):
            return list(self.specs)
        return [a for a, s in self.specs.items() if reaches(s, p.x, p.y)]

    def target_arms(self, step: dict) -> list[str]:
        """Arms that reach where the step puts its object."""
        if step["skill"] in ("move_object", "insert"):
            return self.place_arms(step["target"])
        if step["skill"] == "stack_on":
            return self.object_arms(step["target"])
        return list(self.specs)

    def where(self, oid: str) -> str:
        at = self.at.get(oid)
        if isinstance(at, tuple):
            return f"x {at[0]:.2f} m, y {at[1]:.2f} m"
        return {"person": "held by the person"}.get(at, f"where only {at[4:]} reaches" if at else "unknown")

    def apply(self, step: dict, arm: str) -> None:
        sk, o, t = step["skill"], step["object"], step["target"]
        if sk in NO_OBJECT or o not in self.at:
            return
        if sk in ("move_object", "insert"):
            p = self.places.get(t)
            self.at[o] = (p.x, p.y) if p is not None and p.kind in POINT_PLACES else f"arm:{arm}"
        elif sk == "stack_on":
            self.at[o] = self.at.get(t, f"arm:{arm}")
        elif sk == "hand_over":
            self.at[o] = "person"
        else:                                     # push: wherever it slid, in this arm's reach
            self.at[o] = f"arm:{arm}"


def reach_errors(order: Iterable[dict], world: RobotWorld, specs: Mapping[str, ArmSpec]) -> list[str]:
    """Steps (in run order, e12v2 names) given to an arm that cannot reach their object where it will
    be by then, or their destination."""
    r, errs = Reach(world, specs), []
    for s in order:
        arm = s.get("arm")
        if arm not in specs or s["skill"] in NO_OBJECT or s["object"] not in r.at:
            continue
        o = s["object"]
        if arm not in r.object_arms(o):
            errs.append(f"step {s['id']}: {arm} cannot reach {o} where it will be then ({r.where(o)}); "
                        f"arms that can: {', '.join(r.object_arms(o)) or 'none'}")
        if s["skill"] in ("move_object", "stack_on", "insert") and arm not in r.target_arms(s):
            errs.append(f"step {s['id']}: {arm} cannot reach {s['target']}; arms that can: {', '.join(r.target_arms(s)) or 'none'}")
        r.apply(s, arm)
    return errs


def holding_errors(order: Iterable[dict], holdings: Mapping[str, str | None]) -> list[str]:
    """e12v2's holding rule per arm: an arm holding an object must put it somewhere with its first step."""
    steps, errs = list(order), []
    for arm, held in holdings.items():
        if not held:
            continue
        first = next((s for s in steps if s.get("arm") == arm), None)
        if first is None:
            errs.append(f"{arm} is holding {held}: the plan needs a step on {arm} that puts it somewhere")
        elif first["object"] != held or first["skill"] not in GRASPING:
            errs.append(f"{arm} is holding {held}: its first step must put it somewhere "
                        f"(pick_and_place, stack_on, hand_over or insert of {held})")
    return errs


def _uses(step: dict, source: str | None, objects: set[str], exclusive: set[str]) -> set[str]:
    """What a step uses that another arm must not use at the same time: objects, and one-object
    places (its target, and the place its object leaves)."""
    out = set()
    for x in (step["object"], step["target"], source):
        if not x or x == P.NONE:
            continue
        if x.startswith("on:"):
            x = x[3:]
        if x in objects or x in exclusive:
            out.add(x)
    return out


def waits_for(sid: str, queue: list[str], steps: Mapping[str, dict], status: Mapping[str, str], world: RobotWorld,
              default_arm: str, names: Mapping[str, str] | None = None) -> str | None:
    """Why step `sid` must wait for another arm (literal text, things named by `names`), or None.
    The rule is the module docstring's "Side by side"."""
    names = names or {}
    s = steps[sid]
    arm = arm_of(s, default_arm)
    o = world.objects.get(s["object"])
    if o is not None:
        if o.where.startswith("gripper:") and o.where != f"gripper:{arm}":
            return f"{names.get(o.id, o.id)} is in {o.where[8:]}'s gripper"
        if o.claimed_by not in (None, arm):
            return f"{names.get(o.id, o.id)} is claimed by {o.claimed_by}"
    objects = set(world.objects)
    exclusive = {p.id for p in world.places.values() if p.capacity == 1}
    where = {x.id: x.where for x in world.objects.values()}
    uses: dict[str, set[str]] = {}
    for q in queue:                               # replay: the place each step's object leaves
        t = steps[q]
        uses[q] = _uses(t, where.get(t["object"]), objects, exclusive)
        g = P.step_goal(t)
        if g:
            where[g[0]] = g[1]
    mine, earlier = uses.get(sid, set()), True
    for q in queue:
        if q == sid:
            earlier = False
            continue
        t, st = steps[q], status.get(q)
        other = arm_of(t, default_arm)
        if other == arm or not (st == "running" or (earlier and st == "pending")):
            continue
        shared = sorted(mine & uses[q])
        if shared:
            return (f"{other}'s step {P.step_text(t, names)} ({'running' if st == 'running' else 'earlier in the plan'}) "
                    f"also uses {names.get(shared[0], shared[0])}")
    return None
