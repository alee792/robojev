"""Motor-only skills (docs/v2.md "Skills"): move_object, hand_over, stack_on, push, survey, hold.

Each implements core's Skill interface: precondition / start / step (one short motion chunk, which the
harness passes through the safety filter) / status, plus the literal phase text, the heading for
"in the arm's path", re-target and a safe point. Skills know how to move the arm, not why: no task
logic, no constraints (the safety filter enforces those). They read the perceived world state only;
the SimWorld reference is used for two motor facts (where a place's drop point is, a free table spot)
and to post the arm's intent for the scripted person's triggers.
"""

from __future__ import annotations

import math

from ..core.data import Command, WorldState, dxy
from .world import BLOCK_CM, LOW_Z, TICKS_PER_S, TRAVEL_Z, SimWorld

EPS = 0.3
STUCK_TICKS = 15          # refused by the safety filter this long (1.5 s) = failed
HANDOVER_WAIT_TICKS = 15 * TICKS_PER_S
SPEED = 2.0
DIRS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "toward_robot": (0.0, -1.0), "away_from_robot": (0.0, 1.0)}


def _where(names, where):
    if where == "table":
        return "the table"
    if where.startswith("on:"):
        return f"top of {names.get(where[3:], where[3:])}"
    return names.get(where, where)


class SkillBase:
    name = "skill"

    def __init__(self, world: SimWorld):
        self.world = world
        self._status, self._reason = "done", None
        self.stuck = 0
        self.wait = 0
        self.step_args: dict = {}

    def names(self):
        return self.world.names()

    def precondition(self, state: WorldState, step: dict) -> str | None:
        return None

    def start(self, state: WorldState, step: dict) -> None:
        self.step_args = step
        self._status, self._reason, self.stuck, self.wait = "running", None, 0, 0

    def feedback(self, executed: bool, reason: str | None) -> None:
        if executed:
            self.stuck = 0
            return
        self.stuck += 1
        if self.stuck >= STUCK_TICKS:
            self.fail(f"blocked by the safety filter for {STUCK_TICKS / TICKS_PER_S:.1f} s: {reason}")

    def status(self):
        return self._status, self._reason

    def fail(self, why: str):
        self._status, self._reason = "failed", why
        self.world.arm.intent = None

    def finish(self):
        self._status = "done"
        self.world.arm.intent = None

    def heading(self, state):
        return None

    def retarget(self, state):
        pass

    def safe_point(self, state: WorldState):
        a = state.arm
        return Command("move", (a.x, a.y, TRAVEL_Z), SPEED) if a.z < TRAVEL_Z - EPS else None

    def phase_text(self, state) -> str:
        return self.name

    @staticmethod
    def at(state, xyz) -> bool:
        a = state.arm
        return math.dist((a.x, a.y, a.z), xyz) < EPS


class Survey(SkillBase):
    name = "survey"

    def step(self, state):
        a = state.arm
        if a.z < TRAVEL_Z - EPS:
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        self.wait += 1
        if self.wait >= 5:
            self.finish()
        return Command("wait")

    def phase_text(self, state):
        return "lifting the gripper to look over the table"


class Hold(SkillBase):
    name = "hold"

    def step(self, state):
        self.wait += 1
        if self.wait >= TICKS_PER_S:
            self.finish()
        return Command("wait")

    def phase_text(self, state):
        return "staying still for a second (a planned hold step)"


class _PickPlace(SkillBase):
    """pick the object (unless already holding it), carry it to `dest`, put it down or hand it over."""
    handover = False

    def precondition(self, state, step):
        o = state.obj(step["object"])
        n = self.names()
        if o is None:
            return f"{n.get(step['object'], step['object'])} is not in view"
        if o.where == "person":
            return f"{o.name} is held by the person"
        holding = state.arm.holding
        if holding and holding != o.id:
            return f"the gripper is holding {n.get(holding, holding)}"
        top = state.top_of(o.id)
        if top and holding != o.id:
            return f"{o.name} has {n.get(top, top)} on top of it"
        return self._dest_problem(state, step)

    def _dest_problem(self, state, step):
        return None

    def start(self, state, step):
        super().start(state, step)
        self.obj = step["object"]
        o = state.obj(self.obj)
        self.obj_xyz = (o.x, o.y, o.z)
        self.dest = None
        a = state.arm
        if a.holding == self.obj:
            self.phase = "lift" if a.z < TRAVEL_Z - EPS else "carry"
        else:
            self.phase = "rise0" if a.z < TRAVEL_Z - EPS else "approach"
        self._intent()

    def _intent(self):
        self.world.arm.intent = {"skill": self.name, "object": self.obj, "phase": self.phase,
                                 "dest": self.dest[:2] if self.dest else None, "target": self.step_args.get("target")}

    def _set_dest(self, state):
        raise NotImplementedError

    def heading(self, state):
        if self.phase in ("rise0", "approach"):
            return self.obj_xyz[:2]
        if self.phase in ("lift", "carry"):
            if self.dest is None:
                self._set_dest(state)
            return self.dest[:2] if self.dest else None
        return None

    def retarget(self, state):
        o = state.obj(self.obj)
        if o is None or state.arm.holding == self.obj:
            return
        self.obj_xyz = (o.x, o.y, o.z)
        if self.phase in ("descend", "grip"):
            self.phase = "rise0"
        self._intent()

    def safe_point(self, state):
        if self.phase in ("descend", "grip"):
            self.phase = "rise0"
        elif self.phase in ("lower", "release"):
            self.phase = "carry"
        return super().safe_point(state)

    def step(self, state):
        a = state.arm
        ph = self.phase
        if ph == "rise0":
            if a.z >= TRAVEL_Z - EPS:
                self.phase = "approach"
                return self.step(state)
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        if ph == "approach":
            tgt = (self.obj_xyz[0], self.obj_xyz[1], TRAVEL_Z)
            if self.at(state, tgt):
                self.phase = "descend"
                self._intent()
                return self.step(state)
            return Command("move", tgt, SPEED)
        if ph == "descend":
            tgt = self.obj_xyz
            if self.at(state, tgt):
                self.phase, self.wait = "grip", 0
                return Command("grip")
            return Command("move", tgt, SPEED)
        if ph == "grip":
            self.wait += 1
            if self.wait < 2:
                return Command("wait")
            if a.holding != self.obj:
                o = state.obj(self.obj)
                where = f"it is {dxy((a.x, a.y), (o.x, o.y)):.0f} cm away" if o else "it is not in view"
                self.fail(f"nothing grasped: {self.names().get(self.obj)} was not at the grasp spot ({where})")
                return None
            self.phase = "lift"
            self._intent()
            return self.step(state)
        if ph == "lift":
            if a.z >= TRAVEL_Z - EPS:
                self.phase = "carry"
                self._set_dest(state)
                self._intent()
                return self.step(state)
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        if ph == "carry":
            if self.dest is None:
                self._set_dest(state)
            if self.dest is None:
                return Command("wait")       # hand_over: waiting for a hand to be held out
            tgt = (self.dest[0], self.dest[1], TRAVEL_Z)
            if self.at(state, tgt):
                self.phase = "lower"
                self._intent()
                return self.step(state)
            return Command("move", tgt, SPEED)
        if ph == "lower":
            if self.at(state, self.dest):
                self.phase, self.wait = "release", 0
                return Command("wait")
            return Command("move", self.dest, SPEED)
        if ph == "release":
            self.wait += 1
            if self.wait < 2:
                return Command("wait")
            self.phase = "rise"
            return Command("release")
        if ph == "rise":
            if a.z >= TRAVEL_Z - EPS:
                self.finish()
                return None
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        return None

    def dest_name(self):
        return self.names().get(self.step_args.get("target"), self.step_args.get("target"))

    def phase_text(self, state):
        n = self.names()
        on = n.get(self.obj, self.obj)
        a = state.arm
        ph = self.phase
        if ph in ("rise0", "approach"):
            d = dxy((a.x, a.y), self.obj_xyz[:2])
            o = state.obj(self.obj)
            off = dxy(self.obj_xyz[:2], (o.x, o.y)) if o else 0.0
            if off > 2.0:
                return (f"moving the empty gripper toward where {on} was when this step started, {d:.0f} cm away; "
                        f"{on} itself is now {off:.0f} cm from that spot; nothing grasped yet")
            return f"moving the empty gripper toward {on} at travel height, {d:.0f} cm away; nothing grasped yet"
        if ph == "descend":
            return f"lowering the open gripper onto {on}; not grasped yet"
        if ph == "grip":
            return f"closing the gripper on {on}"
        if ph == "lift":
            return f"holding {on} and lifting it"
        if ph == "carry":
            if self.dest is None:
                return f"holding {on} at travel height, waiting for the person to hold out a hand"
            d = dxy((a.x, a.y), self.dest[:2])
            return f"holding {on} at travel height, carrying it to {self.dest_name()}, {d:.0f} cm away"
        if ph == "lower":
            return f"holding {on} and lowering it at {self.dest_name()}, not yet released"
        if ph == "release":
            return f"lowered at {self.dest_name()} holding {on}, not yet released"
        if ph == "rise":
            return f"released {on} at {self.dest_name()}; lifting the empty gripper"
        return ph


class MoveObject(_PickPlace):
    name = "move_object"

    def _dest_problem(self, state, step):
        p = state.places.get(step["target"])
        if p is None:
            return f"{step['target']} is not a place"
        if p.capacity == 1:
            occ = state.occupant(p.id)
            if occ and occ != step["object"]:
                return f"{p.name} is occupied by {self.names().get(occ, occ)}"
        return None

    def _set_dest(self, state):
        t = self.step_args["target"]
        if t == "table":
            o = state.obj(self.obj)
            x, y = self.world.free_spot(near=(self.obj_xyz[0], min(self.obj_xyz[1], 40.0)), exclude=(self.obj,))
            _ = o
        else:
            x, y = self.world.place_xy(t, self.obj)
        self.dest = (x, y, LOW_Z)


class StackOn(_PickPlace):
    name = "stack_on"

    def _dest_problem(self, state, step):
        b = state.obj(step["target"])
        n = self.names()
        if b is None:
            return f"{n.get(step['target'], step['target'])} is not in view"
        top = state.top_of(b.id)
        if top and top != step["object"]:
            return f"{b.name} already has {n.get(top, top)} on top of it"
        if b.where in ("gripper", "person"):
            return f"{b.name} is not on the table"
        return None

    def _set_dest(self, state):
        b = state.obj(self.step_args["target"])
        self.dest = (b.x, b.y, b.z + BLOCK_CM)


class HandOver(_PickPlace):
    name = "hand_over"

    def _set_dest(self, state):
        h = state.hand
        if h is None or not h.held_out:
            self.dest = None
            return
        a = state.arm
        vx, vy = a.x - h.x, a.y - h.y
        n = math.hypot(vx, vy) or 1.0
        self.dest = (h.x + 8 * vx / n, h.y + 8 * vy / n, TRAVEL_Z)

    def dest_name(self):
        return "the person's held-out hand"

    def step(self, state):
        if self.phase == "carry":
            self.wait += 1
            if self.dest is None:
                self._set_dest(state)
            if self.wait > HANDOVER_WAIT_TICKS:
                self.fail("no hand was held out to take it within 15 s")
                return None
        if self.phase == "lower":          # presenting: wait for the person to take it
            if state.arm.holding != self.obj:
                self.phase = "rise"
                return self.step(state)
            self.wait += 1
            if self.wait > HANDOVER_WAIT_TICKS:
                self.fail("the person did not take it within 15 s")
                return None
            if state.hand is None or not state.hand.held_out:
                self.phase, self.dest = "carry", None
            return Command("wait")
        return super().step(state)

    def phase_text(self, state):
        if self.phase == "lower":
            return f"holding {self.names().get(self.obj)} out to the person's hand, waiting for them to take it"
        return super().phase_text(state)


class Push(SkillBase):
    """One fingertip on the object's top, slide it ~6 cm in a direction, lift."""
    name = "push"

    def precondition(self, state, step):
        o = state.obj(step["object"])
        if o is None:
            return f"{step['object']} is not in view"
        if state.arm.holding:
            return "the gripper is holding something"
        if o.where not in ("table",) and not o.where.startswith("tray_slot"):
            return f"{o.name} is not on the table"
        return None

    def start(self, state, step):
        super().start(state, step)
        self.obj = step["object"]
        o = state.obj(self.obj)
        self.o_xy = (o.x, o.y)
        ux, uy = DIRS[step["direction"]]
        self.end = (o.x + 6 * ux, o.y + 6 * uy)
        self.phase = "rise0" if state.arm.z < TRAVEL_Z - EPS else "approach"
        self.world.arm.intent = {"skill": "push", "object": self.obj, "phase": self.phase, "dest": self.end, "target": None}

    def heading(self, state):
        return self.o_xy if self.phase in ("rise0", "approach") else self.end

    def retarget(self, state):
        o = state.obj(self.obj)
        if o is not None and self.phase in ("rise0", "approach"):
            ux, uy = DIRS[self.step_args["direction"]]
            self.o_xy, self.end = (o.x, o.y), (o.x + 6 * ux, o.y + 6 * uy)

    def step(self, state):
        a = state.arm
        if self.phase == "rise0":
            if a.z >= TRAVEL_Z - EPS:
                self.phase = "approach"
                return self.step(state)
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        if self.phase == "approach":
            tgt = (*self.o_xy, TRAVEL_Z)
            if self.at(state, tgt):
                self.phase = "press"
                return self.step(state)
            return Command("move", tgt, SPEED)
        if self.phase == "press":
            tgt = (*self.o_xy, LOW_Z + 2.5)
            if self.at(state, tgt):
                self.phase = "slide"
                return self.step(state)
            return Command("move", tgt, SPEED, fingertip=True)
        if self.phase == "slide":
            tgt = (*self.end, LOW_Z + 2.5)
            if self.at(state, tgt):
                self.phase = "rise"
                return self.step(state)
            return Command("move", tgt, 1.0, fingertip=True, drag=self.obj)
        if self.phase == "rise":
            if a.z >= TRAVEL_Z - EPS:
                self.finish()
                return None
            return Command("move", (a.x, a.y, TRAVEL_Z), SPEED)
        return None

    def phase_text(self, state):
        on = self.names().get(self.obj, self.obj)
        return {"rise0": f"lifting to travel height before pushing {on}",
                "approach": f"moving above {on} to push it {self.step_args['direction']}",
                "press": f"lowering one fingertip onto the top of {on}",
                "slide": f"sliding {on} {self.step_args['direction']} with one fingertip",
                "rise": f"pushed {on}; lifting the gripper"}.get(self.phase, self.phase)


def make_skills(world: SimWorld) -> dict:
    return {c.name: c(world) for c in (MoveObject, HandOver, StackOn, Push, Survey, Hold)}
