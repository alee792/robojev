"""Skills (code), the Limiter, and which skills are offered (preconditions, filtered in code).

A skill is a generator: one `next()` per tick while it runs; it returns a result string when done and
raises SkillFailed(reason) when it fails. Every motion step goes through the Limiter, which enforces
the active Orders plus two code rules that no layer can switch off: a hand within HAND_SLOW_CM caps
the pace at slow, and no step may move toward a hand closer than FLOOR_CM.
"""

from __future__ import annotations

import math

from . import plan as P
from .world import (ABOVE_CM, FINGER_CM, FLOOR_CM, HAND_SLOW_CM, LOW_CM, LOW_Z, TIP_CM, TRAVEL_Z, XY_SPEED,
                    Z_SPEED, dxy)

ARRIVED, MOVED = "arrived", "moved"
STUCK_TICKS = 50
EVASIONS = ("back_off", "evade_up")


class SkillFailed(Exception):
    pass


# ---------------------------------------------------------------- Limiter

class Limiter:
    """Code, not Jev. Always-on Orders (e.g. never touch the green block) come from the Plan; conditional
    Orders apply only while a layer (the Spotter) has switched them on (`active`)."""

    def __init__(self, plan: dict):
        self.forbidden = P.forbidden(plan)
        self.conditional = {o["id"]: o for o in plan["orders"] if o["kind"] == "conditional"}
        self.active: set[str] = set()
        self.refusals = 0
        self.floor_stops = 0
        self.last_refusal = ""

    def pace_cap(self, scene: dict, arm) -> str | None:
        h = scene.get("hand")
        if h and math.dist(arm.xyz, (h["x"], h["y"], h["z"])) < HAND_SLOW_CM:
            return "slow"   # code rule, regardless of any layer
        for oid in self.active:
            enf = self.conditional[oid]["enforce"]
            if enf.get("pace") == "slow":
                return "slow"
        return None

    def clearance(self) -> float:
        return max([self.conditional[o]["enforce"].get("hand_clearance_cm", 0) for o in self.active] or [0])

    def check_skill(self, key: str, scene: dict) -> str | None:
        name, _, arg = key.partition(":")
        if name in ("pick", "nudge_clear") and arg in self.forbidden:
            return self._refuse(f"order: {scene['blocks'].get(arg, {}).get('name', arg)} must not be touched")
        if name == "pick":
            f = P.crowded(scene, arg, self.forbidden)
            if f:
                return self._refuse(f"order: grasping it would touch the {scene['blocks'][f]['name']}")
        return None

    def check_step(self, scene: dict, cur, new, footprint: float) -> str | None:
        if new[2] < LOW_CM:
            for f in self.forbidden:
                e = scene["blocks"].get(f)
                if e and e["where"] != "gripper" and dxy(new, (e["x"], e["y"])) < footprint:
                    return self._refuse(f"order: would touch the {e['name']}")
        h = scene.get("hand")
        if h:
            hp = (h["x"], h["y"], h["z"])
            dn, dc = math.dist(new, hp), math.dist(cur, hp)
            if dn < dc and dn < FLOOR_CM:
                self.floor_stops += 1
                return self._refuse("code floor: a hand is too close")
            c = self.clearance()
            if c and dn < dc and dn < c:
                return self._refuse("order: keep clear of hands")
        return None

    def _refuse(self, why: str) -> str:
        self.refusals += 1
        self.last_refusal = why
        return why


# ---------------------------------------------------------------- motion

def step_arm(rt, target, xy_speed=None, z_speed=Z_SPEED, drag: str | None = None) -> str:
    """One tick of motion toward `target` (x, y, z), vetted by the Limiter."""
    arm = rt.world.arm
    cur = arm.xyz
    v = xy_speed or XY_SPEED[rt.pace()]
    dx, dy, dz = target[0] - cur[0], target[1] - cur[1], target[2] - cur[2]
    dh = math.hypot(dx, dy)
    s = 1.0 if dh <= v else v / dh
    new = (cur[0] + dx * s, cur[1] + dy * s, cur[2] + max(-z_speed, min(z_speed, dz)))
    why = rt.limiter.check_step(rt.scene, cur, new, arm.footprint)
    if why:
        return why
    arm.x, arm.y, arm.z = new
    if arm.holding:
        b = rt.world.blocks[arm.holding]
        b.x, b.y = arm.x, arm.y
    if drag:
        b = rt.world.blocks[drag]
        b.x, b.y = b.x + dx * s, b.y + dy * s
    return ARRIVED if math.dist(new, target) < 1e-6 else MOVED


def goto(rt, target, **kw):
    stuck = 0
    while True:
        r = step_arm(rt, target, **kw)
        stuck = 0 if r in (ARRIVED, MOVED) else stuck + 1
        if stuck > STUCK_TICKS:
            raise SkillFailed(f"blocked for {STUCK_TICKS // 10} s ({r})")
        yield
        if r == ARRIVED:
            return


def rise(rt, z=TRAVEL_Z):
    arm = rt.world.arm
    if abs(arm.z - z) > 1e-6:
        yield from goto(rt, (arm.x, arm.y, z))


def wait(n):
    for _ in range(n):
        yield


def _block(rt, bid) -> dict:
    e = rt.scene["blocks"].get(bid)
    if e is None:
        raise SkillFailed(f"cannot see {bid}")
    return e


# ---------------------------------------------------------------- skills

def sk_survey(rt, _):
    yield from rise(rt)
    yield from wait(5)
    return "done: looked over the table"


def sk_move_above(rt, bid):
    e = _block(rt, bid)
    rt.world.arm.target = bid
    tx, ty = e["x"], e["y"]
    yield from rise(rt)
    yield from goto(rt, (tx, ty, TRAVEL_Z))
    e = rt.scene["blocks"].get(bid)
    if e is None or dxy((tx, ty), (e["x"], e["y"])) > ABOVE_CM:
        raise SkillFailed(f"{rt.name(bid)} moved during the approach")
    return f"done: above {rt.name(bid)}"


def sk_pick(rt, bid):
    w, arm = rt.world, rt.world.arm
    arm.target, arm.dest = bid, None
    spot = (arm.x, arm.y)
    yield from goto(rt, (*spot, LOW_Z))
    yield from wait(2)   # closing
    got = next((b for b in w.blocks.values() if b.where != "gripper" and dxy((b.x, b.y), spot) <= ABOVE_CM), None)
    if got:
        arm.holding, got.where = got.id, "gripper"
    yield from goto(rt, (*spot, TRAVEL_Z))
    if not got:
        raise SkillFailed("nothing at the grasp spot")
    return f"done: holding {rt.name(got.id)}"


def sk_carry_to(rt, dest):
    arm = rt.world.arm
    if dest == "park":
        xy = rt.world.free_spot(near=(arm.x, 40.0))
    else:
        xy = P.dest_xy(rt.scene, dest)
    arm.dest = (dest, *xy)
    yield from rise(rt)
    yield from goto(rt, (*xy, TRAVEL_Z))
    return f"done: above {rt.dest_name(dest)}"


def sk_place(rt, _):
    arm = rt.world.arm
    yield from goto(rt, (arm.x, arm.y, LOW_Z + 1))
    return "done: lowered the block"


def sk_release(rt, _):
    w, arm = rt.world, rt.world.arm
    yield from wait(2)
    bid, arm.holding = arm.holding, None
    w.set_down(bid, arm.x, arm.y)
    return f"done: released {rt.name(bid)} ({rt.where_text(w.blocks[bid].where)})"


def sk_retreat(rt, _):
    yield from rise(rt)
    return "done: gripper up"


def sk_hold(rt, _):
    yield from wait(10)
    return "done: held still for 1 s"


def sk_back_off(rt, _):
    arm = rt.world.arm
    h = rt.scene.get("hand")
    ox, oy = (h["x"], h["y"]) if h else (arm.x, arm.y + 1)
    vx, vy = arm.x - ox, arm.y - oy
    n = math.hypot(vx, vy) or 1.0
    tx = min(95.0, max(5.0, arm.x + 10 * vx / n))
    ty = min(65.0, max(3.0, arm.y + 10 * vy / n))
    yield from goto(rt, (tx, ty, TRAVEL_Z + 7), xy_speed=2.0)
    return "done: backed off"


def sk_evade_up(rt, _):
    arm = rt.world.arm
    yield from goto(rt, (arm.x, arm.y, TRAVEL_Z + 13), z_speed=5.0)
    return "done: lifted clear"


def sk_nudge_clear(rt, bid):
    """Press one fingertip on the block's top and drag it 6 cm directly away from the forbidden block."""
    w, arm = rt.world, rt.world.arm
    arm.target = bid
    e = _block(rt, bid)
    f = P.crowded(rt.scene, bid, rt.limiter.forbidden)
    if f is None:
        return "done: nothing to clear"
    fe = rt.scene["blocks"][f]
    vx, vy = e["x"] - fe["x"], e["y"] - fe["y"]
    n = math.hypot(vx, vy) or 1.0
    ux, uy = vx / n, vy / n
    arm.footprint = TIP_CM
    try:
        yield from goto(rt, (e["x"], e["y"], TRAVEL_Z))
        yield from goto(rt, (e["x"], e["y"], 4.5))
        yield from goto(rt, (e["x"] + 6 * ux, e["y"] + 6 * uy, 4.5), xy_speed=1.0, drag=bid)
        yield from rise(rt)
    finally:
        arm.footprint = FINGER_CM
    w.blocks[bid].where = w.locate(w.blocks[bid].x, w.blocks[bid].y)
    return f"done: slid {rt.name(bid)} clear of the {fe['name']}"


SKILLS = {
    "survey": sk_survey, "move_above": sk_move_above, "pick": sk_pick, "carry_to": sk_carry_to,
    "place": sk_place, "release": sk_release, "retreat": sk_retreat, "hold": sk_hold,
    "back_off": sk_back_off, "evade_up": sk_evade_up, "nudge_clear": sk_nudge_clear,
}


# ---------------------------------------------------------------- preconditions -> offered options

def offered(rt) -> dict[str, str]:
    """Skill key -> description, for every skill whose preconditions hold now (perceived Scene)."""
    scene, arm, plan = rt.scene, rt.world.arm, rt.plan
    forb = rt.limiter.forbidden
    opts: dict[str, str] = {}
    if arm.high and not arm.holding:
        opts["survey"] = "look over the table again (after a failure, or if something seems off)"
        for bid in plan["bindings"]["blocks"]:
            e = scene["blocks"].get(bid)
            if e is None or bid in forb or e["where"] == "gripper":
                continue
            if P.above(arm, (e["x"], e["y"])):
                f = P.crowded(scene, bid, forb)
                if f:
                    opts[f"nudge_clear:{bid}"] = (f"slide {e['name']} (right below the gripper) a few cm away from the "
                                                  f"{scene['blocks'][f]['name']} with one fingertip, so it can be grasped without touching it")
                else:
                    opts[f"pick:{bid}"] = f"grasp {e['name']}, which is right below the gripper, and lift it"
            else:
                opts[f"move_above:{bid}"] = f"move the empty gripper above {e['name']} ({rt.where_text(e['where'], e)})"
    if arm.high and arm.holding:
        name = rt.name(arm.holding)
        for k, s in scene["slots"].items():
            if s["has"] is None:
                opts[f"carry_to:slot:{k}"] = f"carry {name} to above tray slot {k} (empty)"
        for b in scene["bins"]:
            opts[f"carry_to:bin:{b}"] = f"carry {name} to above the {b} bin"
        opts["carry_to:park"] = f"carry {name} to a free spot on the table, to set it aside for now"
        if arm.dest and P.above(arm, arm.dest[1:]) and (not arm.dest[0].startswith("slot:") or P.goal_free(scene, arm.dest[0], arm.holding)):
            opts["place"] = f"lower {name} into {rt.dest_name(arm.dest[0])}, which is right below"
    if not arm.high:
        if arm.holding:
            opts["release"] = f"open the gripper and let go of {rt.name(arm.holding)} here"
        opts["retreat"] = "lift the gripper back up to travel height"
    opts["hold"] = "stay still for a second"
    return opts
