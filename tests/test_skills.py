"""The primitive library offers only what its preconditions allow, and its done conditions hold."""
import time

import numpy as np

from robojev.arm import ArmSnapshot
from robojev.brain import Brain
from robojev.config import DEFAULT as cfg
from robojev.perception.memory import Entity
from robojev.skills import offered, goal_for, resolve_place
from robojev.world import build_world

TABLE = -0.02


def ents():
    now = time.time()
    cup = Entity("C", np.array([0.37, -0.07, TABLE]), 0.11, 0.08, "white", now, now, seen_count=20, frozen=True)
    phone = Entity("E", np.array([0.30, 0.10, TABLE]), 0.024, 0.13, "black", now, now, seen_count=20, frozen=True)
    return [cup, phone]


def world(ee, gripper=0.04, holding=False, brain=None, gripper_goal=None):
    b = brain or Brain(cfg)
    snap = ArmSnapshot(time.time(), ee, gripper, setpoint=ee, status="live", holding=holding, gripper_goal=gripper_goal)
    return build_world(cfg, snap, ents(), lambda e, now: True, TABLE, "pick up the paper cup", [], b.state()), b


def keys(w, b):
    return {p.key for p in offered(cfg, w, b)}


def test_offer_progression():
    w, b = world((0.28, 0.0, 0.16))
    k = keys(w, b)
    assert "move_above:white cup-like object C" in k and "close_gripper" not in k and "lift" not in k
    # above the cup at hover height: descend offered, close not yet (too high)
    w, b = world((0.37, -0.07, 0.16))
    k = keys(w, b)
    assert "descend_to_grasp:white cup-like object C" in k and "close_gripper" not in k
    # at grasp height: close offered
    w, b = world((0.37, -0.07, 0.05))
    assert "close_gripper" in keys(w, b)
    # holding, low: lift offered, no move_above, open only near the table
    b = Brain(cfg); b.held = "white cup-like object C"; b.place = "left_of:black flat object E"
    w, b = world((0.37, -0.07, 0.05), gripper=0.02, holding=True, brain=b, gripper_goal=0.0)
    k = keys(w, b)
    assert "lift" in k and not any(x.startswith("move_above") for x in k) and "open_gripper" in k
    # holding at carry height: move_to_place offered, set_down_here offered, open not offered
    w, b = world((0.37, -0.07, 0.15), gripper=0.02, holding=True, brain=b, gripper_goal=0.0)
    k = keys(w, b)
    assert "move_to_place" in k and "set_down_here" in k and "open_gripper" not in k


def test_place_resolves_inside_box():
    w, b = world((0.28, 0.0, 0.16))
    xy = resolve_place(cfg, w, "left_of:black flat object E", None)
    assert xy is not None and cfg.workspace.contains((xy[0], xy[1], cfg.workspace.z[0] + 0.001))
    assert xy[1] > 0.10  # to the robot's left of the phone
    assert resolve_place(cfg, w, "unspecified", None) is None


def test_goal_done_conditions():
    b = Brain(cfg)
    w, _ = world((0.28, 0.0, 0.16), brain=b)
    b._start("move_above:white cup-like object C", w)
    goal, grip, done, fail, _ = goal_for(cfg, w, b, time.time())
    assert not done and abs(goal[0] - 0.37) < 1e-6 and abs(goal[1] + 0.07) < 1e-6
    w, _ = world(goal, brain=b)
    _, _, done, fail, _ = goal_for(cfg, w, b, time.time())
    assert done and fail is None
    # closing on nothing fails after the settle time
    b._start("close_gripper", w); b.prim_started_t -= 5
    w, _ = world(goal, gripper=0.0, holding=False, brain=b, gripper_goal=0.0)
    _, grip, done, fail, _ = goal_for(cfg, w, b, time.time())
    assert grip == 0.0 and done and fail == "closed on nothing"
