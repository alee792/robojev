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
    cup = Entity("C", np.array([0.37, -0.07, TABLE]), 0.11, 0.035, "white", now, now, seen_count=20, frozen=True)   # narrow: top-down grasp
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
    assert grip < 0.036 and done and fail == "closed on nothing"


def test_side_grasp_progression():
    """A cup too wide for the fingers is approached level with the table, entered from the side,
    grasped low, and left by backing out before rising."""
    import robojev.skills as sk
    wide = Entity("C", np.array([0.37, -0.07, TABLE]), 0.11, 0.08, "white", time.time(), time.time(), seen_count=20, frozen=True)
    def w_at(ee, pitch, gripper=0.04, holding=False, b=None):
        b = b or Brain(cfg)
        snap = ArmSnapshot(time.time(), ee, gripper, setpoint=ee, status="live", holding=holding, pitch=pitch)
        return build_world(cfg, snap, [wide], lambda e, now: True, TABLE, "pick up the paper cup", [], b.state()), b
    down = cfg.motion.down_orientation[1]
    w, b = w_at((0.28, 0.0, 0.16), down)
    k = keys(w, b)
    assert "approach_side:white cup-like object C" in k and not any(x.startswith("move_above") for x in k)
    assert "advance_to_grasp:white cup-like object C" not in k
    b.prim, b.prim_subject, b.prim_status, b.prim_started_t = "approach_side", "white cup-like object C", "running", time.time()
    goal, grip, done, fail, _ = goal_for(cfg, w, b, time.time())
    assert b.pitch == cfg.motion.side_pitch and abs(goal[0] - (0.37 - 0.04 - cfg.motion.side_standoff)) < 1e-6 and abs(goal[2] - (TABLE + cfg.motion.side_grasp_height)) < 1e-6
    # at the approach point, level: advance is offered, close is not
    w, b = w_at(goal, cfg.motion.side_pitch, b=b)
    k = keys(w, b)
    assert "advance_to_grasp:white cup-like object C" in k and "close_gripper" not in k
    b.prim, b.prim_status = "advance_to_grasp", "running"
    goal, grip, done, fail, _ = goal_for(cfg, w, b, time.time())
    assert abs(goal[0] - (0.37 - cfg.motion.side_grasp_depth)) < 1e-6
    # around the cup: close is offered; after closing, lift is offered and retreat backs out first
    w, b = w_at(goal, cfg.motion.side_pitch, b=b)
    assert "close_gripper" in keys(w, b)
    b.grasp_dz = goal[2] - TABLE
    w, b = w_at(goal, cfg.motion.side_pitch, gripper=0.03, holding=True, b=b)
    assert "lift" in keys(w, b)
    w, b = w_at(goal, cfg.motion.side_pitch, gripper=0.04, holding=False, b=b)
    b.prim, b.prim_status = "retreat", "running"
    g2, _, done, _, reason = goal_for(cfg, w, b, time.time())
    assert not done and g2[0] < goal[0] - 0.05 and "backing out" in reason
