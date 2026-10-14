"""The real Franka driver's logic, with a stand-in for franky (no arm, no libfranka): the straight-down
pose and its read-back, the no-op on a repeated goal and the preemption on a new one, the gripper
mapping with wide epsilons, and franky exceptions landing in `FrankaState.error` instead of the policy
loop. When the real franky is installed (project extra `franka`) the pose maths is checked against
its Affine too."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from robojev.robots.franka.franky_driver import READY_Q, FrankyDriver, down_matrix, read_pose


# ---------------------------------------------------------------- a stand-in franky


class _Affine:
    def __init__(self, matrix=None, quaternion=None):
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)

    @property
    def translation(self):
        return self.matrix[:3, 3]


class _Errors:
    cartesian_reflex = False
    joint_reflex = False

    def __str__(self):
        return "[]"


class _Mode:
    def __init__(self, name):
        self.name = name


class _State:
    def __init__(self, matrix, errors=None, mode="Idle"):
        self.O_T_EE = _Affine(matrix)
        self.current_errors = errors or _Errors()
        self.robot_mode = _Mode(mode)


class _Future:
    def __init__(self, done=True):
        self.done = done

    def wait(self, timeout=None):
        return self.done


class _Robot:
    def __init__(self, ip, **kw):
        self.ip, self.kw, self.moves, self.stopped, self.recovered = ip, kw, [], 0, 0
        self.matrix = down_matrix((0.3, 0.0, 0.5), 0.0)
        self.errors, self.mode, self.refuse = _Errors(), "Idle", None
        self.in_control = False

    translation_velocity_limit = 1.7

    @property
    def state(self):
        return _State(self.matrix, self.errors, self.mode)

    @property
    def has_errors(self):
        return any(getattr(self.errors, n) is True for n in ("cartesian_reflex", "joint_reflex"))

    @property
    def is_in_control(self):
        return self.in_control

    def move(self, motion, asynchronous=False):
        if self.refuse:
            raise RuntimeError(self.refuse)
        self.moves.append((motion, asynchronous))
        self.in_control = True

    def stop(self):
        self.stopped += 1
        self.in_control = False

    def recover_from_errors(self):
        self.recovered += 1
        self.errors, self.mode = _Errors(), "Idle"
        return True


class _Gripper:
    def __init__(self, ip):
        self.calls, self.width, self.is_grasped, self.homed = [], 0.08, False, 0

    def homing(self):
        self.homed += 1
        return True

    def move_async(self, width, speed):
        self.calls.append(("move", width, speed))
        return _Future()

    def grasp_async(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
        self.calls.append(("grasp", width, speed, force, epsilon_inner, epsilon_outer))
        return _Future()


class _Motion:
    def __init__(self, *a, **kw):
        self.args, self.kw = a, kw


def stand_in():
    fk = types.ModuleType("franky_stand_in")
    fk.Robot, fk.Gripper, fk.Affine = _Robot, _Gripper, _Affine
    fk.CartesianMotion = fk.JointMotion = fk.CartesianState = fk.RobotPose = fk.JointState = _Motion
    fk.RelativeDynamicsFactor = lambda *a: tuple(a)
    return fk


def driver(**kw) -> FrankyDriver:
    return FrankyDriver("172.16.0.2", franky_module=stand_in(), **kw)


# ---------------------------------------------------------------- the pose maths


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, math.pi / 2, 2.9])
def test_the_down_pose_reads_back_as_the_same_fingertip_point_and_yaw(yaw):
    m = down_matrix((0.4, -0.1, 0.02), yaw, tip_offset=0.01)
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)       # a proper rotation
    assert np.allclose(r[:, 2], [0, 0, -1])                                              # tool straight down
    tip, got = read_pose(m, tip_offset=0.01)
    assert np.allclose(tip, (0.4, -0.1, 0.02)) and math.isclose(math.atan2(math.sin(got - yaw), math.cos(got - yaw)), 0.0, abs_tol=1e-9)
    assert math.isclose(m[2, 3], 0.03)                                                   # the EE origin sits above the tip


def test_the_closing_axis_knob_moves_the_yaw_to_the_other_tool_axis():
    m = down_matrix((0.4, 0.0, 0.1), 0.5, closing_axis="x")
    r = m[:3, :3]
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r[:, 0], [math.cos(0.5), math.sin(0.5), 0])
    assert math.isclose(read_pose(m, closing_axis="x")[1], 0.5)
    with pytest.raises(ValueError):
        down_matrix((0, 0, 0), 0.0, closing_axis="z")


def test_the_pose_maths_agrees_with_the_real_franky_affine_when_installed():
    franky = pytest.importorskip("franky")
    m = down_matrix((0.35, 0.05, 0.1), 0.7)
    a = franky.Affine(m)
    assert np.allclose(a.matrix, m)
    q = a.quaternion                                            # [x, y, z, w]; a straight-down tool has |x|^2+|y|^2 = 1
    assert math.isclose(q[0] ** 2 + q[1] ** 2, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------- the driver's rules


def test_a_repeated_goal_is_a_no_op_and_a_new_goal_preempts():
    d = driver()
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)
    d.set_target((0.4, 0.0003, 0.1), 0.002, 0.15)              # within the same-goal tolerance
    assert len(d.robot.moves) == 1 and d.robot.moves[0][1] is True
    d.set_target((0.4, 0.0, 0.05), 0.0, 0.15)
    assert len(d.robot.moves) == 2
    factor = d.robot.moves[1][0].kw["relative_dynamics_factor"]
    assert math.isclose(factor[0], 0.15 / 1.7) and factor[1:] == (0.2, 0.2)
    d.set_target((0.4, 0.0, 0.2), 0.0, 5.0)                     # asked faster than the cap: capped
    assert math.isclose(d.robot.moves[2][0].kw["relative_dynamics_factor"][0], 0.25 / 1.7)
    assert d.read().moving is True


def test_the_gripper_opens_with_move_and_closes_with_a_wide_epsilon_grasp():
    d = driver()
    d.set_gripper(0.07)
    d.set_gripper(0.07)
    d.set_gripper(0.032, force=20.0)
    assert d.gripper.calls == [("move", 0.07, 0.1), ("grasp", 0.032, 0.1, 20.0, 0.04, 0.04)]
    d.gripper.is_grasped, d.gripper.width = True, 0.04
    s = d.read()
    assert s.grasped is True and math.isclose(s.width, 0.04) and s.gripper_moving is False
    d.set_gripper(0.07)                                         # an opening: never "grasped"
    d.gripper.is_grasped = True
    assert d.read().grasped is False


def test_stop_forgets_the_goal_so_the_next_target_starts_a_motion():
    d = driver()
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)
    d.stop()
    assert d.robot.stopped == 1 and d.read().moving is False
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)
    assert len(d.robot.moves) == 2


def test_a_reflex_shows_as_the_error_until_recovered():
    d = driver()
    d.robot.errors.cartesian_reflex = True
    assert d.read().error == "cartesian_reflex"
    d.robot.errors.cartesian_reflex = False
    d.robot.mode = "UserStopped"
    assert d.read().error.startswith("user stopped")
    d.recover()
    assert d.robot.recovered == 1 and d.read().error is None


def test_a_refused_move_lands_in_the_error_not_in_the_caller():
    d = driver()
    d.robot.refuse = "libfranka: command rejected, robot in reflex mode"
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)                   # does not raise
    assert d.read().error.startswith("move refused: RuntimeError")
    d.robot.refuse = None
    d.recover()
    d.set_target((0.4, 0.0, 0.1), 0.0, 0.15)
    assert d.read().error is None and len(d.robot.moves) == 1


def test_park_is_a_blocking_joint_motion_at_a_low_factor_and_read_survives_a_lost_connection():
    d = driver()
    d.park()
    motion, asynchronous = d.robot.moves[-1]
    assert asynchronous is False and motion.kw["relative_dynamics_factor"] == (0.1,) and motion.args[0].args[0] == list(READY_Q)

    class Dead(_Robot):
        @property
        def state(self):
            raise ConnectionError("lost")
    d.robot = Dead("x")
    assert d.read().error == "driver: ConnectionError: lost"


def test_the_connection_sets_the_collision_thresholds_and_can_home_the_gripper():
    d = driver(torque_threshold=15.0, force_threshold=12.0, home_gripper=True)
    assert d.robot.kw == {"default_torque_threshold": 15.0, "default_force_threshold": 12.0}
    assert d.gripper.homed == 1
