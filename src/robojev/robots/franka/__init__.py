"""A Franka arm (Panda / FR3) behind the robot protocol, over a small driver interface of our own.

The same policies as the sim and the real WidowX (robots/widowx_sim/skills.py, imported unchanged)
on `arm.FrankaArmIO`, which speaks to a `driver.FrankaDriver`: a fingertip point, a yaw, a gripper
width, stop, recover, park. Built and tested here against `fake_driver.FakeFrankaDriver`; the real
driver (franky over libfranka on the RT control box) is a later file, `franky_driver.py`, and
driver.py's docstring says how it maps. Every assumption about the hardware is marked "verify" in
the docstrings.

Modules: driver (the Protocol and the state record), fake_driver (the stand-in), arm (the ArmIO
adapter and `FrankaLimits`), server (FrankaRobotServer, the pre-flight gate).
"""
from robojev.robots.franka.arm import FRANKA_LIMITS, FRANKA_PARAMS, FrankaArmIO, FrankaLimits
from robojev.robots.franka.driver import FRANKA_IP, FrankaDriver, FrankaState
from robojev.robots.franka.fake_driver import FakeFrankaDriver
from robojev.robots.franka.server import FrankaRobotServer, Preflight, PreflightError

__all__ = ["FRANKA_IP", "FRANKA_LIMITS", "FRANKA_PARAMS", "FakeFrankaDriver", "FrankaArmIO", "FrankaDriver",
           "FrankaLimits", "FrankaRobotServer", "FrankaState", "Preflight", "PreflightError"]
