"""The real WidowX AI follower behind the robot protocol (spike D1, docs/harness-spikes.md).

The same policies as the sim (robots/widowx_sim/skills.py, imported unchanged) on v1's `RealArm`
(arm/real.py, wrapped, not edited) over the trossen_arm driver. Built and tested here against
`fake_driver.FakeTrossenDriver`; every assumption about the hardware is marked "verify on hardware"
in the docstrings and collected in docs/day-one.md.

Modules: orientation (yaw to the driver's angle-axis), world (typed-in block positions with the
server's own grasp/place bookkeeping; a perception stub), arm (the ArmIO adapter and the real-arm
tolerances), fake_driver (the driver stand-in), server (RealRobotServer).
"""
