"""The driver surface the Franka server is built on: the least a Panda or FR3 has to offer for the
skill policies to run, in metres, seconds and the arm's base frame. `FrankaDriver` is a Protocol;
`fake_driver.FakeFrankaDriver` implements it here with no arm, and a real driver module is a LATER
FILE (`franky_driver.py`, see below), written once we know the control computer and the library.

Why so small: the policies (robots/widowx_sim/skills.py) only ever stream a Cartesian goal with a
straight-down tool orientation and a yaw, open or close the gripper to a width, stop, and read back
where the fingertips are and how wide they are. Everything the arm does at 1 kHz stays in the
driver; the server ticks at 50 Hz and never sees a joint.

Frame and conventions
  - Positions are the point between the fingertips (the sim's `ee_site`, the WidowX adapter's EE):
    the policies put that point `tip_height` above the table to grasp. For the Franka Hand that is
    the flange frame pushed along the tool z axis by the hand's length plus the fingertip offset
    (about 0.1034 m flange-to-hand plus the fingers; the real driver measures it). Base frame:
    panda_link0, +x forward, +z up.
  - `yaw` is the direction of the closing axis (fingertip to fingertip) in the base xy plane, rad,
    as `skills.ArmIO.ee_yaw` defines it. The tool points straight down at every setpoint: the policies
    never choose a pitch, exactly as on the WidowX. The driver turns (xyz, yaw) into a full pose.
  - Widths are between the finger pads: Franka Hand 0-0.08 m.

How franky maps onto it (the likely real driver; libfranka directly or panda-py would map the same way)
  - `set_target(xyz, yaw, speed)`: `Robot.move(CartesianMotion(Affine(xyz, quat_down(yaw))),
    asynchronous=True)` with `relative_dynamics_factor` chosen so the Cartesian speed stays under
    `speed`. The server calls it every policy tick (50 Hz) with the SAME goal while a phase runs,
    so the driver must treat a repeat as a no-op and a changed goal as a new (preempting) motion:
    franky's asynchronous move replaces the running one and smooths the transition in its own
    online trajectory generator (Ruckig). Our 50 Hz setpoints are therefore targets the driver
    tracks, not a 1 kHz stream: the 1 kHz loop lives in libfranka on the RT Linux control box, and
    this process never has to meet its deadline.
  - `set_gripper(width, force)`: `Gripper.move_async(width, speed)` when `force` is None (an
    opening), `Gripper.grasp_async(width, speed, force, epsilon_inner, epsilon_outer)` when it is set
    (a close on something). The policies command a close 8 mm NARROWER than the block (the squeeze),
    so `epsilon_outer` must be at least 0.01 m or libfranka reports the grasp failed although the
    block is held; the driver should pass wide epsilons and let `FrankaState.grasped` mean "the
    fingers stopped short of the width on something".
  - `stop()`: `Robot.stop()` ends the asynchronous motion; libfranka decelerates within one control
    step's limits (a few ms), the gripper command is untouched so the grip stays. Nothing here can
    stop faster than the arm's own limits, and nothing here is the e-stop: the external activation
    device in the operator's hand is.
  - `recover()`: `Robot.recover_from_errors()`. A collision or a Cartesian reflex puts the arm in an
    error state in which every motion command raises until recovered; the arm does not move itself.
    `read()` reports that state as `error`, the server turns it into a `safety_trip`, and only the
    brain's resume() calls `recover()`. A reflex that fires again on resume trips again.
  - `park()`: a joint motion to a safe pose (franky's `JointMotion` to the "ready" configuration),
    blocking; used by STOP, from its own thread. The grip is kept.
  - `read()`: `Robot.state()` for the O_T_EE pose (fingertip point and closing-axis yaw derived from
    it), `Gripper.width`/`is_grasped`, the motion's `is_running`, and the current error, if any.

What libfranka can and cannot do, so the layers above do not assume more
  - It stops a motion within one control step and holds position; it does not back off, open the
    fingers or leave the error state by itself.
  - A collision reflex or a joint/Cartesian limit violation stops the arm and latches an error; the
    arm accepts no motion until `recover_from_errors`, and the reflex thresholds are the robot's
    own, set in Desk or via `set_collision_behavior`, not here.
  - The gripper is a separate device with its own, slower loop (a move takes ~0.5 s across its
    range); its width readback lags a little and a grasp reports success only inside its epsilons.
  - FCI must be enabled in Desk and the joints unlocked, by a person; the activation device must be
    held; the control computer must run the RT kernel and hold 1 kHz, or libfranka aborts with a
    communication error mid-motion. None of that is observable from here: `server.preflight()`
    records that a person checked it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

FRANKA_IP = "172.16.0.2"          # the controller's default address on the FCI link (Desk is at the same one)
REAL_DRIVER_MODULE = "robojev.robots.franka.franky_driver"
REAL_DRIVER_CLASS = "FrankyDriver"


@dataclass(frozen=True)
class FrankaState:
    """One read of the arm. `t` is on the driver's clock (seconds); `error` is None or the arm's own
    words for why it will not move ("cartesian_reflex: collision detected on joint 4")."""
    t: float
    xyz: tuple[float, float, float]      # fingertip point, base frame, m
    yaw: float                            # closing-axis direction in the xy plane, rad
    width: float                          # between the finger pads, m
    moving: bool                          # the arm's motion is still running (the gripper has its own flag)
    error: str | None = None
    grasped: bool = False                 # the fingers stopped short of their commanded width on something
    gripper_moving: bool = False          # the fingers are still moving toward their width


@runtime_checkable
class FrankaDriver(Protocol):
    """See the module docstring. Every call returns fast except `park()`; none raises into the policy
    loop for an arm in error: a command the arm refuses is dropped and the refusal shows up in
    `read().error`."""

    def read(self) -> FrankaState: ...

    def set_target(self, xyz: tuple[float, float, float], yaw: float, speed: float) -> None:
        """Track this fingertip point and closing-axis yaw, tool straight down, no faster than
        `speed` m/s. Non-blocking; repeated with the same goal every tick while a phase runs."""

    def set_gripper(self, width: float, force: float | None = None) -> None:
        """Move the fingers to `width`; with `force` (N) close on whatever is between them. Non-blocking;
        repeated with the same width every tick while a phase runs, so a repeat must be a no-op."""

    def stop(self) -> None:
        """Stop the arm's motion where it is, keep the grip. Idempotent."""

    def recover(self) -> None:
        """Clear a reflex or collision error so the arm accepts motion again."""

    def park(self) -> None:
        """A safe joint pose, grip kept. May block for the move (STOP calls it from its own thread)."""

    def close(self) -> None:
        """Release the connection. The arm stays where it is."""
