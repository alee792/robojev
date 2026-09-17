"""Every tunable in one place: limits, bands, thresholds, hysteresis, timing.

Code owns safety. Nothing here is a suggestion to Jev; these are the numbers code enforces.
Units: metres, seconds, radians unless a name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned box in the arm base frame (+x forward, +y left, +z up) the EE may occupy."""
    # From the MuJoCo reachability map (2026-09-17, gripper pitched 75 deg): x 0.18-0.42 is solid
    # up to base z 0.20; straight-down (90 deg) only reaches z 0.15. Table is at base z ~ -0.02.
    x: tuple[float, float] = (0.18, 0.40)
    y: tuple[float, float] = (-0.18, 0.18)
    z: tuple[float, float] = (0.05, 0.19)

    def clamp(self, p):
        return (min(max(p[0], self.x[0]), self.x[1]),
                min(max(p[1], self.y[0]), self.y[1]),
                min(max(p[2], self.z[0]), self.z[1]))

    def contains(self, p, margin: float = 0.0) -> bool:
        return (self.x[0] + margin <= p[0] <= self.x[1] - margin
                and self.y[0] + margin <= p[1] <= self.y[1] - margin
                and self.z[0] + margin <= p[2] <= self.z[1] - margin)


@dataclass(frozen=True)
class Motion:
    # EE speed caps per Jev speed level, m/s. Index = score level.
    speed_levels: tuple[float, ...] = (0.01, 0.03, 0.06, 0.10)
    speed_names: tuple[str, ...] = ("very slow", "slow", "normal", "fast")
    hard_speed_cap: float = 0.10          # never exceeded whatever Jev says
    hover_heights: dict = field(default_factory=lambda: {"low": 0.10, "high": 0.16})  # above table plane
    safe_height: float = 0.18             # rise here on ladder step 2 / effort trip
    object_clearance: float = 0.05        # z floor while moving = tallest object + this
    hover_start: tuple[float, float] = (0.28, 0.0)  # xy where streaming begins after staging
    standoff: float = 0.08                # lateral offset for hover_position != directly_above
    back_off_distance: float = 0.08       # how far back_off retreats from the target, horizontally
    avoid_distance: float = 0.16          # keep the gripper this far (horizontally) from an avoided object
    down_orientation: tuple[float, float, float] = (0.0, 1.309, 0.0)  # 75 deg pitch: far larger reachable envelope than straight down (see reachability map)
    real_tick_hz: float = 20.0            # arm I/O thread rate (real)
    real_goal_time: float = 0.1           # per-command interpolation horizon (0.001..0.2 = linear)
    sim_physics_dt: float = 0.002


@dataclass(frozen=True)
class Safety:
    effort_trip_n: float = 40.0           # coarse backstop: |F_ext| deviation (N); the driver's estimate shifts ~25 N with motion direction
    effort_persist_ticks: int = 5
    lag_trip_m: float = 0.02              # primary obstacle detector: EE lagging the setpoint by this much (normal ~3 mm at 3 cm/s)
    lag_persist_ticks: int = 3
    effort_baseline_s: float = 1.0        # seconds of samples to establish the baseline after staging
    silence_hold_s: float = 0.5           # no fresh Jev answer for this long -> hold (freeze goal)
    silence_rise_s: float = 2.0           # ... this long -> rise to safe_height at slowest speed
    stale_answer_s: float = 0.4           # ignore answers whose state snapshot is older than this
    request_timeout_s: float = 0.7
    max_in_flight: int = 3


@dataclass(frozen=True)
class Thresholds:
    """Confidence gates. Choice confidence = (p_max - 1/n)/(1 - 1/n) (E4). Nouls have none."""
    target_p_max: float = 0.60            # dynamic option count -> gate on p_max, not confidence
    motion_conf: float = 0.40
    hover_position_conf: float = 0.40
    hover_height_conf: float = 0.40
    orders_violated_p: float = 0.70
    avoid_p_max: float = 0.50             # dynamic option count -> gate on p_max
    avoid_clear_consecutive: int = 3      # answers of no_avoidance_needed before an avoid latch clears
    # hysteresis: conservative picks latch on 1 answer; aggressive picks need N consecutive
    aggressive_consecutive: int = 3
    target_switch_consecutive: int = 3


@dataclass(frozen=True)
class Bands:
    """Distance bands relative to the gripper (4 cm wide), in metres. Rendered as words + numbers."""
    distance: tuple[tuple[str, float], ...] = (
        ("touching", 0.03), ("very close", 0.10), ("near", 0.25), ("mid-range", 0.50), ("far", float("inf")))
    height: tuple[tuple[str, float], ...] = (
        ("at table level", 0.03), ("just above", 0.08), ("low", 0.16), ("high", 0.30), ("very high", float("inf")))


@dataclass(frozen=True)
class Loop:
    tick_hz: float = 10.0
    question_set: str = "v0"
    model: str = "jev-1.13.0"
    perception_hz: float = 10.0
    remembered_ttl_s: float = 120.0       # drop remembered entities unseen for this long
    out_of_view_s: float = 1.0            # unseen for this long -> status "out of view"


@dataclass(frozen=True)
class Config:
    workspace: Workspace = Workspace()
    motion: Motion = Motion()
    safety: Safety = Safety()
    thresholds: Thresholds = Thresholds()
    bands: Bands = Bands()
    loop: Loop = Loop()
    table_z: float = 0.0                  # table plane in base frame; set from touch-off

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
