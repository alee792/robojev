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
    x: tuple[float, float] = (0.18, 0.44)
    y: tuple[float, float] = (-0.18, 0.18)
    z: tuple[float, float] = (0.0, 0.19)   # floor: fingertips 4.5 cm above the table (side grasps run low)

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
    speed_levels: tuple[float, ...] = (0.03, 0.07, 0.12, 0.18)   # real run 8 at 3 cm/s took 36 s for a pick-and-place; lag was millimetres
    speed_names: tuple[str, ...] = ("very slow", "slow", "normal", "fast")
    hard_speed_cap: float = 0.18          # never exceeded whatever Jev says
    hover_heights: dict = field(default_factory=lambda: {"low": 0.10, "high": 0.16})  # above table plane
    safe_height: float = 0.18             # rise here on ladder step 2 / effort trip
    object_clearance: float = 0.05        # z floor while moving = tallest object + this
    hover_start: tuple[float, float] = (0.28, 0.0)  # xy where streaming begins after staging
    standoff: float = 0.08                # lateral offset for hover_position != directly_above
    back_off_distance: float = 0.08       # how far back_off retreats from the target, horizontally
    avoid_distance: float = 0.16          # keep the gripper this far (horizontally) from an avoided object
    evade_step: float = 0.10              # how far a directional evade moves, per latch
    evade_min_s: float = 0.6              # an evade holds at least this long before it can clear
    carry_height: float = 0.12            # above the table while carrying (base z 0.10: well inside the reachable envelope)
    grasp_fraction: float = 0.5           # grasp at this fraction of the object's height
    shift_distance: float = 0.15          # how far a 'move it to the left/right/away/closer' place is from the pickup spot
    place_gap: float = 0.04               # clearance between a placed object and its reference object
    gripper_settle_s: float = 0.7         # wait after a gripper command before judging the result
    primitive_timeout_s: float = 10.0     # a primitive that has not finished by then is reported failed
    stall_s: float = 3.0                  # a primitive whose EE has not moved for this long (and is not done) is reported failed
    down_orientation: tuple[float, float, float] = (0.0, 1.309, 0.0)  # 75 deg pitch: far larger reachable envelope than straight down (see reachability map)
    pitch_rate: float = 1.2               # rad/s: how fast the wrist pitch setpoint may change
    gripper_opening: float = 0.08         # m between the pads fully open (two 4 cm carriages)
    side_grasp_min_width: float = 0.045   # objects at least this wide are grasped from the side, not from above
    hover_pitch: float = 0.5              # staging/hover wrist pitch: the camera surveys the table instead of the patch under the fingers
    side_pitch: float = 0.5               # rad below level for a side grasp: fully level is unreachable low over the table (sim IK map), 0.5 is solid everywhere
    side_standoff: float = 0.06           # approach point: this far behind the object's near edge, wrist level
    side_grasp_height: float = 0.02       # tips 2 cm up: the pads (1.4-6.9 cm behind the tips, tilted) then meet a tapered cup where it is narrowest
    advance_push_n: float = 6.0           # F_x rise above the pre-advance baseline that means the fingers are shoving the object      # fingertips this far above the table for a side grasp (a tapered cup is narrowest low down)
    side_grasp_depth: float = -0.03       # tips this far short of the object's centre when closing: negative = past it, so the pads (1.4-6.9 cm behind the tips) straddle the centre
    side_pitch_advance: float = 0.35      # flatter wrist while sliding around the object: the pads span less height, so they meet the cup where it is narrower
    real_tick_hz: float = 20.0            # arm I/O thread rate (real)
    real_goal_time: float = 0.1           # per-command interpolation horizon (0.001..0.2 = linear)
    sim_physics_dt: float = 0.002


@dataclass(frozen=True)
class Safety:
    effort_trip_n: float = 40.0           # coarse backstop: |F_ext| deviation (N); the driver's estimate shifts ~25 N with motion direction
    effort_persist_ticks: int = 5
    lag_trip_m: float = 0.03              # primary obstacle detector: EE lagging the setpoint by this much (normal ~3 mm at 3 cm/s, ~1 cm at 12)
    lag_persist_ticks: int = 3
    effort_baseline_s: float = 1.0        # seconds of samples to establish the baseline after staging
    silence_hold_s: float = 0.5           # no fresh Jev answer for this long -> hold (freeze goal)
    silence_rise_s: float = 2.0           # ... this long -> rise to safe_height at slowest speed
    stale_answer_s: float = 0.4           # ignore answers whose state snapshot is older than this
    request_timeout_s: float = 0.7
    max_in_flight: int = 4


@dataclass(frozen=True)
class Thresholds:
    """Confidence gates. Choice confidence = (p_max - 1/n)/(1 - 1/n) (E4). Nouls have none."""
    target_p_max: float = 0.60            # dynamic option count -> gate on p_max, not confidence
    motion_conf: float = 0.40
    hover_position_conf: float = 0.40
    hover_height_conf: float = 0.40
    orders_violated_p: float = 0.90       # a logged signal; EVADE is the action channel, so this only brakes when near-certain
    avoid_p_max: float = 0.50             # dynamic option count -> gate on p_max
    evade_p_max: float = 0.50             # EVADE (Doom's DODGE): dynamic option count -> gate on p_max
    evade_clear_consecutive: int = 2
    next_p_max: float = 0.45              # next-primitive pick (dynamic option count)
    next_consecutive: int = 2             # non-safety primitives need this many consecutive picks
    next_interrupt_p: float = 0.65        # a safety pick (hold/back_off/rise_away) interrupts a running primitive only above this
    place_p_max: float = 0.50
    task_done_p: float = 0.80
    task_done_consecutive: int = 3
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
        ("very low, just above the table", 0.03), ("just above", 0.08), ("low", 0.16), ("high", 0.30), ("very high", float("inf")))


@dataclass(frozen=True)
class Orders:
    """Operator baseline standing orders, always present and listed before the user's runtime
    orders (Doom's +guide strategy text). Anthony's runtime orders are appended by the dashboard."""
    default: tuple[str, ...] = (
        "Never touch, bump or pass over a hand, an arm, or any object that just appeared or is moving; keep at least 10 cm from it and wait for it to leave.",
        "Never drop a held object anywhere but on the table.",
    )


@dataclass(frozen=True)
class Loop:
    evade_enabled: bool = True            # False: the evade / orders_violated channels are logged but never act (pickup-only demos)
    tick_hz: float = 10.0
    question_set: str = "v1"
    model: str = "jev-1.13.0"
    perception_hz: float = 10.0
    remembered_ttl_s: float = 20.0        # drop remembered entities unseen for this long (a hand that left is gone)
    out_of_view_s: float = 1.0            # unseen for this long -> status "out of view"
    # event-driven requests: tick at tick_hz, but only ask Jev when something material changed
    event_driven: bool = True             # False (CLI --clocked) = one request per tick, as before
    max_silence_s: float = 1.0            # ask anyway if this long has passed since the last sent request
    entity_move_m: float = 0.02           # an entity moving more than this is a material change


@dataclass(frozen=True)
class Config:
    workspace: Workspace = Workspace()
    motion: Motion = Motion()
    safety: Safety = Safety()
    thresholds: Thresholds = Thresholds()
    bands: Bands = Bands()
    loop: Loop = Loop()
    orders: Orders = Orders()
    table_z: float = 0.0                  # table plane in base frame; set from touch-off

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
