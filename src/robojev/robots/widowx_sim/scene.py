"""The sim scenes: the WidowX AI follower from trossen_arm_mujoco on a table with free-body blocks
and a tray of slots (K1), and the two-arm pair of docs/widowx-pair.md (K5): two followers facing
each other with a shared zone between them. Physics, not kinematic attachment: the fingers'
collision boxes hold the blocks.

Frames. Every coordinate in a SceneSpec, and everything the server reports, is in the COMMON frame:
arm_0's base frame as on the real arm, +x forward, +y left, +z up, metres, the table at z = TABLE_Z.
One arm: that is the only frame. Two arms: arm_1's base sits at (spacing, 0) turned 180 deg, so its
own +x points back at arm_0; `BasePose` converts between the common frame and an arm's own.

The pair's layout (docs/widowx-pair.md): each arm owns the half of the table nearest it; the strip
down the middle that both reach straight down is the shared zone (`SharedZone`), with hand-over
spots in it; blocks start in arm_0's half, the tray is in arm_1's half, and arm_0 has a staging row
in its half, so a block reaches the tray only by crossing the zone.

The numbers (measured on the model, 2026-10-11). Straight down, the arm grasps with 0 mm error from
x 0.10 to 0.42 from its base at grasp and travel height, so the pair's workspace box starts at 0.10.
The wrist-roll housing (link_5) stands 10.4 cm AHEAD of the fingertips along the arm's forward
direction, 17 cm above them, at every straight-down pose: two facing arms touch wrist to wrist with
their fingertips still 21 cm apart. So the smallest spacing at which one arm can work at the zone's
centre line while the other stands clear inside its own reach is 2 x (0.10 + server.KEEPOUT 0.26)
= 0.72 m; the default is 0.76 m with an 8 cm strip, which puts each arm's side line (where it waits
while the other holds the zone) at x 0.12 from its base, its tray or staging row just behind it at
0.11, and its blocks at x 0.17-0.21 (conformance.PAIR_X_RANGE). The doc's 0.6-0.7 m guess is too
close for this arm.

Two models in one spec: each follower file is loaded as its own MjSpec and attached under a frame
at its base pose with the prefix "<arm id>/", which MjSpec applies to every named element (bodies,
joints, actuators, sites, meshes, the gripper equality), so "arm_1/joint_0" and "arm_1/ee_site"
address the second arm. The one-arm scene keeps the file as the root spec, names unprefixed, as K1
built it: its physics and trial numbers are unchanged.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robojev.protocol import PlaceObs

SIM_DIR = Path(os.environ.get("TROSSEN_ARM_MUJOCO_DIR", Path.home() / "trossen_arm_mujoco"))
FOLLOWER_XML = SIM_DIR / "trossen_arm_mujoco/assets/wxai/wxai_follower.xml"
TABLE_Z = -0.02          # table surface in base frame (the base plate sits 2 cm proud, as in the bay)
ARM_0, ARM_1 = "arm_0", "arm_1"


@dataclass(frozen=True)
class BasePose:
    """Where an arm's base stands in the common frame: position and heading (yaw about +z). Converts
    points and yaws between the common frame and that arm's own base frame; z is shared (every base
    stands on the same table). `ORIGIN` is arm_0's, the identity."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def to_local(self, p) -> tuple[float, ...]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = p[0] - self.x, p[1] - self.y
        return (dx * c + dy * s, dy * c - dx * s) + tuple(float(v) for v in p[2:])

    def to_common(self, p) -> tuple[float, ...]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + p[0] * c - p[1] * s, self.y + p[0] * s + p[1] * c) + tuple(float(v) for v in p[2:])

    def yaw_to_local(self, yaw: float) -> float:
        return yaw - self.yaw

    def yaw_to_common(self, yaw: float) -> float:
        return yaw + self.yaw

    def rotate_to_local(self, v) -> tuple[float, ...]:
        """A direction (a velocity): rotated, not translated."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (v[0] * c + v[1] * s, v[1] * c - v[0] * s) + tuple(float(x) for x in v[2:])

    def box_to_common(self, box) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """An axis-aligned box ((x0,x1),(y0,y1),(z0,z1)) in the arm's frame as the axis-aligned box in
        the common frame (exact for the headings used here, 0 and 180 deg; the hull otherwise)."""
        corners = [self.to_common((x, y)) for x in box[0] for y in box[1]]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        return (min(xs), max(xs)), (min(ys), max(ys)), tuple(box[2])


ORIGIN = BasePose()


@dataclass
class Block:
    name: str
    xy: tuple[float, float]
    yaw: float = 0.0                  # rad about +z
    size: float = 0.04                # edge, m
    mass: float = 0.03                # kg (a 4 cm wooden cube)
    rgba: tuple = (0.85, 0.25, 0.2, 1)
    label: str | None = None          # what is read off it: a letter ("A") or a number; None = from the name's tail


def label_of(block: Block) -> str | None:
    """The label the world reports: the block's own, else the name's tail when it is one character
    or a number ("block_5" -> "5"), as K1 named things."""
    if block.label is not None:
        return block.label
    tail = block.name.rsplit("_", 1)[-1]
    return tail if len(tail) == 1 or tail.isdigit() else None


def letter_names(labels: Sequence[str | None]) -> list[str]:
    """How people call each block, in order: "the letter A"; duplicates "the second letter A", "the
    third letter A"; anything that is not a single letter keeps no name here (None)."""
    ordinal = ("", "second ", "third ", "fourth ", "fifth ", "sixth ")
    seen: dict[str, int] = {}
    out: list[str | None] = []
    for lab in labels:
        if lab is None or len(lab) != 1 or not lab.isalpha():
            out.append(None)
            continue
        k = seen.get(lab, 0)
        seen[lab] = k + 1
        out.append(f"the {ordinal[k] if k < len(ordinal) else f'{k + 1}th '}letter {lab}")
    return out


def block_name(block: Block, letter_name: str | None) -> str:
    """The name the world reports for a block: its letter name when it has one, else the K1 form
    ("block_5" -> "block 5")."""
    return letter_name if letter_name is not None else block.name.replace("_", " ")


# The letter layout (docs/showcase-demo.md: letter cubes in the arm's half, a row of positions in
# front). Two rows of up to 6 cubes at the slot pitch (8 cm: the jaws open to block + 3 cm), 8 cm
# apart in x, so any cube can be picked with its neighbours in place. Cubes "not on the table yet"
# (a letter the person adds later) are parked on the far corner of the table, out of every arm's
# reach, one spot each so they never overlap.
LETTER_ROWS = (0.19, 0.27)            # x of the two rows (one arm; the positions row sits at x 0.36)
LETTER_Y0, LETTER_PITCH, LETTERS_PER_ROW = 0.20, 0.08, 6
PARK = tuple((0.75, 0.45 - 0.08 * i) for i in range(6))     # off-workspace spots for parked cubes


def letter_spots(rows: tuple[float, ...] = LETTER_ROWS, y0: float = LETTER_Y0, pitch: float = LETTER_PITCH,
                 per_row: int = LETTERS_PER_ROW) -> list[tuple[float, float]]:
    """Every spot of the letter layout, row by row, left (+y) to right: where `letters` puts the
    cubes, and where a cube may be put back."""
    return [(rows[i // per_row], round(y0 - pitch * (i % per_row), 4)) for i in range(len(rows) * per_row)]


def letters(pool: str, parked: str = "", rows: tuple[float, ...] = LETTER_ROWS, y0: float = LETTER_Y0,
            pitch: float = LETTER_PITCH, per_row: int = LETTERS_PER_ROW, park: tuple[tuple[float, float], ...] = PARK,
            rgba: tuple = (0.9, 0.85, 0.6, 1)) -> list[Block]:
    """Letter cubes for the letters of `pool` (duplicates allowed: "OPENAAID"), laid out row by row,
    left (+y) to right, in the arm's half, plus `parked` letters on the park spots. Ids are
    `letter_<L>` for the first of a letter and `letter_<L>2`, `letter_<L>3`... for its duplicates
    (across pool and parked); `label` is the letter; the world names them "the letter A", "the
    second letter A" (letter_names)."""
    if len(pool) > len(rows) * per_row:
        raise ValueError(f"{len(pool)} letters do not fit {len(rows)} rows of {per_row}")
    if len(parked) > len(park):
        raise ValueError(f"{len(parked)} parked letters but {len(park)} park spots")
    counts: dict[str, int] = {}
    out: list[Block] = []

    def make(ch: str, xy: tuple[float, float]) -> Block:
        if len(ch) != 1 or not ch.isalpha():
            raise ValueError(f"a letter cube needs one letter, not {ch!r}")
        ch = ch.upper()
        k = counts.get(ch, 0) + 1
        counts[ch] = k
        return Block(f"letter_{ch}" + (str(k) if k > 1 else ""), xy, label=ch, rgba=rgba)
    for ch, xy in zip(pool, letter_spots(rows, y0, pitch, per_row)):
        out.append(make(ch, xy))
    for i, ch in enumerate(parked):
        out.append(make(ch, park[i]))
    return out


@dataclass
class Tray:
    """A row of square slots marked on the table (visual only: placing is judged by position). Ids
    are `<slot_id>_<i>` and the group `<id>`; the defaults give K1's `slot_0..` and `tray`."""
    origin: tuple[float, float] = (0.36, 0.12)     # centre of slot 0
    step: tuple[float, float] = (0.0, -0.08)       # slot i is at origin + i * step; 8 cm: the gripper opens to block + 3 cm, so 6 cm pitch put a finger on the neighbour
    n: int = 5
    slot: float = 0.05
    id: str = "tray"
    slot_id: str = "slot"
    label: str = "tray slot"                       # how a person calls one: "tray slot 3", "staging spot 1"
    suffix: str = ""                               # after the index: "position 2, counting from the left"

    def slot_xy(self, i: int) -> tuple[float, float]:
        return (self.origin[0] + i * self.step[0], self.origin[1] + i * self.step[1])

    def places(self) -> dict[str, PlaceObs]:
        """The tray as world-state places: `slot_i` (capacity 1, radius half the slot) and the group
        `tray` listing them, centred on them. Shared by the sim's world and the real server's."""
        out: dict[str, PlaceObs] = {}
        for i in range(self.n):
            x, y = self.slot_xy(i)
            tag = " (the leftmost slot)" if i == 0 else " (the rightmost slot)" if i == self.n - 1 else ""
            sid = f"{self.slot_id}_{i}"
            out[sid] = PlaceObs(sid, f"{self.label} {i}{self.suffix}{tag}", "slot", x, y, 1, radius=self.slot / 2)
        ids = tuple(out)
        cx = sum(out[i].x for i in ids) / len(ids)
        cy = sum(out[i].y for i in ids) / len(ids)
        out[self.id] = PlaceObs(self.id, f"the {self.id} (any free slot)", "group", cx, cy, self.n, ids)
        return out


def positions(n: int = 6, origin: tuple[float, float] = (0.36, 0.20), step: tuple[float, float] = (0.0, -0.08)) -> Tray:
    """The row of `n` positions a word is spelled into (docs/showcase-demo.md), left (+y) to right:
    the K1 tray with its ids kept (`slot_i`, group `tray`) and its slots named "position i, counting
    from the left"."""
    return Tray(origin, step, n, label="position", suffix=", counting from the left")


@dataclass(frozen=True)
class FreeSpots:
    """The "table" place: a free spot on the table, chosen when something is to be put down there
    (SimWorld.free_spot) from a grid of candidates, the one farthest from every block, at least
    `clearance` from each (a 4 cm cube's half-diagonal plus the open jaws' reach past it). With two
    arms the candidates are filtered to the asking arm's half. As a PlaceObs it is kind "table",
    capacity 0, centred on the grid."""
    xs: tuple[float, ...] = (0.16, 0.20, 0.24, 0.28, 0.32)
    ys: tuple[float, ...] = tuple(round(0.24 - 0.04 * i, 2) for i in range(13))
    clearance: float = 0.075
    id: str = "table"

    def candidates(self) -> list[tuple[float, float]]:
        return [(x, y) for x in self.xs for y in self.ys]

    def place(self) -> PlaceObs:
        return PlaceObs(self.id, "the table (a free spot)", "table", sum(self.xs) / len(self.xs), sum(self.ys) / len(self.ys), 0,
                        radius=max(self.xs[-1] - self.xs[0], self.ys[0] - self.ys[-1]) / 2)


@dataclass(frozen=True)
class SharedZone:
    """The strip across the table both arms reach, in the common frame: x0..x1 along the line between
    the bases, y0..y1 across it, with `spots` hand-over spots (`zone_i`, capacity 1) in a row down its
    middle. As a place it is `shared_zone` (kind "zone", capacity 0); PlaceObs describes a circle, so
    its radius is the strip's half width and the server judges "in the zone" from this geometry."""
    x0: float
    x1: float
    y0: float = -0.20
    y1: float = 0.20
    spots: int = 3
    spot_pitch: float = 0.10          # between hand-over spots; 10 cm keeps the pads clear of a neighbour
    spot_radius: float = 0.025
    id: str = "shared_zone"

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def spot_xy(self, i: int) -> tuple[float, float]:
        return (self.cx, (i - (self.spots - 1) / 2) * self.spot_pitch)

    def places(self, holder: str | None = None) -> dict[str, PlaceObs]:
        """The zone and its spots as places; the zone's name says who holds its lock."""
        out: dict[str, PlaceObs] = {}
        for i in range(self.spots):
            x, y = self.spot_xy(i)
            out[f"zone_{i}"] = PlaceObs(f"zone_{i}", f"shared-zone spot {i}", "slot", x, y, 1, radius=self.spot_radius)
        who = "free" if holder is None else f"held by {holder}"
        out[self.id] = PlaceObs(self.id, f"the shared zone ({who})", "zone", self.cx, (self.y0 + self.y1) / 2, 0,
                                tuple(f"zone_{i}" for i in range(self.spots)), radius=self.width / 2)
        return out


# The pair's default layout, common frame: the rows of spots sit just behind each arm's side line
# (x 0.12 from its base at the default spacing), the blocks in front of them (conformance.PAIR_X_RANGE).
PAIR_SPACING = 0.76
PAIR_ZONE_WIDTH = 0.08                # one block plus slack; the spots are on its centre line
PAIR_TRAY_X = 0.11                    # the tray's slots, from arm_1's base
PAIR_STAGE_X = 0.11                   # arm_0's staging spots, from arm_0's base
PAIR_TRAY_Y0, PAIR_TRAY_PITCH = -0.16, 0.08
PAIR_STAGE_Y = 0.10                   # the two staging spots at +-this


@dataclass
class SceneSpec:
    """What to build. One arm (the default) is K1's scene. `arms=(ARM_0, ARM_1)` is the pair: bases
    `spacing` apart facing each other on the x axis, a `SharedZone` of `zone_width` centred between
    them (None: no zone, no lock), and optionally arm_0's `stage` row beside the tray. `pair()` fills
    in the layout of docs/widowx-pair.md."""
    blocks: list[Block] = field(default_factory=list)
    tray: Tray | None = field(default_factory=Tray)
    arms: tuple[str, ...] = (ARM_0,)
    spacing: float = PAIR_SPACING
    zone_width: float | None = PAIR_ZONE_WIDTH
    stage: Tray | None = None
    free_spots: FreeSpots | None = None      # a "table" place for putting things down anywhere free (the spelling demo)

    @property
    def single(self) -> bool:
        return len(self.arms) == 1

    def bases(self) -> dict[str, BasePose]:
        """Each arm's base in the common frame: arm_0 at the origin, arm_1 `spacing` along +x turned
        to face it."""
        if len(self.arms) > 2:
            raise ValueError("a scene holds one arm or a facing pair")
        out = {self.arms[0]: ORIGIN}
        if len(self.arms) == 2:
            out[self.arms[1]] = BasePose(self.spacing, 0.0, math.pi)
        return out

    def zone(self) -> SharedZone | None:
        if self.single or self.zone_width is None:
            return None
        return SharedZone(self.spacing / 2 - self.zone_width / 2, self.spacing / 2 + self.zone_width / 2)

    def half(self, arm: str) -> tuple[float, float]:
        """The x range (common frame) of the arm's own half: from its base to the zone's near edge, or
        to the midline without a zone. (While the other arm holds the zone, only the part behind the
        arm's side line is usable; see server.KEEPOUT.)"""
        z = self.zone()
        mid = self.spacing / 2
        if arm == self.arms[0]:
            return (0.0, z.x0 if z else mid)
        return (z.x1 if z else mid, self.spacing)

    def places(self) -> dict[str, PlaceObs]:
        """Every fixed place: the tray, the stage, the zone and its spots (lock state not included)."""
        out: dict[str, PlaceObs] = {}
        for t in (self.tray, self.stage):
            if t is not None:
                out.update(t.places())
        z = self.zone()
        if z is not None:
            out.update(z.places())
        if self.free_spots is not None:
            out[self.free_spots.id] = self.free_spots.place()
        return out

    def trays(self) -> tuple[Tray, ...]:
        return tuple(t for t in (self.tray, self.stage) if t is not None)


def pair(blocks: list[Block] | None = None, spacing: float = PAIR_SPACING, zone_width: float | None = PAIR_ZONE_WIDTH,
         n_slots: int = 5) -> SceneSpec:
    """The pair scene of docs/widowx-pair.md: the tray (5 slots, 8 cm pitch, slot_0 at arm_1's left)
    in arm_1's half, a 2-spot staging row (`stage_0`, `stage_1`; group `stage`) in arm_0's half,
    `blocks` where given (in arm_0's half by convention) and the shared zone between."""
    tray = Tray((spacing - PAIR_TRAY_X, PAIR_TRAY_Y0), (0.0, PAIR_TRAY_PITCH), n_slots)
    stage = Tray((PAIR_STAGE_X, PAIR_STAGE_Y), (0.0, -2 * PAIR_STAGE_Y), 2, id="stage", slot_id="stage", label="staging spot")
    return SceneSpec(list(blocks or []), tray, (ARM_0, ARM_1), spacing, zone_width, stage)


def _add_table(s, spec: SceneSpec) -> None:
    """The table: K1's 1.2 m square in front of the base, or one spanning both bases. Block-table
    friction 0.5 (wood on wood): a pushed cube slides. MuJoCo takes the larger of a pair's frictions,
    so at the old 1.0 the pads' push, 2.9 cm up a 4 cm cube, rolled it over (K4 probe); the grip is
    unaffected, the pads' 1.5 still wins there."""
    import mujoco
    half = 0.6 if spec.single else spec.spacing / 2 + 0.3
    cx = 0.3 if spec.single else spec.spacing / 2
    s.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, 0.6, 0.01], pos=[cx, 0, TABLE_Z - 0.01],
                         rgba=[0.75, 0.6, 0.4, 1], friction=[0.5, 0.01, 0.001])


def _add_marks(s, spec: SceneSpec) -> None:
    """Visual-only markers: slots, staging spots, the zone strip and its spots."""
    import mujoco
    w = s.worldbody
    for t in spec.trays():
        for i in range(t.n):
            x, y = t.slot_xy(i)
            w.add_geom(name=f"{t.slot_id}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t.slot / 2, t.slot / 2, 0.0005],
                       pos=[x, y, TABLE_Z + 0.0005], rgba=[0.2, 0.2, 0.25, 1], contype=0, conaffinity=0)
    z = spec.zone()
    if z is not None:
        w.add_geom(name=z.id, type=mujoco.mjtGeom.mjGEOM_BOX, size=[z.width / 2, (z.y1 - z.y0) / 2, 0.0003],
                   pos=[z.cx, (z.y0 + z.y1) / 2, TABLE_Z + 0.0003], rgba=[0.9, 0.8, 0.3, 0.5], contype=0, conaffinity=0)
        for i in range(z.spots):
            x, y = z.spot_xy(i)
            w.add_geom(name=f"zone_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[z.spot_radius, 0.0005],
                       pos=[x, y, TABLE_Z + 0.0005], rgba=[0.3, 0.3, 0.2, 1], contype=0, conaffinity=0)


def _add_blocks(s, spec: SceneSpec) -> None:
    import mujoco
    for b in spec.blocks:
        h = b.size / 2
        body = s.worldbody.add_body(name=b.name, pos=[b.xy[0], b.xy[1], TABLE_Z + h],
                                    quat=[np.cos(b.yaw / 2), 0, 0, np.sin(b.yaw / 2)])
        body.add_freejoint(name=b.name + "_free")
        body.add_geom(name=b.name + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=b.mass,
                      rgba=list(b.rgba), friction=[0.5, 0.01, 0.001], condim=4)


def _attach_arm(s, arm: str, base: BasePose) -> None:
    """A second copy of the follower under its own frame and prefix "<arm>/". Its keyframe goes
    (keyframes do not attach); defaults, actuator gains and gravity compensation carry over."""
    import mujoco
    child = mujoco.MjSpec.from_file(str(FOLLOWER_XML))
    for k in list(child.keys):
        child.delete(k)
    frame = s.worldbody.add_frame(pos=[base.x, base.y, 0.0], quat=[math.cos(base.yaw / 2), 0, 0, math.sin(base.yaw / 2)])
    frame.attach_body(child.body("base_link"), f"{arm}/", "")


def prefix_for(spec: SceneSpec, arm: str) -> str:
    """How the arm's model elements are named: "" for the one-arm scene, "<arm>/" in the pair."""
    return "" if spec.single else f"{arm}/"


def build(spec: SceneSpec):
    """Compile the scene. Returns the MjModel."""
    import mujoco
    if not FOLLOWER_XML.exists():
        raise FileNotFoundError(f"{FOLLOWER_XML} (set TROSSEN_ARM_MUJOCO_DIR to a trossen_arm_mujoco clone)")
    if spec.single:
        s = mujoco.MjSpec.from_file(str(FOLLOWER_XML))
        s.modelname = "robojev_k1"
    else:
        s = mujoco.MjSpec()
        s.modelname = "robojev_k5_pair"
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST       # as the follower file sets it
        for arm, base in spec.bases().items():
            _attach_arm(s, arm, base)
    s.option.timestep = 0.002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10.0           # stiffer friction: blocks don't creep out of the grip
    w = s.worldbody
    w.add_light(pos=[spec.spacing / 2 if not spec.single else 0.3, 0, 1.2], dir=[0, 0, -1],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    _add_table(s, spec)
    _add_marks(s, spec)
    _add_blocks(s, spec)
    # finger pads: high friction, torsional resistance, so a held block doesn't spin or slip
    for g in s.geoms:
        if g.parent.name in ("left_carriage_link", "right_carriage_link") or "carriage" in (g.parent.name or ""):
            if g.contype or g.conaffinity:
                g.friction = [1.5, 0.02, 0.002]
                g.condim = 4
    w.add_camera(name="overhead", pos=[0.33 if spec.single else spec.spacing / 2, 0.0, 0.80], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    return s.compile()
