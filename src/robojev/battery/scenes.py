"""An episode's table on each world: the letter cubes in the arm's half (scene.letters), the row of
positions in front (scene.positions; ids `slot_i` kept, named "position i, counting from the left"),
a "table" place for putting a letter down anywhere free, and, with two arms, the pair layout of
docs/widowx-pair.md: letters in arm_0's half, positions in arm_1's, the shared zone between.

    stub_server(ep)   the physics-free stub (robojev.robots.stub), objects and places built here
    sim_server(ep)    the MuJoCo WidowX (robojev.robots.widowx_sim), realtime, one arm
"""
from __future__ import annotations

from dataclasses import dataclass

from robojev.protocol import ObjectObs, PlaceObs
from robojev.robots import stub as S
from robojev.robots.widowx_sim import scene

from .generate import Episode

PICK_S = {"stub": 0.8, "sim": 7.5}          # one pick_and_place's worth of time on each world (K2: 7.3 s in sim)
PAIR_LETTER_ROWS = (0.14, 0.22)              # the stub pair: letters in arm_0's half (x 0.10-0.30 in the table frame)
PAIR_POSITIONS_X = 0.54                      # ... positions in arm_1's half (x 0.30-0.60), as stub.pair_scene's tray
STUB_TABLE = {1: (0.22, -0.22), 2: (0.20, -0.22)}    # the stub's "table" place centre per arm count


@dataclass(frozen=True)
class Layout:
    blocks: list[scene.Block]
    tray: scene.Tray
    parked: frozenset[str]                   # ids of cubes not on the table at the start
    arms: int
    spots: list[tuple[float, float]]         # every letter spot of the layout, occupied or not

    @property
    def positions(self) -> list[str]:
        """Position ids, left to right."""
        return [f"{self.tray.slot_id}_{i}" for i in range(self.tray.n)]


def layout(ep: Episode) -> Layout:
    if ep.arms == 1:
        rows = scene.LETTER_ROWS
        tray = scene.positions(ep.positions)
    elif ep.arms == 2:
        rows = PAIR_LETTER_ROWS
        tray = scene.positions(ep.positions, origin=(PAIR_POSITIONS_X, scene.LETTER_Y0))
    else:
        raise ValueError(f"{ep.arms} arms: the layouts are for one arm or the pair")
    blocks = scene.letters(ep.pool, ep.parked, rows=rows)
    parked = frozenset(b.name for b in blocks[len(ep.pool):])
    return Layout(blocks, tray, parked, ep.arms, scene.letter_spots(rows))


def stub_objects(lay: Layout) -> list[ObjectObs]:
    labels = [scene.label_of(b) for b in lay.blocks]
    return [ObjectObs(b.name, scene.block_name(b, nm), b.xy[0], b.xy[1], 0.0, "table", label=lab, size=b.size)
            for b, lab, nm in zip(lay.blocks, labels, scene.letter_names(labels))]


def stub_places(lay: Layout) -> list[PlaceObs]:
    places = list(lay.tray.places().values())
    tx, ty = STUB_TABLE[lay.arms]
    places.append(PlaceObs("table", "the table (a free spot)", "table", tx, ty, 0, radius=0.30))
    places.append(PlaceObs("person", "the person", "person", 0.55 if lay.arms == 1 else 0.35, 0.0 if lay.arms == 1 else -0.45, 0))
    if lay.arms == 2:
        zone = [PlaceObs(f"zone_slot_{i}", f"shared zone slot {i}", "slot", S.ZONE_X, y, 1) for i, y in ((1, 0.05), (2, -0.05))]
        places += zone
        places.append(PlaceObs("shared_zone", "the shared zone (both arms reach it; one arm inside at a time)", "zone",
                               S.ZONE_X, 0.0, 0, tuple(p.id for p in zone), radius=0.08))
    return places


def stub_server(ep: Episode, pick_s: float = PICK_S["stub"], **kw) -> S.StubRobotServer:
    """The stub for `ep`, not yet opened."""
    lay = layout(ep)
    manifest = S.widowx_like() if lay.arms == 1 else S.widowx_pair()
    return S.StubRobotServer(manifest, stub_objects(lay), stub_places(lay), durations={"pick_and_place": pick_s}, **kw)


def sim_server(ep: Episode, recorder=None, realtime: bool = True):
    """The MuJoCo WidowX for `ep` (one arm), not yet opened. The scene has a FreeSpots "table"
    place, so a letter can be cleared to anywhere free."""
    from robojev.robots.widowx_sim.server import SimRobotServer
    lay = layout(ep)
    if lay.arms != 1:
        raise ValueError("the physics battery runs one arm; the pair runs on the stub (docs/battery.md)")
    spec = scene.SceneSpec(blocks=list(lay.blocks), tray=lay.tray, free_spots=scene.FreeSpots())
    return SimRobotServer(spec, realtime=realtime, recorder=recorder)
