from robojev.arm import Mover, EffortWatchdog
from robojev.config import Workspace


def test_mover_never_exceeds_cap_and_clamps():
    m = Mover(Workspace(), hard_speed_cap=0.1)
    m.init_at((0.25, 0.0, 0.2))
    m.set_goal((0.9, 0.0, 0.2), speed_cap=0.5)  # outside box, cap above hard cap
    assert m.goal == (0.42, 0.0, 0.2)
    sp = m.step(0.1)
    assert abs(sp[0] - 0.26) < 1e-9  # 0.1 m/s * 0.1 s
    for _ in range(100):
        sp = m.step(0.1)
    assert sp == (0.42, 0.0, 0.2)


def test_freeze_holds():
    m = Mover(Workspace(), 0.1)
    m.init_at((0.25, 0.0, 0.2))
    m.set_goal((0.4, 0.0, 0.2), 0.1)
    m.step(0.1)
    m.freeze("test")
    before = m.setpoint
    m.step(1.0)
    assert m.setpoint == before
    m.resume()
    m.set_goal((0.4, 0.0, 0.2), 0.1)
    assert m.step(0.1) != before


def test_watchdog_uses_deviation():
    w = EffortWatchdog(trip_n=10, baseline_s=0.5)
    t = 0.0
    for _ in range(10):
        assert w.update((-33, -6, -25), t) is None
        t += 0.1
    assert w.baseline is not None
    assert not w.tripped(w.update((-33, -6, -25), t))
    assert w.tripped(w.update((-33, -6, -40), t))
