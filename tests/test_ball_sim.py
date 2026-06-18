from __future__ import annotations

import numpy as np

from tennis_robot_sim.sim.ball import landing_point, simulate_projectile
from tennis_robot_sim.sim.scenario import make_scenario


def test_default_scenario_is_deterministic():
    a = make_scenario("default")
    b = make_scenario("default")
    assert len(a.states) >= 60
    assert np.allclose(a.positions, b.positions)
    assert a.landing_time > 0


def test_landing_point_matches_projectile_solution():
    landing = landing_point((0.0, 0.0, 1.0), (1.0, 0.5, 2.0), gravity=-10.0)
    assert landing is not None
    x, y, t = landing
    assert t > 0
    assert abs(x - t) < 1e-9
    assert abs(y - 0.5 * t) < 1e-9


def test_bounce_clamps_to_ground():
    states = simulate_projectile((0.0, 0.0, 0.1), (0.0, 0.0, -2.0), dt=0.05, steps=8, bounce_coefficient=0.5)
    assert min(s.position[2] for s in states) >= 0.0
    assert any(s.bounced for s in states)

