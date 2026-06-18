from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .ball import BallState, landing_point, simulate_projectile


@dataclass
class Scenario:
    name: str
    seed: int
    initial_position: Tuple[float, float, float]
    initial_velocity: Tuple[float, float, float]
    states: list[BallState]
    landing_xy: Tuple[float, float]
    landing_time: float

    @property
    def times(self) -> np.ndarray:
        return np.array([state.timestamp for state in self.states], dtype=float)

    @property
    def positions(self) -> np.ndarray:
        return np.array([state.position for state in self.states], dtype=float)


SCENARIOS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "default": ((2.00, 0.55, 1.10), (1.20, -0.38, 4.00)),
    "high_lob": ((2.10, -0.55, 1.15), (0.95, 0.40, 5.20)),
    "fast_cross": ((2.10, 1.25, 0.95), (2.40, -1.55, 3.35)),
    "noisy_camera": ((2.00, 0.30, 1.05), (1.20, -0.25, 4.00)),
}


def make_scenario(name: str = "default", cfg: Optional[dict] = None, seed: Optional[int] = None) -> Scenario:
    if name not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown scenario '{name}', valid scenarios: {valid}")
    sim_cfg = (cfg or {}).get("sim", {})
    ball_cfg = (cfg or {}).get("ball", {})
    frames = int(sim_cfg.get("frames", 90))
    dt = float(sim_cfg.get("dt", 1.0 / 60.0))
    scenario_seed = int(seed if seed is not None else sim_cfg.get("seed", 7))
    gravity = float(ball_cfg.get("gravity_mps2", -9.81))
    bounce = float(ball_cfg.get("bounce_coefficient", 0.0))

    initial_position, initial_velocity = SCENARIOS[name]
    if name == "noisy_camera":
        rng = np.random.default_rng(scenario_seed)
        initial_position = tuple(np.asarray(initial_position) + rng.normal(0.0, 0.015, size=3))
        initial_velocity = tuple(np.asarray(initial_velocity) + rng.normal(0.0, 0.02, size=3))

    states = simulate_projectile(
        initial_position,
        initial_velocity,
        dt=dt,
        steps=frames,
        gravity=gravity,
        bounce_coefficient=bounce,
    )
    landing = landing_point(initial_position, initial_velocity, gravity=gravity)
    if landing is None:
        raise RuntimeError(f"scenario '{name}' does not intersect the ground plane")
    return Scenario(
        name=name,
        seed=scenario_seed,
        initial_position=tuple(float(v) for v in initial_position),
        initial_velocity=tuple(float(v) for v in initial_velocity),
        states=states,
        landing_xy=(landing[0], landing[1]),
        landing_time=landing[2],
    )
