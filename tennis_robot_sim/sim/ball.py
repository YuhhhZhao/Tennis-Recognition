from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass
class BallState:
    timestamp: float
    position: Vec3
    velocity: Vec3
    bounced: bool = False


def landing_point(
    initial_position: Iterable[float],
    initial_velocity: Iterable[float],
    gravity: float = -9.81,
    ground_z: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    """Return the first ground-plane intersection as (x, y, t)."""
    p0 = np.asarray(tuple(initial_position), dtype=float)
    v0 = np.asarray(tuple(initial_velocity), dtype=float)
    z0 = p0[2] - ground_z
    a = 0.5 * gravity
    b = v0[2]
    c = z0
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        if t <= 0:
            return None
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
        future = [root for root in roots if root > 1e-9]
        if not future:
            return None
        t = min(future)
    xy = p0[:2] + v0[:2] * t
    return (float(xy[0]), float(xy[1]), float(t))


def simulate_projectile(
    initial_position: Iterable[float],
    initial_velocity: Iterable[float],
    dt: float,
    steps: int,
    gravity: float = -9.81,
    bounce_coefficient: float = 0.0,
    ground_z: float = 0.0,
) -> List[BallState]:
    """Integrate projectile motion with an optional simple bounce."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    position = np.asarray(tuple(initial_position), dtype=float)
    velocity = np.asarray(tuple(initial_velocity), dtype=float)
    states: List[BallState] = []

    for i in range(steps):
        t = i * dt
        bounced = False
        states.append(
            BallState(
                timestamp=float(t),
                position=(float(position[0]), float(position[1]), float(position[2])),
                velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                bounced=False,
            )
        )
        next_velocity = velocity.copy()
        next_velocity[2] += gravity * dt
        next_position = position + velocity * dt + np.array([0.0, 0.0, 0.5 * gravity * dt * dt])
        if next_position[2] < ground_z:
            next_position[2] = ground_z
            if bounce_coefficient > 0.0 and velocity[2] < 0.0:
                next_velocity[2] = -next_velocity[2] * bounce_coefficient
                bounced = True
            else:
                next_velocity[2] = 0.0
        position = next_position
        velocity = next_velocity
        if bounced and states:
            states[-1].bounced = True

    return states

