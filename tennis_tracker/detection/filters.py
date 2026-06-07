from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..state import Detection, Point, TrackState, now


@dataclass
class AlphaBetaFilter:
    alpha: float = 0.75
    beta: float = 0.20

    def update(self, state: TrackState, detection: Detection) -> TrackState:
        t = detection.timestamp
        if state.center is None or state.last_update <= 0.0:
            state.center = detection.center
            state.velocity = (0.0, 0.0)
            state.radius = max(3.0, detection.radius)
            state.confidence = min(10, state.confidence + 4)
            state.missing_frames = 0
            state.last_update = t
            state.source = detection.source
            return state

        dt = max(1e-3, t - state.last_update)
        px, py = self.predict(state, dt)
        mx, my = detection.center
        rx, ry = mx - px, my - py

        vx, vy = state.velocity
        state.center = (px + self.alpha * rx, py + self.alpha * ry)
        state.velocity = (
            vx + self.beta * rx / dt,
            vy + self.beta * ry / dt,
        )
        state.radius = 0.7 * state.radius + 0.3 * max(3.0, detection.radius)
        state.confidence = min(10, state.confidence + 2)
        state.missing_frames = 0
        state.last_update = t
        state.source = detection.source
        return state

    def predict(self, state: TrackState, dt: float) -> Point:
        if state.center is None:
            return (0.0, 0.0)
        cx, cy = state.center
        vx, vy = state.velocity
        return (cx + vx * dt, cy + vy * dt)

    def predict_latency(self, state: TrackState, latency_ms: int) -> Point:
        return self.predict(state, latency_ms / 1000.0)


def clamp_point(point: Point, width: int, height: int) -> Tuple[int, int]:
    x = int(max(0, min(width - 1, point[0])))
    y = int(max(0, min(height - 1, point[1])))
    return x, y
