from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from tennis_robot_sim.data import Detection, LandingPrediction
from tennis_robot_sim.estimation.geometry import detection_to_world
from tennis_robot_sim.sim.camera import PinholeCamera


@dataclass
class TrajectoryEstimator:
    cfg: dict
    camera: PinholeCamera
    observations: List[Tuple[float, Tuple[float, float, float]]] = field(default_factory=list)

    def reset(self) -> None:
        self.observations.clear()

    @property
    def min_samples(self) -> int:
        return int(self.cfg.get("trajectory", {}).get("min_samples_for_fit", 6))

    def update_detection(self, detection: Optional[Detection]) -> Optional[tuple[float, float, float]]:
        if detection is None:
            return None
        point = detection_to_world(self.camera, detection, float(self.cfg["ball"]["radius_m"]))
        if point is None or not all(math.isfinite(v) for v in point):
            return None
        self.add_observation(detection.timestamp, point)
        return point

    def add_observation(self, timestamp: float, point: tuple[float, float, float]) -> tuple[float, float, float]:
        self.observations.append((float(timestamp), tuple(float(v) for v in point)))
        max_history = int(self.cfg.get("trajectory", {}).get("max_history", 80))
        if len(self.observations) > max_history:
            self.observations = self.observations[-max_history:]
        return self.observations[-1][1]

    def predict(self, now: Optional[float] = None) -> Optional[LandingPrediction]:
        if len(self.observations) < self.min_samples:
            return None
        gravity = float(self.cfg["ball"].get("gravity_mps2", -9.81))
        times = np.array([item[0] for item in self.observations], dtype=float)
        points = np.array([item[1] for item in self.observations], dtype=float)
        if now is None:
            now = float(times[-1])
        t0 = float(times[0])
        tau = times - t0
        current_tau = float(now - t0)

        a_lin = np.column_stack([np.ones_like(tau), tau])
        x_coef, _, _, _ = np.linalg.lstsq(a_lin, points[:, 0], rcond=None)
        y_coef, _, _, _ = np.linalg.lstsq(a_lin, points[:, 1], rcond=None)

        z_adjusted = points[:, 2] - 0.5 * gravity * tau * tau
        z_coef, _, _, _ = np.linalg.lstsq(a_lin, z_adjusted, rcond=None)

        x_now = x_coef[0] + x_coef[1] * current_tau
        y_now = y_coef[0] + y_coef[1] * current_tau
        z_now = z_coef[0] + z_coef[1] * current_tau + 0.5 * gravity * current_tau * current_tau
        vx = x_coef[1]
        vy = y_coef[1]
        vz = z_coef[1] + gravity * current_tau

        a = 0.5 * gravity
        b = vz
        c = z_now
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        roots = [(-b + math.sqrt(disc)) / (2.0 * a), (-b - math.sqrt(disc)) / (2.0 * a)]
        future = [root for root in roots if root >= 0.0]
        if not future:
            return None
        dt_land = min(future)
        landing_xy = (float(x_now + vx * dt_land), float(y_now + vy * dt_land))

        pred_z = z_coef[0] + z_coef[1] * tau + 0.5 * gravity * tau * tau
        pred_x = x_coef[0] + x_coef[1] * tau
        pred_y = y_coef[0] + y_coef[1] * tau
        residual = np.sqrt((pred_x - points[:, 0]) ** 2 + (pred_y - points[:, 1]) ** 2 + (pred_z - points[:, 2]) ** 2)
        uncertainty = float(max(np.mean(residual), 0.02))
        confidence = float(max(0.0, min(1.0, 1.0 - uncertainty / 0.6)))
        return LandingPrediction(
            landing_xy=landing_xy,
            time_to_land=float(dt_land),
            uncertainty=uncertainty,
            confidence=confidence,
            debug_metrics={
                "samples": float(len(self.observations)),
                "x_now": float(x_now),
                "y_now": float(y_now),
                "z_now": float(z_now),
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
                "residual_mean_m": uncertainty,
            },
        )
