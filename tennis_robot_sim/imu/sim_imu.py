from __future__ import annotations

import math
from typing import Optional

import numpy as np

from tennis_robot_sim.data import ControlCommand, IMUSample, RobotState


class IMUSimulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg["imu"] if "imu" in cfg else cfg
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 123)))
        self._previous_state: Optional[RobotState] = None

    def reset(self) -> None:
        self._previous_state = None

    def sample(self, state: RobotState, command: ControlCommand, dt: float) -> IMUSample:
        prev_v = self._previous_state.v if self._previous_state is not None and dt > 0 else state.v
        linear_accel = (state.v - prev_v) / dt if dt > 0 else 0.0
        accel_body = np.array([linear_accel, 0.0, 0.0], dtype=float)
        yaw_rate = state.omega if math.isfinite(state.omega) else command.omega
        gyro = np.array([0.0, 0.0, yaw_rate], dtype=float)

        accel_bias = np.asarray(self.cfg.get("accel_bias", [0.0, 0.0, 0.0]), dtype=float)
        gyro_bias = np.asarray(self.cfg.get("gyro_bias", [0.0, 0.0, 0.0]), dtype=float)
        accel_noise = self.rng.normal(0.0, float(self.cfg.get("accel_noise_std", 0.0)), size=3)
        gyro_noise = self.rng.normal(0.0, float(self.cfg.get("gyro_noise_std", 0.0)), size=3)
        jitter = float(self.rng.normal(0.0, float(self.cfg.get("timestamp_jitter_std", 0.0))))
        self._previous_state = state
        accel = accel_body + accel_bias + accel_noise
        gyro = gyro + gyro_bias + gyro_noise
        return IMUSample(
            timestamp=float(state.timestamp + jitter),
            accel_mps2=(float(accel[0]), float(accel[1]), float(accel[2])),
            gyro_radps=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
            yaw_rate_radps=float(gyro[2]),
            ground_truth_pose=state,
        )

