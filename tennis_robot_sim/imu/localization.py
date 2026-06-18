from __future__ import annotations

import math

from tennis_robot_sim.data import ControlCommand, IMUSample, RobotState
from tennis_robot_sim.robot.kinematics import wrap_angle


class ComplementaryLocalizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        start = cfg["robot"].get("start_pose", [0.0, 0.0, 0.0])
        self.state = RobotState(float(start[0]), float(start[1]), float(start[2]), timestamp=0.0)
        self._imu_yaw = self.state.yaw
        self.covariance = 1.0

    def reset(self, state: RobotState | None = None) -> None:
        if state is not None:
            self.state = state
        else:
            start = self.cfg["robot"].get("start_pose", [0.0, 0.0, 0.0])
            self.state = RobotState(float(start[0]), float(start[1]), float(start[2]), timestamp=0.0)
        self._imu_yaw = self.state.yaw
        self.covariance = 1.0

    def predict(self, command: ControlCommand, dt: float) -> RobotState:
        yaw = wrap_angle(self.state.yaw + command.omega * dt)
        x = self.state.x + command.v * math.cos(yaw) * dt
        y = self.state.y + command.v * math.sin(yaw) * dt
        self.state = RobotState(float(x), float(y), float(yaw), v=command.v, omega=command.omega, timestamp=self.state.timestamp + dt)
        self.covariance += abs(command.v) * dt * 0.01 + abs(command.omega) * dt * 0.02
        return self.state

    def update_imu(self, sample: IMUSample, dt: float) -> RobotState:
        alpha = float(self.cfg["imu"].get("yaw_complementary_alpha", 0.92))
        self._imu_yaw = wrap_angle(self._imu_yaw + sample.yaw_rate_radps * dt)
        fused_yaw = wrap_angle(alpha * self._imu_yaw + (1.0 - alpha) * self.state.yaw)
        self.state.yaw = fused_yaw
        self.state.timestamp = max(self.state.timestamp, sample.timestamp)
        self.covariance = max(0.01, self.covariance * 0.98)
        return self.state

    def update_odometry(self, odom_state: RobotState, weight: float = 0.15) -> RobotState:
        weight = max(0.0, min(1.0, weight))
        self.state.x = (1.0 - weight) * self.state.x + weight * odom_state.x
        self.state.y = (1.0 - weight) * self.state.y + weight * odom_state.y
        self.state.yaw = wrap_angle((1.0 - weight) * self.state.yaw + weight * odom_state.yaw)
        self.covariance = max(0.01, self.covariance * (1.0 - 0.5 * weight))
        return self.state

    def get_state(self) -> RobotState:
        return RobotState(
            x=float(self.state.x),
            y=float(self.state.y),
            yaw=float(self.state.yaw),
            v=float(self.state.v),
            omega=float(self.state.omega),
            timestamp=float(self.state.timestamp),
        )
