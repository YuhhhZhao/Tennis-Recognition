from __future__ import annotations

import math

from tennis_robot_sim.data import ControlCommand, RobotState
from tennis_robot_sim.robot.kinematics import clamp, wrap_angle


class PoseController:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def step(self, state: RobotState, target: RobotState, dt: float) -> ControlCommand:
        del dt
        dx = target.x - state.x
        dy = target.y - state.y
        distance = math.hypot(dx, dy)
        tolerance = float(self.cfg["robot"].get("position_tolerance_m", 0.15))
        if distance <= tolerance:
            return ControlCommand(0.0, 0.0)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(desired_yaw - state.yaw)
        controller = self.cfg["controller"]
        robot = self.cfg["robot"]
        safety = self.cfg.get("safety", {})
        max_speed = min(float(robot["max_speed_mps"]), float(safety.get("max_command_speed_mps", robot["max_speed_mps"])))
        max_omega = min(float(robot["max_angular_rate_radps"]), float(safety.get("max_command_omega_radps", robot["max_angular_rate_radps"])))
        omega = clamp(float(controller["angular_gain"]) * yaw_error, -max_omega, max_omega)
        heading_scale = max(0.0, math.cos(yaw_error))
        v = clamp(float(controller["linear_gain"]) * distance * heading_scale, -max_speed, max_speed)
        return ControlCommand(float(v), float(omega))

