from __future__ import annotations

import math

from tennis_robot_sim.data import InterceptionPlan, LandingPrediction, RobotState
from tennis_robot_sim.robot.kinematics import clamp


class InterceptionPlanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def plan(self, prediction: LandingPrediction, current_state: RobotState) -> InterceptionPlan:
        court = self.cfg["court"]
        offset = self.cfg["robot"].get("catch_offset_m", [0.0, 0.0])
        desired_x = prediction.landing_xy[0] + float(offset[0])
        desired_y = prediction.landing_xy[1] + float(offset[1])
        target_x = clamp(desired_x, float(court["x_min"]), float(court["x_max"]))
        target_y = clamp(desired_y, float(court["y_min"]), float(court["y_max"]))
        dx = target_x - current_state.x
        dy = target_y - current_state.y
        distance = math.hypot(dx, dy)
        max_speed = float(self.cfg["robot"]["max_speed_mps"])
        eta = distance / max_speed if max_speed > 0 else float("inf")
        reachable = eta <= prediction.time_to_land + 0.15
        reason = "reachable"
        if desired_x != target_x or desired_y != target_y:
            reason = "target clipped to court bounds"
        if not reachable:
            travel = max(0.0, max_speed * max(0.0, prediction.time_to_land))
            ratio = min(1.0, travel / distance) if distance > 1e-9 else 1.0
            target_x = current_state.x + dx * ratio
            target_y = current_state.y + dy * ratio
            reason = "unreachable before landing; best effort target selected"
        yaw = math.atan2(target_y - current_state.y, target_x - current_state.x) if distance > 1e-9 else current_state.yaw
        return InterceptionPlan(
            target_pose=RobotState(float(target_x), float(target_y), float(yaw), timestamp=current_state.timestamp),
            reachable=bool(reachable),
            eta=float(eta),
            distance=float(distance),
            reason=reason,
        )

