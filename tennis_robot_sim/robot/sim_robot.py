from __future__ import annotations

from tennis_robot_sim.data import ControlCommand, RobotState
from tennis_robot_sim.robot.kinematics import clamp, integrate_unicycle


class SimRobot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        start = cfg["robot"].get("start_pose", [0.0, 0.0, 0.0])
        self.state = RobotState(float(start[0]), float(start[1]), float(start[2]), timestamp=0.0)

    def reset(self, state: RobotState | None = None) -> None:
        if state is not None:
            self.state = state
        else:
            start = self.cfg["robot"].get("start_pose", [0.0, 0.0, 0.0])
            self.state = RobotState(float(start[0]), float(start[1]), float(start[2]), timestamp=0.0)

    def step(self, command: ControlCommand, dt: float) -> RobotState:
        robot_cfg = self.cfg["robot"]
        safety_cfg = self.cfg.get("safety", {})
        max_speed = min(float(robot_cfg["max_speed_mps"]), float(safety_cfg.get("max_command_speed_mps", robot_cfg["max_speed_mps"])))
        max_omega = min(float(robot_cfg["max_angular_rate_radps"]), float(safety_cfg.get("max_command_omega_radps", robot_cfg["max_angular_rate_radps"])))
        max_accel = float(robot_cfg.get("max_accel_mps2", 999.0))
        dv = clamp(command.v - self.state.v, -max_accel * dt, max_accel * dt)
        limited = ControlCommand(
            v=clamp(self.state.v + dv, -max_speed, max_speed),
            omega=clamp(command.omega, -max_omega, max_omega),
        )
        next_state = integrate_unicycle(self.state, limited, dt)
        court = self.cfg["court"]
        next_state.x = clamp(next_state.x, float(court["x_min"]), float(court["x_max"]))
        next_state.y = clamp(next_state.y, float(court["y_min"]), float(court["y_max"]))
        self.state = next_state
        return self.state

