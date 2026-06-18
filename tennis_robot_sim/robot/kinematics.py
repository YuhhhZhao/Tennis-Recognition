from __future__ import annotations

import math

from tennis_robot_sim.data import ControlCommand, RobotState


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def integrate_unicycle(state: RobotState, command: ControlCommand, dt: float) -> RobotState:
    yaw = wrap_angle(state.yaw + command.omega * dt)
    x = state.x + command.v * math.cos(yaw) * dt
    y = state.y + command.v * math.sin(yaw) * dt
    return RobotState(x=float(x), y=float(y), yaw=float(yaw), v=float(command.v), omega=float(command.omega), timestamp=state.timestamp + dt)

