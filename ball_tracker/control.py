from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import ControlConfig


@dataclass
class ControlCommand:
    turn: float
    forward: float


class CarController:
    def __init__(self, cfg: ControlConfig, frame_width: int, frame_height: int):
        self.cfg = cfg
        self.frame_width = frame_width
        self.frame_height = frame_height

    def command_from_target(self, target: Tuple[float, float]) -> ControlCommand:
        tx, ty = target
        cx = self.frame_width / 2.0
        cy = self.frame_height / 2.0

        error_x = tx - cx
        error_y = cy - ty

        turn = self._normalize(error_x, cx)
        forward = self._normalize(error_y, cy)

        if abs(error_x) < self.cfg.deadband_px:
            turn = 0.0
        if abs(error_y) < self.cfg.deadband_px:
            forward = 0.0

        return ControlCommand(turn=turn, forward=forward)

    def send(self, command: ControlCommand) -> None:
        if not self.cfg.enabled:
            return

        # Replace this with UART/CAN/ROS2/UDP for the real vehicle.
        print(f"control turn={command.turn:.3f} forward={command.forward:.3f}")

    def _normalize(self, value: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        value = value / scale
        value = max(-self.cfg.max_command, min(self.cfg.max_command, value))
        return value

