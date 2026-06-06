from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import ControlConfig
from .state import LandingPoint


@dataclass
class ControlCommand:
    turn: float  # -1 (左) ~ +1 (右)
    forward: float  # -1 (后退) ~ +1 (前进)


class CarController:
    def __init__(self, cfg: ControlConfig, frame_width: int, frame_height: int):
        self.cfg = cfg
        self.frame_width = frame_width
        self.frame_height = frame_height

    # ---- 3D landing (primary) -------------------------------------------

    def command_from_landing(self, lp: LandingPoint) -> ControlCommand:
        """从 3D 落点生成控制指令.

        落点坐标系: X=前方, Y=左方, Z=上方 (机器人自身).
        turn: 横向偏移 Y → 转向, 正值=左方 → 机器人左转.
        forward: 纵向距离 X → 前进量, 到达 X 后停下.
        """
        x, y, _z = lp.pos
        # turn: 归一化横向偏移
        # 假设正前方 2m 处横向 1m = 最大转向
        max_lateral = max(0.1, x * 0.5)
        turn = y / max_lateral
        turn = max(-1.0, min(1.0, turn))

        # forward: 距离越远越前进
        # 假设 3m 前为全速前进
        forward = x / 3.0
        forward = max(-1.0, min(1.0, forward))

        # deadband
        if abs(y) < 0.1:
            turn = 0.0
        if abs(x) < 0.2:
            forward = 0.0

        return ControlCommand(turn=float(turn), forward=float(forward))

    # ---- 2D fallback ---------------------------------------------------

    def command_from_target(self, target: Tuple[float, float]) -> ControlCommand:
        """2D 像素目标 → 控制 (原有降级逻辑)."""
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

    # ---- unified send --------------------------------------------------

    def send_landing(self, lp: LandingPoint) -> None:
        """发送 3D 落点控制指令."""
        cmd = self.command_from_landing(lp)
        self._send(cmd)

    def send_target(self, target: Tuple[float, float]) -> None:
        """发送 2D 像素控制指令 (降级)."""
        cmd = self.command_from_target(target)
        self._send(cmd)

    # ---- internals ------------------------------------------------------

    def _send(self, command: ControlCommand) -> None:
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

