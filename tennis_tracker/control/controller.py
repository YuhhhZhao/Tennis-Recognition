from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config import ControlConfig
from ..state import LandingPoint
from .uart_bridge import UartBridge


@dataclass
class ControlCommand:
    turn: float  # -1 (左) ~ +1 (右)
    forward: float  # -1 (后退) ~ +1 (前进)


class CarController:
    def __init__(
        self,
        cfg: ControlConfig,
        frame_width: int,
        frame_height: int,
        uart: Optional[UartBridge] = None,
    ):
        self.cfg = cfg
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.uart = uart

    # ---- 3D landing → UART direct (primary) --------------------------------

    def send_landing(self, lp: LandingPoint) -> None:
        """发送 3D 落点给 ESP32 下位板.

        有 UART 时直接发 TARGET 指令 (ESP32 自己做运动学).
        无 UART 时打印控制量 (模拟).
        """
        x, y, _z = lp.pos
        t = lp.t_arrival

        if self.uart and self.uart.is_open:
            ok = self.uart.send_target(x, y, t)
            if ok:
                # 检查 ESP32 是否有响应
                resp = self.uart.read_response()
                if resp:
                    print(f"[CTRL] ESP32: {resp}")
            else:
                print("[CTRL] UART send failed, falling back to print")
                self._print_landing(x, y, t)
        else:
            self._print_landing(x, y, t)

    # ---- 2D fallback -------------------------------------------------------

    def send_target(self, target: Tuple[float, float]) -> None:
        """2D 像素目标 → 控制 (降级)."""
        cmd = self.command_from_target(target)
        if not self.cfg.enabled:
            return
        print(f"control turn={cmd.turn:.3f} forward={cmd.forward:.3f}")

    # ---- command builders ---------------------------------------------------

    def command_from_landing(self, lp: LandingPoint) -> ControlCommand:
        """从 3D 落点生成麦轮控制指令 (无 UART 时使用)."""
        x, y, _z = lp.pos
        max_lateral = max(0.1, x * 0.5)
        turn = y / max_lateral
        turn = max(-1.0, min(1.0, turn))
        forward = x / 3.0
        forward = max(-1.0, min(1.0, forward))
        if abs(y) < 0.1:
            turn = 0.0
        if abs(x) < 0.2:
            forward = 0.0
        return ControlCommand(turn=float(turn), forward=float(forward))

    def command_from_target(self, target: Tuple[float, float]) -> ControlCommand:
        """2D 像素目标 → 控制."""
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

    # ---- internals ----------------------------------------------------------

    def _print_landing(self, x: float, y: float, t: float) -> None:
        if not self.cfg.enabled:
            return
        print(
            f"[CTRL] TARGET x={x:.3f}m y={y:.3f}m t={t:.3f}s"
        )

    def _normalize(self, value: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        value = value / scale
        return max(-self.cfg.max_command, min(self.cfg.max_command, value))
