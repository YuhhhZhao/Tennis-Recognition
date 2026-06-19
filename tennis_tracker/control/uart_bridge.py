"""Jetson ↔ ESP32 TinyBee UART 通信桥 v3.1

协议 (文本, 115200 8N1):
  Jetson → ESP32:  TARGET <x> <y> <t>\n
                    VEL <vx> <vy> <w>\n
                    STOP\n
                    PING\n         → 返回里程计 (x,y,yaw) + 步数
                    ODOM\n         → 纯里程计查询
                    RESET_ODOM\n   → 里程计归零
                    STAT\n
  ESP32 → Jetson:  RDY\n  |  OK\n  |  ERR <msg>\n
                    PONG x=<m> y=<m> yaw=<rad> steps=<s0>,<s1>,<s2>,<s3>\n
                    ODOM x=<m> y=<m> yaw=<rad>\n
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import time

from ..config import UartConfig

_SER_EXC = (OSError,)
try:
    import serial as _serial_mod
    _SER_EXC = (OSError, getattr(_serial_mod, "SerialException", OSError))
except ImportError:
    _serial_mod = None


@dataclass
class StepCounts:
    """ESP32 四轮带符号步数."""
    s0: int = 0  # 左前
    s1: int = 0  # 右前
    s2: int = 0  # 左后
    s3: int = 0  # 右后

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.s0, self.s1, self.s2, self.s3)


class UartBridge:
    """与 ESP32 下位板通信的串口桥."""

    def __init__(self, cfg: UartConfig):
        self.cfg = cfg
        self._ser: Any = None

    # ---- lifecycle --------------------------------------------------------

    def open(self) -> bool:
        """打开串口, 等待 ESP32 RDY 握手."""
        if not self.cfg.enabled or _serial_mod is None:
            return False
        try:
            print(
                f"[UART] opening {self.cfg.port} @ {self.cfg.baudrate}...",
                flush=True,
            )
            self._ser = _serial_mod.Serial(
                port=self.cfg.port,
                baudrate=self.cfg.baudrate,
                timeout=self.cfg.timeout_s,
                write_timeout=0.1,
            )
            print(f"[UART] port opened: {self.cfg.port}", flush=True)
            if self.cfg.handshake_timeout_s <= 0:
                print("[UART] RDY wait skipped", flush=True)
                return True

            print(
                f"[UART] waiting RDY up to {self.cfg.handshake_timeout_s:.1f}s...",
                flush=True,
            )
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.cfg.handshake_timeout_s:
                line = self._readline()
                if line and "RDY" in line:
                    print(f"[UART] ESP32 ready on {self.cfg.port}", flush=True)
                    return True
            print(
                f"[UART] WARNING: no RDY from ESP32, port open but unconfirmed",
                flush=True,
            )
            return True
        except _SER_EXC as e:
            print(f"[UART] ERROR: cannot open {self.cfg.port}: {e}", flush=True)
            self._ser = None
            return False

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ---- commands ---------------------------------------------------------

    def send_target(self, x_m: float, y_m: float, t_arrival_s: float) -> bool:
        """发送落点指令."""
        msg = f"TARGET {x_m:.3f} {y_m:.3f} {t_arrival_s:.3f}\n"
        return self._write(msg)

    def send_vel(self, vx: float, vy: float, w: float) -> bool:
        """发送原始速度指令 (调试用)."""
        msg = f"VEL {vx:.3f} {vy:.3f} {w:.3f}\n"
        return self._write(msg)

    def send_stop(self) -> bool:
        """紧急停车."""
        return self._write("STOP\n")

    def send_ping(self) -> Optional[StepCounts]:
        """发送 PING, 返回四轮步数."""
        if not self._write("PING\n"):
            return None
        line = self._wait_for_prefix("PONG", timeout=0.3)
        if line:
            return self._parse_pong(line)
        return None

    def send_reset_steps(self) -> bool:
        """重置 ESP32 步数计数."""
        if not self._write("RESET_STEPS\n"):
            return False
        line = self._wait_for_prefix("STEPS_RESET", timeout=0.3)
        return line is not None

    def read_response(self) -> Optional[str]:
        """非阻塞读取一行 ESP32 响应."""
        return self._readline()

    # ---- parsers ----------------------------------------------------------

    @staticmethod
    def _parse_pong(line: str) -> Optional[StepCounts]:
        """解析 PONG s=<s0>,<s1>,<s2>,<s3>"""
        try:
            # "PONG s=1234,-567,890,42"
            s = line.split("=", 1)[1]  # "1234,-567,890,42"
            vals = [int(v) for v in s.split(",")]
            return StepCounts(s0=vals[0], s1=vals[1], s2=vals[2], s3=vals[3])
        except (IndexError, ValueError, AttributeError):
            return None

    # ---- internals --------------------------------------------------------

    def _wait_for_prefix(self, prefix: str, timeout: float) -> Optional[str]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            line = self._readline()
            if line and line.startswith(prefix):
                return line
        return None

    def _write(self, data: str) -> bool:
        if not self.is_open:
            return False
        try:
            self._ser.write(data.encode("ascii"))
            self._ser.flush()
            return True
        except _SER_EXC:
            return False

    def _readline(self) -> Optional[str]:
        if not self.is_open:
            return None
        try:
            if self._ser.in_waiting > 0:
                line = self._ser.readline()
                if line:
                    text = line.decode("ascii", errors="ignore").strip()
                    return text or None
        except _SER_EXC:
            pass
        return None
