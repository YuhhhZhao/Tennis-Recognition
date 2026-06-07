"""Jetson ↔ ESP32 TinyBee UART 通信桥

协议 (文本, 115200 8N1):
  Jetson → ESP32:  TARGET <x> <y> <t>\n
  ESP32 → Jetson:  RDY\n  |  OK\n  |  DONE\n  |  ERR <msg>\n  |  PONG x=... y=...\n
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import time

from ..config import UartConfig

# pyserial 是可选依赖, 仅在需要 UART 通信时安装
_SER_EXC = (OSError,)
try:
    import serial as _serial_mod
    _SER_EXC = (OSError, getattr(_serial_mod, "SerialException", OSError))
except ImportError:
    _serial_mod = None
    pass


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
            self._ser = _serial_mod.Serial(
                port=self.cfg.port,
                baudrate=self.cfg.baudrate,
                timeout=self.cfg.timeout_s,
                write_timeout=0.1,
            )
            # 等待 ESP32 上电握手
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                line = self._readline()
                if line and "RDY" in line:
                    print(f"[UART] ESP32 ready on {self.cfg.port}")
                    return True
            print(f"[UART] WARNING: no RDY from ESP32, port open but unconfirmed")
            return True
        except _SER_EXC as e:
            print(f"[UART] ERROR: cannot open {self.cfg.port}: {e}")
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

    def send_stop(self) -> bool:
        """紧急停车."""
        return self._write("STOP\n")

    def send_ping(self) -> Optional[Tuple[float, float]]:
        """发送 PING, 返回 ESP32 报告的里程计 (x, y) 或 None."""
        if not self._write("PING\n"):
            return None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            line = self._readline()
            if line and line.startswith("PONG"):
                parts = line.split()
                try:
                    x = float(parts[1].split("=")[1])
                    y = float(parts[2].split("=")[1])
                    return (x, y)
                except (IndexError, ValueError):
                    return None
        return None

    def read_response(self) -> Optional[str]:
        """非阻塞读取一行 ESP32 响应. 用于状态监控."""
        return self._readline()

    # ---- internals --------------------------------------------------------

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
                    return line.decode("ascii").strip()
        except _SER_EXC:
            pass
        return None
