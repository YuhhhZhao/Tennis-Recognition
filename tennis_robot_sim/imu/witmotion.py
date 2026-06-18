"""
WitMotion IMU 真实驱动 (JY901 / WT901 系列)

协议: 0x55 + type + data[8] (每包 11 字节)
  0x51: 加速度 (ax, ay, az, temp)  — ±16g 量程
  0x52: 角速度 (gx, gy, gz)         — ±2000°/s 量程
  0x53: 姿态角 (roll, pitch, yaw)   — ±180°
  0x54: 磁场   (mx, my, mz)

连接: USB CDC ACM → /dev/ttyACM0, 115200 8N1
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Optional

import serial

from tennis_robot_sim.data import IMUSample

# 量程转换常数
GYRO_SCALE_DPS = 2000.0 / 32768.0    # 原始值 → °/s
ACCEL_SCALE_G = 16.0 / 32768.0        # 原始值 → g
DEG_TO_RAD = 3.1415926535 / 180.0
G_TO_MPS2 = 9.81


class WitMotionIMU:
    """WitMotion 串口 IMU 驱动，后台线程持续读取。"""

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200):
        self.port = port
        self.baud = baud
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # 最新解析值
        self._accel = (0.0, 0.0, 0.0)
        self._gyro = (0.0, 0.0, 0.0)
        self._angle = (0.0, 0.0, 0.0)
        self._sample_count = 0
        self._last_sample_time = 0.0

    # ── lifecycle ─────────────────────────────────────────────────

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        except (OSError, serial.SerialException) as e:
            print(f"[IMU] 无法打开 {self.port}: {e}")
            return False

        time.sleep(0.2)
        self._ser.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[IMU] WitMotion 已连接 ({self.port} @ {self.baud})")
        return True

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        print("[IMU] 已关闭")

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open and self._running

    # ── data access ───────────────────────────────────────────────

    def get_sample(self) -> IMUSample:
        """返回最近一次 IMU 采样，对齐仿真接口 IMUSample。"""
        with self._lock:
            ax, ay, az = self._accel
            gx, gy, gz = self._gyro

        return IMUSample(
            timestamp=time.time(),
            accel_mps2=(ax * G_TO_MPS2, ay * G_TO_MPS2, az * G_TO_MPS2),
            gyro_radps=(
                gx * DEG_TO_RAD,
                gy * DEG_TO_RAD,
                gz * DEG_TO_RAD,
            ),
            yaw_rate_radps=gz * DEG_TO_RAD,
        )

    def get_angle(self) -> tuple:
        """返回 (roll, pitch, yaw) 单位：度."""
        with self._lock:
            return self._angle

    @property
    def sample_count(self) -> int:
        return self._sample_count

    # ── background read ───────────────────────────────────────────

    def _read_loop(self) -> None:
        """后台线程：持续读取并解析."""
        buf = bytearray()
        while self._running:
            try:
                if self._ser.in_waiting:
                    buf.extend(self._ser.read(self._ser.in_waiting))
                    self._parse_buffer(buf)
                    # 保留最后 11 字节防止帧撕裂
                    if len(buf) > 22:
                        buf = buf[-22:]
                else:
                    time.sleep(0.001)
            except (OSError, serial.SerialException):
                time.sleep(0.01)

    def _parse_buffer(self, data: bytearray) -> None:
        """解析 WitMotion 0x55 协议."""
        i = 0
        n = len(data)
        while i <= n - 11:
            if data[i] == 0x55:
                ptype = data[i + 1]
                if ptype == 0x51:  # 加速度
                    ax, ay, az, _ = struct.unpack_from("<hhhH", data, i + 2)
                    with self._lock:
                        self._accel = (
                            ax * ACCEL_SCALE_G,
                            ay * ACCEL_SCALE_G,
                            az * ACCEL_SCALE_G,
                        )
                elif ptype == 0x52:  # 角速度
                    gx, gy, gz, _ = struct.unpack_from("<hhhH", data, i + 2)
                    with self._lock:
                        self._gyro = (
                            gx * GYRO_SCALE_DPS,
                            gy * GYRO_SCALE_DPS,
                            gz * GYRO_SCALE_DPS,
                        )
                        self._sample_count += 1
                        self._last_sample_time = time.time()
                elif ptype == 0x53:  # 姿态角
                    roll, pitch, yaw, _ = struct.unpack_from("<hhhH", data, i + 2)
                    with self._lock:
                        self._angle = (
                            roll * 180.0 / 32768.0,
                            pitch * 180.0 / 32768.0,
                            yaw * 180.0 / 32768.0,
                        )
                # 0x54 磁场暂时不用
                i += 11
            else:
                i += 1
