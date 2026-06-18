"""ESP32 步数计数 → 车体里程计 (Jetson 端计算)

原理:
  ESP32 返回四轮步数 (带符号)
  → 步数 × (2πR / STEPS_PER_REV) = 轮位移 (m)
  → 麦轮正解算 → 车体位移 (dx_body, dy_body, dyaw)
  → 旋转到世界坐标系 → 累积 (x, y, yaw)
"""

from __future__ import annotations

import math
from typing import Tuple

# ── 物理参数 (与 ESP32 固件一致) ──
MICROSTEPS = 8
FULL_STEPS_PER_REV = 200
STEPS_PER_REV = FULL_STEPS_PER_REV * MICROSTEPS  # 1600
WHEEL_RADIUS_M = 0.023
LX = 0.15  # 轴距 X
LY = 0.12  # 轮距 Y
L_SUM = LX + LY  # 0.27

# 每步对应的轮子位移 (m)
STEP_TO_M = (2.0 * math.pi * WHEEL_RADIUS_M) / STEPS_PER_REV

# 正解算系数
INV_4  = 0.25        # dx/dy 系数
INV_4L = 1.0 / (4.0 * L_SUM)  # dyaw 系数


class OdomTracker:
    """从 ESP32 步数追踪车体里程计."""

    def __init__(self):
        self.x = 0.0     # 世界坐标 x (前), m
        self.y = 0.0     # 世界坐标 y (左), m
        self.yaw = 0.0   # 航向角, rad
        self._last_steps: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def reset(self):
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self._last_steps = (0, 0, 0, 0)

    def update(self, steps: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """输入最新四轮步数 (s0,s1,s2,s3), 返回 (x, y, yaw)."""
        s0, s1, s2, s3 = steps
        ls0, ls1, ls2, ls3 = self._last_steps

        # 增量步数
        ds0 = s0 - ls0
        ds1 = s1 - ls1
        ds2 = s2 - ls2
        ds3 = s3 - ls3

        # 步数增量 → 轮位移 (m)
        d0 = ds0 * STEP_TO_M
        d1 = ds1 * STEP_TO_M
        d2 = ds2 * STEP_TO_M
        d3 = ds3 * STEP_TO_M

        # 麦轮正解算: 轮位移 → 车体位移 (局部坐标系)
        # d0 = dx - dy - w*L  (左前)
        # d1 = dx + dy + w*L  (右前)
        # d2 = dx + dy - w*L  (左后)
        # d3 = dx - dy + w*L  (右后)
        dx_body = ( d0 + d1 + d2 + d3) * INV_4
        dy_body = (-d0 + d1 + d2 - d3) * INV_4
        dyaw    = (-d0 + d1 - d2 + d3) * INV_4L

        # 转换到世界坐标系
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        self.x   += dx_body * cos_y - dy_body * sin_y
        self.y   += dx_body * sin_y + dy_body * cos_y
        self.yaw += dyaw

        self._last_steps = steps
        return (self.x, self.y, self.yaw)

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)
