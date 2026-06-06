"""3D 轨迹滤波器 + 抛物线落点求解器.

TrajectoryFilter — 6D Kalman 滤波器, 带重力模型.
BallisticSolver  — 从滤波后的状态求解落点.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import TrajectoryConfig
from ..state import Detection3D, LandingPoint, Vec3, now

Vec3 = Tuple[float, float, float]


class TrajectoryFilter:
    """6D Kalman 滤波器: state = [x, y, z, vx, vy, vz]^T

    过程模型: x/y 匀速, z 匀加速 (重力).
    观测: [x, y, z] 直接测量.
    """

    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg
        self.g = cfg.gravity
        self._x = np.zeros((6, 1), dtype=np.float64)  # state
        self._P = np.eye(6, dtype=np.float64) * 100.0  # covariance
        self._initialized = False
        self._last_t: float = 0.0
        self.history: List[Detection3D] = []

    # ----- public API --------------------------------------------------------

    def update(self, det: Detection3D) -> Optional[np.ndarray]:
        """输入新检测, 返回滤波后的状态 [x,y,z,vx,vy,vz] (6,1)."""
        if not self._initialized:
            self._init_state(det)
            return self._x.copy()

        dt = det.timestamp - self._last_t
        if dt <= 0:
            dt = 0.016  # 兜底 60fps
        if dt > 0.5:  # 间隔太长, 重置
            self._init_state(det)
            return self._x.copy()

        z_meas = np.array([[det.pos[0]], [det.pos[1]], [det.pos[2]]], dtype=np.float64)

        # 预测
        F = self._transition_matrix(dt)
        B, u = self._control_input(dt)
        self._x = F @ self._x + B @ u
        self._P = F @ self._P @ F.T + self._process_noise_cov(dt)

        # 更新
        H = self._measurement_matrix()
        R = np.eye(3, dtype=np.float64) * self.cfg.measurement_noise
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        y = z_meas - H @ self._x  # innovation
        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ H) @ self._P

        self._last_t = det.timestamp
        self._add_history(det)
        return self._x.copy()

    @property
    def state(self) -> np.ndarray:
        """返回当前状态 [x, y, z, vx, vy, vz] (6,1)."""
        return self._x.copy()

    @property
    def position(self) -> Vec3:
        return (float(self._x[0, 0]), float(self._x[1, 0]), float(self._x[2, 0]))

    @property
    def velocity(self) -> Vec3:
        return (float(self._x[3, 0]), float(self._x[4, 0]), float(self._x[5, 0]))

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def ready(self) -> bool:
        return self._initialized and len(self.history) >= self.cfg.min_samples_for_fit

    def reset(self) -> None:
        self._x = np.zeros((6, 1), dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64) * 100.0
        self._initialized = False
        self._last_t = 0.0
        self.history.clear()

    # ----- internals ----------------------------------------------------------

    def _init_state(self, det: Detection3D) -> None:
        self._x[0, 0] = det.pos[0]
        self._x[1, 0] = det.pos[1]
        self._x[2, 0] = det.pos[2]
        self._x[3, 0] = 0.0
        self._x[4, 0] = 0.0
        self._x[5, 0] = 0.0
        self._P = np.eye(6, dtype=np.float64) * 10.0
        self._initialized = True
        self._last_t = det.timestamp
        self._add_history(det)

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _control_input(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """重力作为控制输入: u = [0, 0, g]"""
        B = np.zeros((6, 3), dtype=np.float64)
        B[2, 2] = 0.5 * dt * dt
        B[5, 2] = dt
        u = np.array([[0.0], [0.0], [self.g]], dtype=np.float64)
        return B, u

    def _measurement_matrix(self) -> np.ndarray:
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def _process_noise_cov(self, dt: float) -> np.ndarray:
        """过程噪声协方差 — 离散白噪声加速度模型."""
        q_p = self.cfg.process_noise_pos
        q_v = self.cfg.process_noise_vel
        Q = np.zeros((6, 6), dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt
        # 位置-位置
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q_p * dt3 / 3.0
        # 位置-速度
        Q[0, 3] = Q[3, 0] = Q[1, 4] = Q[4, 1] = Q[2, 5] = Q[5, 2] = q_p * dt2 / 2.0
        # 速度-速度
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q_v * dt
        return Q

    def _add_history(self, det: Detection3D) -> None:
        self.history.append(det)
        if len(self.history) > self.cfg.max_history:
            self.history = self.history[-self.cfg.max_history :]


# ---------------------------------------------------------------------------
# 抛物线落点求解
# ---------------------------------------------------------------------------


class BallisticSolver:
    """从 Kalman 滤波状态求解网球落点.

    解方程:  0.5 * g * t² + vz * t + (z - target_z) = 0
    取满足 t > 0 的根.
    """

    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg

    def solve(self, kf: TrajectoryFilter) -> Optional[LandingPoint]:
        """返回落点预测, 若无法可靠预测则返回 None."""
        if not kf.ready:
            return None

        x, y, z = kf.position
        vx, vy, vz = kf.velocity
        dz = z - self.cfg.target_height_m
        g = self.cfg.gravity

        # 解 0.5*g*t² + vz*t + dz = 0
        # → g*t² + 2*vz*t + 2*dz = 0
        a = 0.5 * g
        discriminant = vz * vz - 2.0 * g * dz

        if discriminant < 0:
            # 球不会到达目标高度
            return None

        sqrt_d = np.sqrt(discriminant)
        t1 = (-vz + sqrt_d) / g  # 注意 g 为负, g = -9.81
        t2 = (-vz - sqrt_d) / g

        # 选正根 (未来时间)
        candidates = [t for t in (t1, t2) if t > 1e-6]
        if not candidates:
            return None
        t_land = min(candidates)  # 取最小的正根 (球到达目标高度的最早时刻)

        if t_land > 3.0:  # 超过 3 秒太久, 不可靠
            return None

        # 预测落点位置
        x_land = x + vx * t_land
        y_land = y + vy * t_land

        # 置信度: 基于方差和预测距离
        P = kf.covariance
        pos_var = np.trace(P[:3, :3])  # 位置协方差迹
        confidence = max(0.0, min(1.0, 1.0 - pos_var / 5.0))

        if confidence < self.cfg.min_prediction_confidence:
            return None

        return LandingPoint(
            pos=(float(x_land), float(y_land), self.cfg.target_height_m),
            t_arrival=float(t_land),
            confidence=float(confidence),
        )
