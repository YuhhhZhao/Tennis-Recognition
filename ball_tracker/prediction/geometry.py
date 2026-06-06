"""单目相机 3D 几何 — 像素 → 相机坐标系 → 机器人坐标系

依赖: numpy, 以及已知的网球直径 0.067m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 标准网球直径 (米)
TENNIS_BALL_DIAMETER_M = 0.067


@dataclass
class CameraIntrinsics:
    """针孔相机内参"""

    K: np.ndarray  # 3x3 camera matrix
    dist_coeffs: np.ndarray  # (5,) or (8,) distortion coefficients

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def focal_mean(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def is_valid(self) -> bool:
        return self.fx > 0 and self.fy > 0


@dataclass
class CameraPose:
    """相机在机器人上的安装位姿

    Attributes
    ----------
    height_m : float
        相机光心距地面高度 (米), 沿机器人 Z 轴向上为正.
    pitch_deg : float
        俯仰角 (度). 正值 = 相机向下倾斜 (光轴指向地面).
        0° = 光轴水平, 90° = 垂直向下.
    yaw_deg : float
        偏航角 (度). 正值 = 顺时针 (从上方看). 0° = 机器人正前方.
    """

    height_m: float = 0.3
    pitch_deg: float = 20.0
    yaw_deg: float = 0.0

    @property
    def pitch_rad(self) -> float:
        return np.deg2rad(self.pitch_deg)

    @property
    def yaw_rad(self) -> float:
        return np.deg2rad(self.yaw_deg)


# ---------------------------------------------------------------------------
# 深度估计
# ---------------------------------------------------------------------------


def depth_from_ball_radius(
    radius_px: float,
    K: CameraIntrinsics,
    real_diameter_m: float = TENNIS_BALL_DIAMETER_M,
) -> float:
    """从网球像素半径估算深度 (Z, 米).

    针孔模型:   apparent_size_px  =  f_px * real_size_m  /  Z
    →  Z  =  f_px * real_size_m  /  apparent_size_px

    radius_px 为 0 或负时返回无穷大 (无法估计).
    """
    if radius_px <= 0:
        return float("inf")
    diameter_px = 2.0 * radius_px
    return K.focal_mean * real_diameter_m / diameter_px


# ---------------------------------------------------------------------------
# 像素 → 相机坐标系
# ---------------------------------------------------------------------------


def pixel_to_ray(u: float, v: float, K: CameraIntrinsics) -> np.ndarray:
    """从像素坐标 (u, v) 算出归一化方向向量 (在相机坐标系中)."""
    x = (u - K.cx) / K.fx
    y = (v - K.cy) / K.fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def pixel_to_camera_frame(
    u: float,
    v: float,
    radius_px: float,
    K: CameraIntrinsics,
    real_diameter_m: float = TENNIS_BALL_DIAMETER_M,
) -> Tuple[float, float, float]:
    """像素 (u, v, radius_px) → 相机坐标系 3D 坐标 (Xc, Yc, Zc).

    相机坐标系 (OpenCV 惯例):
      Xc → 右,  Yc → 下,  Zc → 前方 (光轴方向).
    """
    Z = depth_from_ball_radius(radius_px, K, real_diameter_m)
    X = (u - K.cx) * Z / K.fx
    Y = (v - K.cy) * Z / K.fy
    return (float(X), float(Y), float(Z))


# ---------------------------------------------------------------------------
# 相机坐标系 → 机器人坐标系
# ---------------------------------------------------------------------------


def _build_transform_matrix(pose: CameraPose) -> np.ndarray:
    """构建 相机→机器人 的 3x3 旋转矩阵.

    机器人坐标系:
      Xr → 前方,  Yr → 左方,  Zr → 上方

    推导:
      1. 相机绕自身 X 轴旋转 pitch_rad (向下为正)
      2. 相机绕机器人 Z 轴旋转 yaw_rad
      3. 坐标系重新映射:  Xr←Zc, Yr←-Xc, Zr←-Yc
    """
    sp, cp = np.sin(pose.pitch_rad), np.cos(pose.pitch_rad)
    sy, cy = np.sin(pose.yaw_rad), np.cos(pose.yaw_rad)

    # R_pitch: rotate around camera X axis
    R_pitch = np.array(
        [[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64
    )

    # R_yaw: rotate around robot Z axis (applied to the remapped axes)
    R_yaw = np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64
    )

    # 坐标系映射: [Xr, Yr, Zr] = R_remap @ [Xc, Yc, Zc]
    #   Xr = Zc   (相机前方 → 机器人前方)
    #   Yr = -Xc  (相机右方 → 机器人左方)
    #   Zr = -Yc  (相机下方 → 机器人上方)
    R_remap = np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64
    )

    return R_yaw @ R_remap @ R_pitch


def camera_to_robot(
    Xc: float,
    Yc: float,
    Zc: float,
    pose: CameraPose,
) -> Tuple[float, float, float]:
    """相机坐标系 → 机器人坐标系.

    机器人坐标系:
      Xr → 前方 (m),  Yr → 左方 (m),  Zr → 上方 (m, 地面 = 0).
    """
    R = _build_transform_matrix(pose)
    cam = np.array([Xc, Yc, Zc], dtype=np.float64)
    robot = R @ cam
    robot[2] += pose.height_m  # 相机高度补偿
    return (float(robot[0]), float(robot[1]), float(robot[2]))


# ---------------------------------------------------------------------------
# 主入口: 像素检测 → 机器人 3D 坐标
# ---------------------------------------------------------------------------


def detect_to_robot_3d(
    u: float,
    v: float,
    radius_px: float,
    K: CameraIntrinsics,
    pose: CameraPose,
    real_diameter_m: float = TENNIS_BALL_DIAMETER_M,
) -> Optional[Tuple[float, float, float]]:
    """从 2D 检测结果直接算出机器人坐标系中的 3D 位置.

    返回 None 表示深度不可靠 (球太小/太远/半径为零).
    """
    if radius_px <= 0 or not K.is_valid():
        return None
    Xc, Yc, Zc = pixel_to_camera_frame(u, v, radius_px, K, real_diameter_m)
    if Zc <= 0 or np.isinf(Zc):
        return None
    return camera_to_robot(Xc, Yc, Zc, pose)
