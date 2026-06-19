"""3D 落点预测子包

geometry    — 单目相机 3D 几何 (像素 → 相机 → 机器人)
calibration — 相机标定 (棋盘格) 加载/保存
trajectory  — 3D Kalman 轨迹滤波 + 抛物线落点求解
"""

from .calibration import (
    calibrate_from_images,
    generate_chessboard_png,
    load_calibration,
    save_calibration,
)
from .geometry import (
    CameraIntrinsics,
    CameraPose,
    camera_to_robot,
    depth_from_ball_radius,
    detect_to_robot_3d,
    pixel_to_camera_frame,
    pixel_to_ray,
)
from .trajectory import BallisticSolver, TrajectoryFilter

__all__ = [
    # geometry
    "CameraIntrinsics",
    "CameraPose",
    "camera_to_robot",
    "depth_from_ball_radius",
    "detect_to_robot_3d",
    "pixel_to_camera_frame",
    "pixel_to_ray",
    # calibration
    "calibrate_from_images",
    "generate_chessboard_png",
    "load_calibration",
    "save_calibration",
    # trajectory
    "BallisticSolver",
    "TrajectoryFilter",
]
