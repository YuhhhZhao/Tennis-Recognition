"""Tennis ball tracking package — Jetson edge deployment.

子包:
  detection/   — 2D 识别 (HSV + YOLO + Alpha-Beta 滤波)
  prediction/  — 3D 落点预测 (单目几何 + Kalman + 抛物线求解)
  control/     — 电控 + 通信 (CarController + UART 桥)
"""

# 向后兼容: 从新子包位置重导出
from .detection import (
    AlphaBetaFilter,
    AsyncYOLOWorker,
    HSVTracker,
    YOLODetector,
    clamp_point,
)
from .control import CarController, ControlCommand, UartBridge, UartConfig
from .prediction import (
    BallisticSolver,
    CameraIntrinsics,
    CameraPose,
    TrajectoryFilter,
    detect_to_robot_3d,
    load_calibration,
)

__all__ = [
    # config
    "config",
    # state
    "state",
    # pipeline
    "pipeline",
    # detection
    "detection",
    "AlphaBetaFilter",
    "AsyncYOLOWorker",
    "HSVTracker",
    "YOLODetector",
    "clamp_point",
    # prediction
    "prediction",
    "BallisticSolver",
    "CameraIntrinsics",
    "CameraPose",
    "TrajectoryFilter",
    "detect_to_robot_3d",
    "load_calibration",
    # control
    "control",
    "CarController",
    "ControlCommand",
    "UartBridge",
    "UartConfig",
]
