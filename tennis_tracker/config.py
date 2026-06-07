from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


@dataclass
class CameraConfig:
    width: int
    height: int
    fps: int


@dataclass
class YoloConfig:
    enabled: bool
    model_path: str
    class_name: str
    confidence: float
    imgsz: int
    device: Union[int, str]
    periodic_interval_ms: int
    request_when_confidence_below: int


@dataclass
class HSVConfig:
    lower: List[int]
    upper: List[int]
    erode_iterations: int
    dilate_iterations: int
    min_area: float
    max_area: float
    min_circularity: float
    min_mask_fill_ratio: float
    max_aspect_ratio: float
    close_iterations: int = 2


@dataclass
class ROIConfig:
    enabled: bool
    base_margin_px: int
    velocity_margin_scale: float
    min_size_px: int
    max_size_px: int


@dataclass
class FilterConfig:
    alpha: float
    beta: float
    max_missing_frames: int
    prediction_latency_ms: int


@dataclass
class DisplayConfig:
    enabled: bool
    window_name: str


@dataclass
class ControlConfig:
    enabled: bool
    deadband_px: int
    max_command: float


@dataclass
class CalibrationConfig:
    path: str  # .npz 标定文件路径
    cols: int = 9  # 棋盘格内角点列数
    rows: int = 6  # 棋盘格内角点行数
    square_size_m: float = 0.025  # 方格边长 (米)


@dataclass
class GeometryConfig:
    calibration_path: str  # .npz 标定文件路径
    ball_diameter_m: float = 0.067
    camera_height_m: float = 0.3  # 相机距地面高度
    camera_pitch_deg: float = 20.0  # 俯仰角 (正值=向下)
    camera_yaw_deg: float = 0.0  # 偏航角


@dataclass
class TrajectoryConfig:
    gravity: float = -9.81
    min_samples_for_fit: int = 6
    max_history: int = 60
    target_height_m: float = 0.0
    process_noise_pos: float = 0.01
    process_noise_vel: float = 0.5
    measurement_noise: float = 0.05
    min_prediction_confidence: float = 0.3


@dataclass
class UartConfig:
    port: str = "/dev/ttyTHS1"  # Jetson UART1
    baudrate: int = 115200
    timeout_s: float = 0.05
    enabled: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig
    yolo: YoloConfig
    hsv: HSVConfig
    roi: ROIConfig
    filter: FilterConfig
    display: DisplayConfig
    control: ControlConfig
    geometry: GeometryConfig
    trajectory: TrajectoryConfig
    uart: UartConfig = field(default_factory=UartConfig)


def load_config(path: Union[str, Path]) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    return AppConfig(
        camera=CameraConfig(**data["camera"]),
        yolo=YoloConfig(**data["yolo"]),
        hsv=HSVConfig(**data["hsv"]),
        roi=ROIConfig(**data["roi"]),
        filter=FilterConfig(**data["filter"]),
        display=DisplayConfig(**data["display"]),
        control=ControlConfig(**data["control"]),
        geometry=GeometryConfig(**data.get("geometry", {})),
        trajectory=TrajectoryConfig(**data.get("trajectory", {})),
        uart=UartConfig(**data.get("uart", {})),
    )
