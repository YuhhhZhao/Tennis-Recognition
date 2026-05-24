from __future__ import annotations

from dataclasses import dataclass
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
class AppConfig:
    camera: CameraConfig
    yolo: YoloConfig
    hsv: HSVConfig
    roi: ROIConfig
    filter: FilterConfig
    display: DisplayConfig
    control: ControlConfig


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
    )
