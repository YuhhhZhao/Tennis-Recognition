from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass
class Detection:
    frame_id: int
    timestamp: float
    center_px: Vec2
    radius_px: float
    confidence: float
    source: str = "color"


@dataclass
class TrackState:
    frame_id: int
    timestamp: float
    center_px: Optional[Vec2]
    velocity_px_s: Vec2 = (0.0, 0.0)
    radius_px: float = 0.0
    confidence: float = 0.0
    missing_frames: int = 0


@dataclass
class LandingPrediction:
    landing_xy: Vec2
    time_to_land: float
    uncertainty: float
    confidence: float
    debug_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    omega: float = 0.0
    timestamp: float = 0.0


@dataclass
class ControlCommand:
    v: float
    omega: float


@dataclass
class InterceptionPlan:
    target_pose: RobotState
    reachable: bool
    eta: float
    distance: float
    reason: str


@dataclass
class IMUSample:
    timestamp: float
    accel_mps2: Vec3
    gyro_radps: Vec3
    yaw_rate_radps: float
    ground_truth_pose: Optional[RobotState] = None

