from dataclasses import dataclass, field
from time import monotonic
from typing import List, Optional, Tuple


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]
Vec3 = Tuple[float, float, float]


@dataclass
class Detection:
    center: Point
    bbox: BBox
    confidence: float
    source: str
    timestamp: float
    radius: float = 0.0

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0


@dataclass
class Detection3D:
    """机器人坐标系中的 3D 检测结果"""

    pos: Vec3  # (Xr, Yr, Zr) — 前方, 左方, 上方 (米)
    confidence: float
    timestamp: float
    radius_px: float = 0.0  # 原始像素半径, 用于调试

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0


@dataclass
class LandingPoint:
    """预测落点, 机器人坐标系"""

    pos: Vec3  # (Xr, Yr, Zr) — 落点位置 (米)
    t_arrival: float  # 预计到达时间 (秒)
    confidence: float = 0.0  # 0-1 预测置信度

    def __repr__(self) -> str:
        x, y, z = self.pos
        return (
            f"Landing(x={x:.2f}, y={y:.2f}, z={z:.2f}, "
            f"t={self.t_arrival:.2f}s, conf={self.confidence:.2f})"
        )


@dataclass
class TrackState:
    center: Optional[Point] = None
    velocity: Point = (0.0, 0.0)
    radius: float = 12.0
    confidence: int = 0
    missing_frames: int = 0
    last_update: float = 0.0
    source: str = "none"

    # 调试
    last_radius_px: float = 0.0  # 最近一次检测的原始像素半径

    # 3D 扩展
    pos_3d: Optional[Vec3] = None  # 机器人坐标系中的当前位置
    vel_3d: Vec3 = (0.0, 0.0, 0.0)
    history_3d: List[Detection3D] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.center is not None and self.confidence > 0

    @property
    def ready_3d(self) -> bool:
        return self.pos_3d is not None and self.confidence > 0

    def mark_missing(self) -> None:
        self.missing_frames += 1
        self.confidence = max(0, self.confidence - 1)

    def reset(self) -> None:
        self.center = None
        self.velocity = (0.0, 0.0)
        self.radius = 12.0
        self.confidence = 0
        self.missing_frames = 0
        self.last_update = 0.0
        self.source = "none"
        self.pos_3d = None
        self.vel_3d = (0.0, 0.0, 0.0)
        self.history_3d.clear()


def now() -> float:
    return monotonic()

