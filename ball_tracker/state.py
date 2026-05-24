from dataclasses import dataclass
from time import monotonic
from typing import Optional, Tuple


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


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
class TrackState:
    center: Optional[Point] = None
    velocity: Point = (0.0, 0.0)
    radius: float = 12.0
    confidence: int = 0
    missing_frames: int = 0
    last_update: float = 0.0
    source: str = "none"

    @property
    def ready(self) -> bool:
        return self.center is not None and self.confidence > 0

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


def now() -> float:
    return monotonic()

