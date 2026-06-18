from __future__ import annotations

import math
from typing import Optional

from tennis_robot_sim.data import Detection, TrackState


class PixelTracker:
    def __init__(
        self,
        alpha: float = 0.72,
        velocity_alpha: float = 0.45,
        max_missing_frames: int = 5,
        max_outlier_px: float = 120.0,
    ):
        self.alpha = float(alpha)
        self.velocity_alpha = float(velocity_alpha)
        self.max_missing_frames = int(max_missing_frames)
        self.max_outlier_px = float(max_outlier_px)
        self.state = TrackState(frame_id=-1, timestamp=0.0, center_px=None)

    def reset(self) -> None:
        self.state = TrackState(frame_id=-1, timestamp=0.0, center_px=None)

    def update(self, detection: Optional[Detection], frame_id: int, timestamp: float) -> TrackState:
        if detection is None:
            return self._predict_missing(frame_id, timestamp)
        if self.state.center_px is None:
            self.state = TrackState(
                frame_id=frame_id,
                timestamp=timestamp,
                center_px=detection.center_px,
                velocity_px_s=(0.0, 0.0),
                radius_px=detection.radius_px,
                confidence=detection.confidence,
                missing_frames=0,
            )
            return self.state

        dt = max(1e-6, timestamp - self.state.timestamp)
        predicted = (
            self.state.center_px[0] + self.state.velocity_px_s[0] * dt,
            self.state.center_px[1] + self.state.velocity_px_s[1] * dt,
        )
        innovation = math.hypot(detection.center_px[0] - predicted[0], detection.center_px[1] - predicted[1])
        if innovation > self.max_outlier_px and self.state.confidence > 0.3:
            return self._predict_missing(frame_id, timestamp)

        center = (
            self.alpha * detection.center_px[0] + (1.0 - self.alpha) * predicted[0],
            self.alpha * detection.center_px[1] + (1.0 - self.alpha) * predicted[1],
        )
        measured_velocity = ((center[0] - self.state.center_px[0]) / dt, (center[1] - self.state.center_px[1]) / dt)
        velocity = (
            self.velocity_alpha * measured_velocity[0] + (1.0 - self.velocity_alpha) * self.state.velocity_px_s[0],
            self.velocity_alpha * measured_velocity[1] + (1.0 - self.velocity_alpha) * self.state.velocity_px_s[1],
        )
        self.state = TrackState(
            frame_id=frame_id,
            timestamp=timestamp,
            center_px=center,
            velocity_px_s=velocity,
            radius_px=0.7 * detection.radius_px + 0.3 * self.state.radius_px,
            confidence=min(1.0, max(detection.confidence, self.state.confidence * 0.95)),
            missing_frames=0,
        )
        return self.state

    def _predict_missing(self, frame_id: int, timestamp: float) -> TrackState:
        if self.state.center_px is None:
            self.state = TrackState(frame_id=frame_id, timestamp=timestamp, center_px=None, missing_frames=self.state.missing_frames + 1)
            return self.state
        dt = max(0.0, timestamp - self.state.timestamp)
        missing = self.state.missing_frames + 1
        center = (
            self.state.center_px[0] + self.state.velocity_px_s[0] * dt,
            self.state.center_px[1] + self.state.velocity_px_s[1] * dt,
        )
        confidence = max(0.0, self.state.confidence * (0.75 ** missing))
        if missing > self.max_missing_frames:
            center = None
            confidence = 0.0
        self.state = TrackState(
            frame_id=frame_id,
            timestamp=timestamp,
            center_px=center,
            velocity_px_s=self.state.velocity_px_s,
            radius_px=self.state.radius_px,
            confidence=confidence,
            missing_frames=missing,
        )
        return self.state

