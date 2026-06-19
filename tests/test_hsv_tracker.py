from __future__ import annotations

import cv2
import numpy as np

from tennis_tracker.config import HSVConfig, ROIConfig
from tennis_tracker.detection.hsv_tracker import HSVTracker
from tennis_tracker.state import TrackState, now


BALL_BGR = (50, 220, 230)


def make_tracker(roi_enabled: bool = True) -> HSVTracker:
    return HSVTracker(
        HSVConfig(
            lower=[25, 60, 60],
            upper=[55, 255, 255],
            erode_iterations=1,
            dilate_iterations=2,
            close_iterations=1,
            min_area=20,
            max_area=20000,
            min_circularity=0.4,
            min_mask_fill_ratio=0.15,
            max_aspect_ratio=2.0,
        ),
        ROIConfig(
            enabled=roi_enabled,
            base_margin_px=10,
            velocity_margin_scale=1.0,
            min_size_px=32,
            max_size_px=80,
        ),
    )


def test_hsv_tracker_ignores_invalid_blob_and_keeps_valid_ball():
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 12), (165, 24), BALL_BGR, -1)
    cv2.circle(frame, (90, 70), 12, BALL_BGR, -1)

    det = make_tracker(roi_enabled=False).track(frame, TrackState())

    assert det is not None
    assert abs(det.center[0] - 90) < 2.0
    assert abs(det.center[1] - 70) < 2.0


def test_hsv_tracker_rejects_border_touching_yellow_wall():
    frame = np.zeros((140, 180, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (179, 80), BALL_BGR, -1)
    cv2.circle(frame, (95, 110), 10, BALL_BGR, -1)

    det = make_tracker(roi_enabled=False).track(frame, TrackState())

    assert det is not None
    assert abs(det.center[0] - 95) < 2.0
    assert abs(det.center[1] - 110) < 2.0


def test_hsv_roi_follows_predicted_motion_after_missing_frames():
    frame = np.zeros((120, 140, 3), dtype=np.uint8)
    cv2.circle(frame, (80, 60), 8, BALL_BGR, -1)
    state = TrackState(
        center=(40.0, 60.0),
        velocity=(160.0, 0.0),
        radius=8.0,
        confidence=5,
        missing_frames=2,
        last_update=now() - 0.25,
        source="HSV",
    )

    det = make_tracker(roi_enabled=True).track(frame, state)

    assert det is not None
    assert abs(det.center[0] - 80) < 2.0
    assert abs(det.center[1] - 60) < 2.0
