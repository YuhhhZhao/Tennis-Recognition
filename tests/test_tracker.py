from __future__ import annotations

from tennis_robot_sim.data import Detection
from tennis_robot_sim.perception.tracker import PixelTracker


def test_tracker_update_predict_and_missing_frames():
    tracker = PixelTracker(max_missing_frames=5)
    tracker.update(Detection(0, 0.0, (10.0, 10.0), 5.0, 1.0), 0, 0.0)
    state = tracker.update(Detection(1, 0.1, (20.0, 10.0), 5.0, 1.0), 1, 0.1)
    assert state.center_px is not None
    assert state.velocity_px_s[0] > 0
    for i in range(2, 7):
        state = tracker.update(None, i, i * 0.1)
    assert state.missing_frames == 5
    assert state.center_px is not None


def test_tracker_rejects_large_outlier():
    tracker = PixelTracker(max_outlier_px=20)
    tracker.update(Detection(0, 0.0, (10.0, 10.0), 5.0, 1.0), 0, 0.0)
    state = tracker.update(Detection(1, 0.1, (500.0, 500.0), 5.0, 1.0), 1, 0.1)
    assert state.missing_frames == 1

