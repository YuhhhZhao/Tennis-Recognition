from __future__ import annotations

import numpy as np

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import Detection
from tennis_robot_sim.estimation.trajectory import TrajectoryEstimator
from tennis_robot_sim.sim import PinholeCamera, make_scenario


def test_trajectory_predicts_default_landing():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    scenario = make_scenario("default", cfg)
    estimator = TrajectoryEstimator(cfg, camera)
    prediction = None
    for frame_id, state in enumerate(scenario.states[:40]):
        projected = camera.world_to_pixel(state.position, cfg["ball"]["radius_m"])
        assert projected is not None
        det = Detection(frame_id, state.timestamp, (projected[0], projected[1]), projected[2], 1.0)
        estimator.update_detection(det)
        prediction = estimator.predict(now=state.timestamp)
    assert prediction is not None
    error = ((prediction.landing_xy[0] - scenario.landing_xy[0]) ** 2 + (prediction.landing_xy[1] - scenario.landing_xy[1]) ** 2) ** 0.5
    assert error < 0.05


def test_trajectory_handles_noisy_observations_and_missing_frames():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    scenario = make_scenario("default", cfg)
    estimator = TrajectoryEstimator(cfg, camera)
    rng = np.random.default_rng(5)
    prediction = None
    for frame_id, state in enumerate(scenario.states[:45]):
        if frame_id % 7 == 0:
            continue
        projected = camera.world_to_pixel(state.position, cfg["ball"]["radius_m"])
        assert projected is not None
        noisy_center = (projected[0] + rng.normal(0.0, 0.8), projected[1] + rng.normal(0.0, 0.8))
        noisy_radius = projected[2] * (1.0 + rng.normal(0.0, 0.015))
        det = Detection(frame_id, state.timestamp, noisy_center, noisy_radius, 0.8)
        estimator.update_detection(det)
        prediction = estimator.predict(now=state.timestamp)
    assert prediction is not None
    error = ((prediction.landing_xy[0] - scenario.landing_xy[0]) ** 2 + (prediction.landing_xy[1] - scenario.landing_xy[1]) ** 2) ** 0.5
    assert error < 0.5
