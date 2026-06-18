from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.perception.detector import ColorBallDetector
from tennis_robot_sim.sim import PinholeCamera, make_scenario
from tennis_robot_sim.sim.render import render_frame


def test_detector_finds_synthetic_ball_in_most_frames():
    cfg = load_config()
    cfg["sim"]["frames"] = 30
    scenario = make_scenario("default", cfg)
    camera = PinholeCamera.from_config(cfg)
    detector = ColorBallDetector(cfg)
    hits = 0
    for idx, state in enumerate(scenario.states):
        frame = render_frame(camera, state.position, cfg)
        if detector.detect(frame, idx, state.timestamp) is not None:
            hits += 1
    assert hits / len(scenario.states) >= 0.9

