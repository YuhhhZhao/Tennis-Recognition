from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import ControlCommand
from tennis_robot_sim.robot.sim_robot import SimRobot


def test_robot_forward_rotation_and_saturation():
    cfg = load_config()
    robot = SimRobot(cfg)
    state = robot.step(ControlCommand(100.0, 100.0), 0.1)
    assert state.x > 0
    assert abs(state.v) <= cfg["robot"]["max_speed_mps"]
    assert abs(state.omega) <= cfg["robot"]["max_angular_rate_radps"]


def test_robot_stays_inside_bounds():
    cfg = load_config()
    robot = SimRobot(cfg)
    for _ in range(300):
        state = robot.step(ControlCommand(10.0, 0.0), 0.1)
    assert cfg["court"]["x_min"] <= state.x <= cfg["court"]["x_max"]

