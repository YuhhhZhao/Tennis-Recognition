from __future__ import annotations

import math

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import RobotState
from tennis_robot_sim.robot.controller import PoseController
from tennis_robot_sim.robot.sim_robot import SimRobot


def test_controller_converges_to_target():
    cfg = load_config()
    robot = SimRobot(cfg)
    controller = PoseController(cfg)
    target = RobotState(1.5, 0.0, 0.0)
    for _ in range(120):
        cmd = controller.step(robot.state, target, 0.05)
        robot.step(cmd, 0.05)
    assert math.hypot(robot.state.x - target.x, robot.state.y - target.y) < 0.2


def test_controller_commands_are_bounded():
    cfg = load_config()
    cmd = PoseController(cfg).step(RobotState(0.0, 0.0, 0.0), RobotState(10.0, 10.0, 0.0), 0.1)
    assert abs(cmd.v) <= cfg["robot"]["max_speed_mps"]
    assert abs(cmd.omega) <= cfg["robot"]["max_angular_rate_radps"]

