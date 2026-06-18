from __future__ import annotations

import pytest

from tennis_robot_sim.config import load_config
from tennis_robot_sim.robot.interface import RealRobotInterface, SimRobotInterface
from tennis_robot_sim.robot.sim_robot import SimRobot


def test_sim_interface_instantiates_without_hardware():
    cfg = load_config()
    iface = SimRobotInterface(SimRobot(cfg))
    assert iface.robot.state.x == 0.0


def test_real_interface_refuses_without_enable_flag():
    cfg = load_config()
    cfg["safety"]["enable_real_hardware"] = False
    with pytest.raises(RuntimeError, match="real hardware is disabled"):
        RealRobotInterface(cfg)

