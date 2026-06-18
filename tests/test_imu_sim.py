from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import ControlCommand
from tennis_robot_sim.imu.sim_imu import IMUSimulator
from tennis_robot_sim.robot.sim_robot import SimRobot


def test_imu_deterministic_seed_and_zero_motion_noise():
    cfg = load_config()
    cfg["imu"]["accel_noise_std"] = 0.0
    cfg["imu"]["gyro_noise_std"] = 0.0
    robot = SimRobot(cfg)
    sample = IMUSimulator(cfg).sample(robot.state, ControlCommand(0.0, 0.0), 0.1)
    assert sample.accel_mps2 == (0.0, 0.0, 0.0)
    assert sample.gyro_radps == (0.0, 0.0, 0.0)


def test_constant_rotation_gyro_sign():
    cfg = load_config()
    cfg["imu"]["accel_noise_std"] = 0.0
    cfg["imu"]["gyro_noise_std"] = 0.0
    robot = SimRobot(cfg)
    state = robot.step(ControlCommand(0.0, 1.0), 0.1)
    sample = IMUSimulator(cfg).sample(state, ControlCommand(0.0, 1.0), 0.1)
    assert sample.yaw_rate_radps > 0

