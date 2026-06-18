from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import ControlCommand
from tennis_robot_sim.imu import ComplementaryLocalizer, IMUSimulator
from tennis_robot_sim.robot.sim_robot import SimRobot


def test_imu_localization_improves_yaw_over_biased_odometry():
    cfg = load_config()
    cfg["imu"]["accel_noise_std"] = 0.0
    cfg["imu"]["gyro_noise_std"] = 0.0
    truth_robot = SimRobot(cfg)
    imu = IMUSimulator(cfg)
    fused = ComplementaryLocalizer(cfg)
    odom_yaw = 0.0
    dt = 0.02
    true_cmd = ControlCommand(0.5, 0.8)
    biased_cmd = ControlCommand(0.5, 1.2)
    for _ in range(100):
        truth = truth_robot.step(true_cmd, dt)
        sample = imu.sample(truth, true_cmd, dt)
        fused.predict(biased_cmd, dt)
        fused.update_imu(sample, dt)
        odom_yaw += biased_cmd.omega * dt
    fused_error = abs(fused.get_state().yaw - truth_robot.state.yaw)
    odom_error = abs(odom_yaw - truth_robot.state.yaw)
    assert fused_error < odom_error

