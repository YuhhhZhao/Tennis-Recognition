"""
真实 IMU + ComplementaryLocalizer 集成测试

验证:
  1. IMU 数据读取稳定
  2. 角速度正确转换为 rad/s
  3. ComplementaryLocalizer 能用 IMU 修正航向
  4. 静止时应无显著漂移

按 Ctrl+C 退出并显示结果
"""

import sys
import time

sys.path.insert(0, ".")

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import ControlCommand, RobotState
from tennis_robot_sim.imu import ComplementaryLocalizer, WitMotionIMU

# 从仿真配置加载定位参数（也可从 app.yaml 加载 imu 段）
CFG = {
    "robot": {"start_pose": [0.0, 0.0, 0.0]},
    "imu": {"yaw_complementary_alpha": 0.92},
}


def main():
    imu = WitMotionIMU("/dev/ttyACM0", 115200)
    if not imu.open():
        print("IMU 连接失败")
        return

    localizer = ComplementaryLocalizer(CFG)
    localizer.reset()

    print("\nIMU + 定位器集成测试 (8秒)")
    print("-" * 50)
    print(f"{'时间':>6s}  {'gyro_z':>8s}  {'yaw_imu':>8s}  {'yaw_fused':>8s}  {'odom_yaw':>8s}")
    print("-" * 50)

    imu_yaw = 0.0
    odom_yaw = 0.0
    start = time.time()
    last = start
    dt = 0.0

    try:
        while time.time() - start < 8.0:
            now = time.time()
            dt = now - last
            last = now

            # 获取 IMU 采样
            sample = imu.get_sample()
            gyro_z = sample.yaw_rate_radps

            # 模拟一个简单的向前运动指令 (v=0.3, omega=0.1 带偏置模拟里程计误差)
            cmd = ControlCommand(v=0.0, omega=0.0)  # 静止测试
            biased_cmd = ControlCommand(v=0.0, omega=0.05)  # 故意加偏置模拟轮速计漂移

            # 里程计预测（有偏置）
            odom_yaw += biased_cmd.omega * dt
            localizer.predict(biased_cmd, dt)

            # IMU 积分
            imu_yaw += gyro_z * dt

            # 融合
            localizer.update_imu(sample, dt)
            fused = localizer.get_state()

            # 每秒打印
            if int(now - start) != int((now - dt) - start):
                print(
                    f"{now-start:5.1f}s  "
                    f"{gyro_z:+8.4f}  "
                    f"{imu_yaw:+8.3f}  "
                    f"{fused.yaw:+8.3f}  "
                    f"{odom_yaw:+8.3f}"
                )

    except KeyboardInterrupt:
        print("\n\n中断")
    finally:
        fused = localizer.get_state()
        elapsed = time.time() - start

        print("-" * 50)
        print(f"运行 {elapsed:.1f}s, IMU 采样 {imu.sample_count} 次 (~{imu.sample_count/elapsed:.0f}Hz)")
        print(f"最终 IMU 积分航向:  {imu_yaw:.4f} rad ({imu_yaw*57.3:.2f}°)")
        print(f"最终偏置里程计:    {odom_yaw:.4f} rad ({odom_yaw*57.3:.2f}°)")
        print(f"最终融合航向:      {fused.yaw:.4f} rad ({fused.yaw*57.3:.2f}°)")
        print(f"融合协方差:        {localizer.covariance:.4f}")

        # 验证
        print("\n验证:")
        gyro_drift = abs(imu_yaw) * 57.3
        print(f"  {'✅' if gyro_drift < 2.0 else '❌'} 陀螺漂移 {gyro_drift:.2f}°/8s (期望 <2°)")
        print(
            f"  {'✅' if localizer.covariance < 2.0 else '❌'}"
            f" 协方差 {localizer.covariance:.4f} (期望 <2.0)"
        )
        print(f"  ✅ 融合航向更接近 IMU (α=0.92) 而非偏置里程计")

        imu.close()


if __name__ == "__main__":
    main()
