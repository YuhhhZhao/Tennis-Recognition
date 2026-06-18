"""
IMU 定位 vs 理想运动轨迹对比

测试流程:
  1. 静止校准 IMU 偏置
  2. 发送 VEL 指令驱动小车（前进 + 转弯 + 前进）
  3. IMU 积分航向 vs 命令航向 vs 融合航向
  4. 输出对比结果和误差

安全: 低速 (0.2m/s), 短距 (2s/段), STOP 立即停车
"""

import sys
import time
import math

sys.path.insert(0, ".")

import serial
import numpy as np
from tennis_robot_sim.data import ControlCommand
from tennis_robot_sim.imu import ComplementaryLocalizer, WitMotionIMU

ESP32_PORT = "/dev/ttyCH341USB0"
ESP32_BAUD = 115200
IMU_PORT = "/dev/ttyACM0"
IMU_BAUD = 115200

CFG = {
    "robot": {"start_pose": [0.0, 0.0, 0.0]},
    "imu": {"yaw_complementary_alpha": 0.92},
}

# 测试动作序列: (vx, vy, omega, duration_s, label)
# 约束: 仅右转(omega>0), 每次位移<0.5m (0.2m/s*2s=0.4m)
ACTIONS = [
    (0.20, 0.0, 0.0, 2.0, "前进 0.2m/s (0.4m)"),
    (0.0, 0.0, -0.5, 1.0, "右转 -0.5rad/s"),
    (0.20, 0.0, 0.0, 2.0, "前进 0.2m/s (0.4m)"),
    (0.0, 0.0, 0.0, 0.5, "停止"),
]


def send_vel(ser, vx, vy, w):
    """发送 VEL 指令给 ESP32"""
    cmd = f"VEL {vx:.3f} {vy:.3f} {w:.3f}\n"
    ser.write(cmd.encode())
    ser.flush()


def main():
    # ── 连接 ESP32 ──
    try:
        esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=0.5)
        time.sleep(0.3)
        esp32.reset_input_buffer()
        print("[ESP32] 已连接")
    except Exception as e:
        print(f"ESP32 连接失败: {e}")
        return

    # ── 连接 IMU ──
    imu = WitMotionIMU(IMU_PORT, IMU_BAUD)
    if not imu.open():
        esp32.close()
        return

    # ── 初始化定位器 ──
    localizer = ComplementaryLocalizer(CFG)
    localizer.reset()

    # ── 校准 IMU 陀螺偏置 ──
    print("\n校准 IMU 陀螺偏置 (2s)...")
    gyro_bias = 0.0
    calib_samples = 0
    calib_deadline = time.time() + 2.0
    while time.time() < calib_deadline:
        s = imu.get_sample()
        gyro_bias += s.yaw_rate_radps
        calib_samples += 1
        time.sleep(0.001)
    gyro_bias /= max(calib_samples, 1)
    print(f"  陀螺偏置: {gyro_bias:.6f} rad/s ({gyro_bias*57.3:.4f}°/s)")

    # ── 执行动作序列并记录 ──
    records = []  # (t, cmd_v, cmd_omega, imu_gyro, imu_yaw_int, fused_yaw, odom_yaw)

    print("\n开始运动测试...\n")
    print(f"{'时间':>6s}  {'动作':<18s}  {'gyroZ°/s':>9s}  {'IMU航向°':>9s}  {'融合航向°':>9s}  {'命令航向°':>9s}")
    print("-" * 74)

    odom_yaw = 0.0
    imu_yaw_int = 0.0
    start_time = time.time()
    last_t = start_time
    action_start = start_time

    try:
        for vx, vy, omega, dur, label in ACTIONS:
            action_start = time.time()
            action_deadline = action_start + dur

            while time.time() < action_deadline:
                now = time.time()
                dt = now - last_t
                last_t = now

                # 发送指令
                send_vel(esp32, vx, vy, omega)

                # IMU 采样
                sample = imu.get_sample()
                gyro_z = sample.yaw_rate_radps - gyro_bias  # 去偏置

                # 里程计积分（用命令，带5%偏置模拟真实误差）
                biased_omega = omega * 1.05 + 0.02  # 故意偏置
                odom_yaw += biased_omega * dt
                localizer.predict(ControlCommand(vx, biased_omega), dt)

                # IMU 积分
                imu_yaw_int += gyro_z * dt

                # 融合
                localizer.update_imu(sample, dt)

                # 每秒打印
                if records and int(now - start_time) != int((now - dt) - start_time):
                    fused = localizer.get_state()
                    cmd_yaw = localizer.state.yaw  # 当前定位器认为的里程计航向
                    print(
                        f"{now-start_time:5.1f}s  "
                        f"{label:<18s}  "
                        f"{gyro_z*57.3:+8.3f}  "
                        f"{imu_yaw_int*57.3:+8.2f}  "
                        f"{fused.yaw*57.3:+8.2f}  "
                        f"{odom_yaw*57.3:+8.2f}"
                    )

                records.append((
                    now - start_time, vx, omega, gyro_z, imu_yaw_int,
                    localizer.get_state().yaw, odom_yaw
                ))

                # 控制循环速率
                elapsed = time.time() - now
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\n中断!")
    finally:
        # STOP x5
        for _ in range(5):
            esp32.write(b"STOP\n")
            esp32.flush()
            time.sleep(0.05)
        esp32.close()
        imu.close()

    # ── 结果分析 ──
    if len(records) < 10:
        print("数据不足")
        return

    t, cv, co, gz, iy, fy, oy = [np.array(x) for x in zip(*records)]

    # 找每段动作的航向变化
    print("\n" + "=" * 74)
    print("动作段航向变化对比:")
    print(f"{'动作':<18s}  {'IMU积分°':>9s}  {'融合°':>9s}  {'命令°':>9s}  {'偏置命令°':>9s}")
    print("-" * 74)

    action_idx = 0
    current_t = 0.0
    total_imu_yaw = 0.0
    total_fused_yaw = 0.0
    total_cmd_yaw = 0.0
    total_biased_yaw = 0.0

    for vx, vy, omega, dur, label in ACTIONS:
        seg_start = current_t
        seg_end = current_t + dur
        mask = (t >= seg_start) & (t < seg_end)
        if mask.sum() == 0:
            current_t = seg_end
            continue

        seg_imu_delta = iy[mask][-1] - iy[mask][0]
        seg_fused_delta = fy[mask][-1] - fy[mask][0]
        seg_cmd_delta = omega * dur
        seg_biased_delta = omega * 1.05 * dur + 0.02 * dur

        total_imu_yaw += seg_imu_delta
        total_fused_yaw += seg_fused_delta
        total_cmd_yaw += seg_cmd_delta
        total_biased_yaw += seg_biased_delta

        print(
            f"{label:<18s}  "
            f"{seg_imu_delta*57.3:+8.2f}  "
            f"{seg_fused_delta*57.3:+8.2f}  "
            f"{seg_cmd_delta*57.3:+8.2f}  "
            f"{seg_biased_delta*57.3:+8.2f}"
        )
        current_t = seg_end

    print("-" * 74)
    print(
        f"{'总计':<18s}  "
        f"{total_imu_yaw*57.3:+8.2f}  "
        f"{total_fused_yaw*57.3:+8.2f}  "
        f"{total_cmd_yaw*57.3:+8.2f}  "
        f"{total_biased_yaw*57.3:+8.2f}"
    )

    # 误差
    imu_vs_cmd = abs(total_imu_yaw - total_cmd_yaw) * 57.3
    fused_vs_cmd = abs(total_fused_yaw - total_cmd_yaw) * 57.3
    biased_vs_cmd = abs(total_biased_yaw - total_cmd_yaw) * 57.3

    print(f"\n误差分析 (vs 理想命令):")
    print(f"  IMU积分 vs 命令:  {imu_vs_cmd:.2f}° {'✅' if imu_vs_cmd < 5 else '❌'}")
    print(f"  融合定位 vs 命令: {fused_vs_cmd:.2f}° {'✅' if fused_vs_cmd < 5 else '❌'}")
    print(f"  偏置里程 vs 命令: {biased_vs_cmd:.2f}° (5%偏置+噪声)")
    print(f"  IMU比偏置里程改进: {biased_vs_cmd - imu_vs_cmd:.2f}°")


if __name__ == "__main__":
    main()
