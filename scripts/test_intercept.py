"""
手持网球 → 检测 → 3D定位 → 小车移动 → 定位追踪

流程:
  1. 手持网球对准相机 3 秒, 采集位置
  2. 计算 3D 落点 (x, y)
  3. 发送 TARGET 指令给 ESP32
  4. 小车移动过程中实时追踪 IMU + 里程计
  5. 到达目标附近停车

安全: 速度上限 0.5m/s, 超时 5s 强制停车
"""

import sys
import time
import math

sys.path.insert(0, ".")

import cv2
import serial
from tennis_tracker.config import load_config
from tennis_tracker.detection import YOLODetector
from tennis_tracker.prediction import (
    CameraIntrinsics,
    CameraPose,
    detect_to_robot_3d,
    load_calibration,
)
from tennis_tracker.control.uart_bridge import UartBridge, StepCounts
from tennis_tracker.config import UartConfig
from tennis_robot_sim.estimation.odometry import OdomTracker
from tennis_robot_sim.imu import WitMotionIMU, ComplementaryLocalizer
from tennis_robot_sim.data import ControlCommand as SimCommand, RobotState as SimState

ESP32_PORT = "/dev/ttyCH341USB0"
IMU_PORT = "/dev/ttyACM0"
BAUD = 115200
MAX_SPEED = 0.5  # m/s, 安全上限
TIMEOUT = 5.0    # 移动超时 (秒)
TARGET_TOLERANCE = 0.15  # 到达判定 (米)


def main():
    cfg = load_config("configs/app.yaml")

    # ── 1. 连接硬件 ──
    print("[1/5] 连接硬件...")
    # ESP32
    uart_cfg = UartConfig(port=ESP32_PORT, baudrate=BAUD, timeout_s=0.05, enabled=True)
    uart = UartBridge(uart_cfg)
    if not uart.open():
        print("ESP32 连接失败"); return

    # IMU
    imu = WitMotionIMU(IMU_PORT, BAUD)
    if not imu.open():
        uart.close(); return

    # 相机
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    if not cap.isOpened():
        print("相机连接失败"); imu.close(); uart.close(); return

    # YOLO
    detector = YOLODetector(cfg.yolo)

    # 3D
    calib_path = "configs/calibration.npz"
    calib = load_calibration(calib_path)
    camera_pose = CameraPose(
        height_m=cfg.geometry.camera_height_m,
        pitch_deg=cfg.geometry.camera_pitch_deg,
        yaw_deg=cfg.geometry.camera_yaw_deg,
    )

    # 定位
    odom = OdomTracker()
    localizer = ComplementaryLocalizer({
        "robot": {"start_pose": [0.0, 0.0, 0.0]},
        "imu": {"yaw_complementary_alpha": 0.92},
    })
    localizer.reset()

    # 复位步数
    uart.send_reset_steps()
    time.sleep(0.1)

    print("硬件全部就绪")

    # ── 2. 预热 YOLO + 检测网球 3D 位置 ──
    print("\n[2/5] 预热YOLO模型...")
    # 预热: 先跑一帧 (慢)
    ok, frame = cap.read()
    if ok:
        _ = detector.detect(frame)
    print("  预热完成")

    print("[2/5] 检测网球位置 (请手持网球对准相机, 采集10次检测)...")
    positions = []
    attempts = 0
    while len(positions) < 10:
        ok, frame = cap.read()
        if not ok:
            continue
        attempts += 1
        det = detector.detect(frame)
        if det is not None:
            u, v = det.center
            pos = detect_to_robot_3d(
                u, v, det.radius, calib, camera_pose,
                real_diameter_m=cfg.geometry.ball_diameter_m,
            )
            if pos is not None:
                positions.append(pos)
                print(f"  [{len(positions):2d}/10] ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m  "
                      f"conf={det.confidence:.2f}  (共{attempts}帧)")
                cv2.circle(frame, (int(u), int(v)), int(det.radius), (0,255,0), 2)
        time.sleep(0.05)

    cap.release()

    if len(positions) < 5:
        print("检测样本不足, 请重试"); imu.close(); uart.close(); return

    # 取中位数
    import numpy as np
    pos_array = np.array(positions)
    target_x = float(np.median(pos_array[:, 0]))
    target_y = float(np.median(pos_array[:, 1]))
    target_z = float(np.median(pos_array[:, 2]))
    print(f"\n  目标位置: x={target_x:.2f}m  y={target_y:.2f}m  z={target_z:.2f}m")

    # ── 3. 计算速度指令 ──
    print("\n[3/5] 计算移动策略...")
    # 简单比例控制: vx = target_x / t_approach
    # 路程 sqrt(x²+y²), 速度 0.3m/s
    distance = math.sqrt(target_x**2 + target_y**2)
    speed = min(MAX_SPEED, max(0.2, distance / 3.0))
    t_approach = distance / speed

    # 如果目标很近 (<0.2m), 不用动
    if distance < TARGET_TOLERANCE:
        print(f"  目标在 {distance:.2f}m 内, 无需移动")
        imu.close(); uart.close(); return

    print(f"  距离: {distance:.2f}m  速度: {speed:.2f}m/s  预计: {t_approach:.1f}s")

    # ── 4. 执行移动 + 追踪 ──
    print(f"\n[4/5] 小车移动 (超时 {TIMEOUT}s)...")
    print(f"{'t':>5s}  {'里程x':>7s}  {'里程y':>7s}  {'里程yaw':>7s}  {'IMU_yaw':>8s}  {'距目标':>7s}")
    print("-" * 60)

    start_time = time.time()
    loc_last_t = start_time
    odom_last_query = start_time
    records = []

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            dt = now - loc_last_t
            loc_last_t = now

            # 超时
            if elapsed > TIMEOUT:
                print(f"\n  超时 {TIMEOUT}s, 停车")
                break

            # 到目标判断
            px, py, pyaw = odom.pose
            dist_to_target = math.sqrt((target_x - px)**2 + (target_y - py)**2)
            if dist_to_target < TARGET_TOLERANCE and elapsed > 1.0:
                print(f"\n  到达目标! 剩余距离 {dist_to_target:.2f}m")
                break

            # 发送 TARGET 指令
            remaining_t = max(0.5, t_approach - elapsed)
            uart.send_target(target_x, target_y, remaining_t)

            # ── 定位更新 ──
            # IMU
            imu_sample = imu.get_sample()

            # 里程计 (每 100ms)
            if now - odom_last_query > 0.1:
                steps_data = uart.send_ping()
                odom_last_query = now
                if steps_data is not None:
                    odom.update(steps_data.to_tuple())

            # 融合
            cmd = SimCommand(v=speed, omega=imu_sample.yaw_rate_radps)
            localizer.predict(cmd, dt)
            localizer.update_imu(imu_sample, dt)
            ox, oy, oyaw = odom.pose
            localizer.update_odometry(SimState(x=ox, y=oy, yaw=oyaw), weight=0.15)
            state = localizer.get_state()

            # 每秒打印
            if records and int(elapsed) != int(elapsed - dt) if dt > 0 else True:
                print(f"{elapsed:4.1f}s  {state.x:+6.2f}  {state.y:+6.2f}  "
                      f"{state.yaw*57.3:+6.1f}°  {imu.get_angle()[2]:+7.1f}°  "
                      f"{dist_to_target:+6.2f}")

            records.append((elapsed, state.x, state.y, state.yaw, dist_to_target))
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n中断!")
    finally:
        # STOP
        for _ in range(5):
            uart.send_stop()
            time.sleep(0.05)

    # ── 5. 结果 ──
    print(f"\n[5/5] 结果")
    print("-" * 60)
    final_pose = localizer.get_state()
    print(f"  目标:       ({target_x:.2f}, {target_y:.2f})m")
    print(f"  最终位姿:   ({final_pose.x:.2f}, {final_pose.y:.2f})m  yaw={final_pose.yaw*57.3:.0f}°")
    final_dist = math.sqrt((target_x - final_pose.x)**2 + (target_y - final_pose.y)**2)
    print(f"  最终距目标: {final_dist:.2f}m")
    print(f"  判定: {'✅ 成功' if final_dist < 0.3 else '⚠️ 未到达'}")

    imu.close()
    uart.close()


if __name__ == "__main__":
    main()
