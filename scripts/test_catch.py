"""
接球测试: 手持网球 → 抛出 → 轨迹预测落点 → 小车缓慢移动到落点

流程:
  1. 手持网球对准相机, YOLO检测初始位置
  2. 抛出网球, 持续检测+追踪轨迹
  3. TrajectoryFilter + BallisticSolver 预测落点
  4. 小车以低速移动到预测落点
  5. 不需要真的接到球, 只需要到达落点附近

安全: 速度 ≤ 0.3m/s, 超时 5s 停车
"""

import sys
import time
import math

sys.path.insert(0, ".")

import cv2
import numpy as np
from tennis_tracker.config import load_config
from tennis_tracker.detection import YOLODetector
from tennis_tracker.prediction import (
    BallisticSolver,
    CameraIntrinsics,
    CameraPose,
    TrajectoryFilter,
    detect_to_robot_3d,
    load_calibration,
)
from tennis_tracker.state import Detection, Detection3D
from tennis_tracker.control.uart_bridge import UartBridge
from tennis_tracker.config import UartConfig
from tennis_robot_sim.estimation.odometry import OdomTracker
from tennis_robot_sim.imu import WitMotionIMU, ComplementaryLocalizer
from tennis_robot_sim.data import ControlCommand as SimCommand, RobotState as SimState

ESP32_PORT = "/dev/ttyCH341USB0"
IMU_PORT = "/dev/ttyACM0"
BAUD = 115200
MAX_SPEED = 0.05   # 极慢速移动 m/s
TIMEOUT = 6.0      # 移动超时
TARGET_TOLERANCE = 0.15  # 到达判定


def main():
    cfg = load_config("configs/app.yaml")

    # ── 连接硬件 ──
    print("[1/4] 连接硬件...")
    uart_cfg = UartConfig(port=ESP32_PORT, baudrate=BAUD, timeout_s=0.05, enabled=True)
    uart = UartBridge(uart_cfg)
    uart.open()

    imu = WitMotionIMU(IMU_PORT, BAUD)
    imu.open()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)

    detector = YOLODetector(cfg.yolo)

    calib = load_calibration("configs/calibration.npz")
    camera_pose = CameraPose(
        height_m=cfg.geometry.camera_height_m,
        pitch_deg=cfg.geometry.camera_pitch_deg,
        yaw_deg=cfg.geometry.camera_yaw_deg,
        offset_x_m=cfg.geometry.camera_offset_x_m,
        offset_y_m=cfg.geometry.camera_offset_y_m,
        offset_z_m=cfg.geometry.camera_offset_z_m,
    )

    # 轨迹预测
    traj_filter = TrajectoryFilter(cfg.trajectory)
    ballistic = BallisticSolver(cfg.trajectory)

    # 定位
    odom = OdomTracker()
    localizer = ComplementaryLocalizer({
        "robot": {"start_pose": [0.0, 0.0, 0.0]},
        "imu": {"yaw_complementary_alpha": 0.92},
    })
    localizer.reset()
    uart.send_reset_steps()
    time.sleep(0.1)

    print("硬件就绪")

    # ── 2. 预热 YOLO ──
    print("\n[2/4] 预热 YOLO...")
    ok, frame = cap.read()
    if ok:
        _ = detector.detect(frame)
    print("  预热完成")

    # ── 3. 检测 + 轨迹预测 ──
    print("\n[3/4] 轨迹采集 (手持网球, 然后抛出! 采集5秒)...")
    print(f"  {'帧':>4s}  {'球3D位置':>20s}  {'速度':>15s}  {'落点预测':>20s}")
    print("-" * 75)

    history_3d = []
    latest_landing = None
    start_t = time.time()
    thrown = False
    pre_throw_history = []
    wait_counter = 0  # 等待消息计数器

    raw_frames = 0
    raw_detections = 0

    while time.time() - start_t < 8.0:
        ok, frame = cap.read()
        if not ok:
            continue
        raw_frames += 1

        t0 = time.time()
        det = detector.detect(frame)
        if det is None:
            if raw_frames % 50 == 0:
                print(f"  [{raw_frames}帧 0检测] 请把球放在相机前", end='\r')
            continue
        raw_detections += 1

        u, v = det.center
        pos = detect_to_robot_3d(
            u, v, det.radius, calib, camera_pose,
            real_diameter_m=cfg.geometry.ball_diameter_m,
        )
        if pos is None:
            continue

        # 野值过滤: 帧间位置跳 >0.5m 则跳过
        if pre_throw_history or history_3d:
            if history_3d:
                prev_pos = history_3d[-1].pos
            else:
                prev_pos = pre_throw_history[-1][0]  # (pos, ts) tuple
            jump = math.sqrt(
                (pos[0] - prev_pos[0])**2 +
                (pos[1] - prev_pos[1])**2 +
                (pos[2] - prev_pos[2])**2
            )
            if jump > 0.5:
                continue

        # ── 抛球检测 (速度突变) ──
        pre_throw_history.append((pos, time.time()))
        if len(pre_throw_history) > 8:
            pre_throw_history = pre_throw_history[-8:]

        if not thrown and len(pre_throw_history) >= 6:
            # 用前后3帧计算速度: v = (p_end - p_start) / dt
            p_start, t_start = pre_throw_history[0]
            p_end, t_end = pre_throw_history[-1]
            dt = t_end - t_start
            if dt > 0.05:
                vx = (p_end[0] - p_start[0]) / dt
                vy = (p_end[1] - p_start[1]) / dt
                vz = (p_end[2] - p_start[2]) / dt
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                # 球速 >0.8m/s → 已被抛出
                if speed > 0.8:
                    thrown = True
                    print(f"\n  *** 检测到抛球! 速度={speed:.1f}m/s ***")
                    traj_filter.reset()
                    history_3d.clear()
                    for p, _ in pre_throw_history[-3:]:
                        d3d = Detection3D(
                            pos=p, confidence=det.confidence,
                            timestamp=time.time(), radius_px=det.radius,
                        )
                        history_3d.append(d3d)
                        traj_filter.update(d3d)
                    continue

        if not thrown:
            wait_counter += 1
            if wait_counter % 20 == 0:
                elapsed = time.time() - start_t
                print(f"  等待抛球... 球在 ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m  "
                      f"速度阈值未触发 ({elapsed:.0f}s)")
            time.sleep(0.03)
            continue

        # ── 抛出后: 正常轨迹追踪 ──
        d3d = Detection3D(
            pos=pos,
            confidence=det.confidence,
            timestamp=time.time(),
            radius_px=det.radius,
        )
        history_3d.append(d3d)
        if len(history_3d) > cfg.trajectory.max_history:
            history_3d = history_3d[-cfg.trajectory.max_history:]

        traj_filter.update(d3d)
        landing = ballistic.solve(traj_filter)
        if landing is not None:
            latest_landing = landing

        # 打印
        vel = traj_filter.velocity
        vel_str = f"({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f})"
        if latest_landing:
            lx, ly, lz = latest_landing.pos
            land_str = f"({lx:.2f},{ly:.2f},{lz:.2f})m t={latest_landing.t_arrival:.1f}s"
        else:
            land_str = "(采集数据中...)"

        elapsed = time.time() - start_t
        print(f"  {len(history_3d):4d}  ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m  "
              f"{vel_str}  {land_str}")

        time.sleep(0.03)

    cap.release()

    if len(history_3d) < 5:
        print("轨迹数据不足"); imu.close(); uart.close(); return

    if latest_landing is None:
        # 用最后一次检测作为目标
        target_x = history_3d[-1].pos[0]
        target_y = history_3d[-1].pos[1]
        print("\n无法预测落点, 用最后检测位置作为目标")
    else:
        target_x, target_y, _ = latest_landing.pos
        print(f"\n落点预测: ({target_x:.2f}, {target_y:.2f})m  "
              f"到达时间: {latest_landing.t_arrival:.1f}s")

    # ── 4. 移动 ──
    distance = math.sqrt(target_x**2 + target_y**2)
    speed = min(MAX_SPEED, max(0.05, distance / 10.0))

    print(f"\n   落点: ({target_x:.2f}, {target_y:.2f})m  距离: {distance:.2f}m  速度: {speed:.2f}m/s")
    print("   2秒后开始移动, 请整理线缆...")
    time.sleep(2.0)

    print(f"\n[4/4] 小车移动中...")
    print(f"{'t':>5s}  {'融合位姿':>20s}  {'距目标':>7s}")
    print("-" * 45)

    start_t = time.time()
    loc_last_t = start_t
    odom_last_query = start_t

    try:
        while True:
            now = time.time()
            elapsed = now - start_t
            dt = now - loc_last_t
            loc_last_t = now

            if elapsed > TIMEOUT:
                print(f"\n超时 {TIMEOUT}s, 停车")
                break

            px, py, _ = odom.pose
            dist = math.sqrt((target_x - px)**2 + (target_y - py)**2)
            if dist < TARGET_TOLERANCE and elapsed > 1.0:
                print(f"\n到达! 距离目标 {dist:.2f}m")
                break

            # TARGET 指令
            remaining_t = max(0.5, 3.0 - elapsed)
            uart.send_target(target_x, target_y, remaining_t)

            # 定位
            imu_sample = imu.get_sample()
            if now - odom_last_query > 0.1:
                steps = uart.send_ping()
                odom_last_query = now
                if steps is not None:
                    odom.update(steps.to_tuple())

            cmd = SimCommand(v=speed, omega=imu_sample.yaw_rate_radps)
            localizer.predict(cmd, dt)
            localizer.update_imu(imu_sample, dt)
            ox, oy, oyaw = odom.pose
            localizer.update_odometry(SimState(x=ox, y=oy, yaw=oyaw), weight=0.15)
            state = localizer.get_state()

            if int(elapsed) != int(elapsed - dt) if dt > 0 else True:
                print(f"{elapsed:4.1f}s  ({state.x:.2f},{state.y:.2f}) yaw={state.yaw*57.3:.0f}°  {dist:+.2f}m")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n中断!")
    finally:
        for _ in range(5):
            uart.send_stop(); time.sleep(0.05)

    state = localizer.get_state()
    final_dist = math.sqrt((target_x - state.x)**2 + (target_y - state.y)**2)
    print(f"\n目标: ({target_x:.2f}, {target_y:.2f})m")
    print(f"到达: ({state.x:.2f}, {state.y:.2f})m  误差: {final_dist:.2f}m")
    print(f"{'✅ 成功' if final_dist < 0.3 else '⚠️ 未到达'}")

    imu.close()
    uart.close()


if __name__ == "__main__":
    main()
