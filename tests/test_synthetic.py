#!/usr/bin/env python3
"""端到端合成视频测试 — 验证 3D 落点预测精度.

原理:
  1. 定义网球 3D 抛物线 (真值已知)
  2. 用相机模型反向投影 3D→2D, 生成每帧的 (u, v, radius_px)
  3. 渲染为合成帧, 逐帧喂入 pipeline
  4. 对比预测落点 vs 真值落点的误差

用法:
  python tests/test_synthetic.py
  python tests/test_synthetic.py --trajectory lob    # 高吊球
  python tests/test_synthetic.py --trajectory drive  # 平击球
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from tennis_tracker.config import (
    AppConfig, CameraConfig, ControlConfig, DisplayConfig,
    FilterConfig, GeometryConfig, HSVConfig, ROIConfig,
    TrajectoryConfig, UartConfig, YoloConfig,
)
from tennis_tracker.pipeline import TrackerPipeline
from tennis_tracker.prediction import CameraIntrinsics, CameraPose, load_calibration


# ═══════════════════════════════════════════════════════════════════════
#  预定义轨迹 (机器人坐标系: X前 Y左 Z上)
# ═══════════════════════════════════════════════════════════════════════

def make_trajectory_lob():
    """高吊球: (2.5, 0, 0.8)m → 垂直速度 5m/s → 落点约 2m 外"""
    x0, y0, z0 = 2.5, 0.0, 0.8
    vx, vy, vz = -1.0, 0.05, 5.0
    return x0, y0, z0, vx, vy, vz


def make_trajectory_drive():
    """平击球: (3.5, 0, 1.0)m → 水平速度 4m/s → 落点约 1.8m 外"""
    x0, y0, z0 = 3.5, 0.0, 1.0
    vx, vy, vz = -4.0, 0.1, 2.0
    return x0, y0, z0, vx, vy, vz


def make_trajectory_drop():
    """近距离下落: (1.5, 0, 0.6)m → 慢速 → 球在画面中更大, 检测更可靠"""
    x0, y0, z0 = 1.5, 0.0, 0.6
    vx, vy, vz = -0.5, 0.02, 1.0
    return x0, y0, z0, vx, vy, vz


G = -9.81


def compute_landing_truth(x0, y0, z0, vx, vy, vz):
    """解析解: 落点位置和到达时间."""
    disc = vz * vz - 2 * G * z0
    if disc < 0:
        return None
    t_land = (-vz - np.sqrt(disc)) / G
    x_land = x0 + vx * t_land
    y_land = y0 + vy * t_land
    return x_land, y_land, 0.0, t_land


# ═══════════════════════════════════════════════════════════════════════
#  反向投影: 机器人坐标 → 像素坐标
# ═══════════════════════════════════════════════════════════════════════

class Projector:
    def __init__(self, calib: CameraIntrinsics, pose: CameraPose, ball_diameter: float):
        self.calib = calib
        self.pose = pose
        self.diameter = ball_diameter

    def project(self, x_r, y_r, z_r):
        """机器人坐标 → (u, v, radius_px)"""
        H = self.pose.height_m
        # robot → camera (pitch=0, yaw=0):
        #   Xc = -Yr (机器人左 → 相机右)
        #   Yc = H - Zr (高度差, 相机下方向)
        #   Zc = Xr (机器人前 → 相机前)
        Xc = -y_r
        Yc = H - z_r
        Zc = x_r

        if Zc <= 0:
            return None

        u = self.calib.fx * Xc / Zc + self.calib.cx
        v = self.calib.fy * Yc / Zc + self.calib.cy
        r_px = self.calib.focal_mean * self.diameter / (2.0 * Zc)
        return u, v, r_px


# ═══════════════════════════════════════════════════════════════════════
#  生成合成帧
# ═══════════════════════════════════════════════════════════════════════

def generate_frames(projector, x0, y0, z0, vx, vy, vz,
                    fps=60, duration=1.5, width=640, height=360):
    """生成合成视频帧和真值."""
    n = int(fps * duration)
    frames = []
    gt = []  # [(t, x, y, z, u, v, r), ...]

    # 模拟草地场景: 上半天空渐变, 下半草地
    base = np.zeros((height, width, 3), dtype=np.uint8)
    # 天空 (上 60%)
    sky_h = int(height * 0.55)
    for i in range(sky_h):
        shade = int(180 + 40 * (1 - i / sky_h))  # 上亮下暗
        base[i, :] = (shade + 20, shade, shade - 20)
    # 草地 (下 40%)
    for i in range(sky_h, height):
        shade = int(50 + 20 * (i - sky_h) / (height - sky_h))
        base[i, :] = (20, shade + 20, 20)

    for i in range(n):
        t = i / fps
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t + 0.5 * G * t * t
        if z < 0:
            z = 0.0

        result = projector.project(x, y, z)
        if result is None:
            continue
        u, v, r = result

        frame = base.copy()

        if 0 <= u < width and 0 <= v < height and r > 1:
            center = (int(round(u)), int(round(v)))
            radius = max(1, int(round(r)))
            # 网球黄绿色填充 + 深色轮廓
            cv2.circle(frame, center, radius, (50, 220, 230), -1)
            cv2.circle(frame, center, radius, (30, 180, 200), 2)
            # 白色高光让轮廓更真实
            highlight = (max(2, radius // 3), max(2, radius // 3))
            cv2.ellipse(frame, center, highlight, -45, 0, 360, (80, 240, 250), -1)

        frames.append(frame)
        gt.append((t, x, y, z, u, v, r))

    return frames, gt


# ═══════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════

def make_config(ball_diameter, cam_height, cam_pitch, calib_path):
    return AppConfig(
        camera=CameraConfig(width=640, height=360, fps=60),
        yolo=YoloConfig(enabled=False, model_path="", class_name="tennis ball",
                        confidence=0.35, imgsz=640, device="cpu",
                        periodic_interval_ms=300, request_when_confidence_below=3),
        hsv=HSVConfig(lower=[25, 60, 60], upper=[55, 255, 255],
                      erode_iterations=1, dilate_iterations=2, close_iterations=1,
                      min_area=10, max_area=20000, min_circularity=0.40,
                      min_mask_fill_ratio=0.15, max_aspect_ratio=2.0),
        roi=ROIConfig(enabled=True, base_margin_px=60, velocity_margin_scale=1.5,
                      min_size_px=96, max_size_px=360),
        filter=FilterConfig(alpha=0.75, beta=0.20,
                            max_missing_frames=12, prediction_latency_ms=80),
        display=DisplayConfig(enabled=False, window_name=""),
        control=ControlConfig(enabled=False, deadband_px=20, max_command=1.0),
        geometry=GeometryConfig(calibration_path=calib_path or "configs/calibration.npz",
                                ball_diameter_m=ball_diameter,
                                camera_height_m=cam_height,
                                camera_pitch_deg=cam_pitch,
                                camera_yaw_deg=0.0),
        trajectory=TrajectoryConfig(gravity=G, min_samples_for_fit=6, max_history=60,
                                    target_height_m=0.0, process_noise_pos=0.01,
                                    process_noise_vel=0.5, measurement_noise=0.05,
                                    min_prediction_confidence=0.3),
        uart=UartConfig(enabled=False),
    )


# ═══════════════════════════════════════════════════════════════════════
#  运行 & 分析
# ═══════════════════════════════════════════════════════════════════════

def draw_tracker_overlay(frame, pipeline, gt_info, truth):
    """在帧上叠加跟踪器信息: 检测圈 + 预测落点 + 真值."""
    out = frame.copy()
    h, w = out.shape[:2]

    # 真值球位置 (绿色虚线)
    if gt_info is not None:
        t, x_g, y_g, z_g, u_g, v_g, r_g = gt_info
        gt_center = (int(round(u_g)), int(round(v_g)))
        cv2.circle(out, gt_center, int(round(r_g)), (0, 255, 0), 2)
        cv2.putText(out, "truth", (gt_center[0] + 10, gt_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 跟踪器检测位置 (红色实心)
    if pipeline.state.center is not None:
        cx, cy = pipeline.state.center
        det_center = (int(round(cx)), int(round(cy)))
        cv2.circle(out, det_center, max(3, int(pipeline.state.radius)),
                   (0, 0, 255), 2)
        cv2.putText(out, "track", (det_center[0] - 50, det_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 预测落点 (黄色)
    if pipeline.latest_landing is not None:
        lp = pipeline.latest_landing
        lx, ly, lz = lp.pos
        text = f"Land: ({lx:.2f},{ly:.2f}) t={lp.t_arrival:.2f}s"
        cv2.putText(out, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # 真值落点 (绿色)
    if truth is not None:
        tx, ty, tz, tt = truth
        cv2.putText(out, f"Truth: ({tx:.2f},{ty:.2f}) T={tt:.2f}s",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 3D 位置
    if pipeline.state.pos_3d is not None:
        x3, y3, z3 = pipeline.state.pos_3d
        cv2.putText(out, f"3D: ({x3:.2f},{y3:.2f},{z3:.2f})m",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 状态栏
    status = f"src={pipeline.state.source} conf={pipeline.state.confidence} miss={pipeline.state.missing_frames}"
    cv2.putText(out, status, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return out


def run_test(name, x0, y0, z0, vx, vy, vz, projector, cfg, save_dir=None):
    truth = compute_landing_truth(x0, y0, z0, vx, vy, vz)
    if truth is None:
        print(f"  SKIP: 落点无解 (球不落地?)")
        return None
    x_truth, y_truth, z_truth, t_truth = truth

    frames, gt = generate_frames(projector, x0, y0, z0, vx, vy, vz)
    print(f"  生成 {len(frames)} 帧, "
          f"v=[{min(g[5] for g in gt):.0f},{max(g[5] for g in gt):.0f}] "
          f"r=[{min(g[6] for g in gt):.1f},{max(g[6] for g in gt):.1f}]px")

    pipeline = TrackerPipeline(cfg, source="")
    pipeline.state.reset()
    pipeline.traj_filter.reset()

    detections = 0
    last_landing = None

    # 用于保存视频
    raw_writer = None
    track_writer = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_writer = cv2.VideoWriter(
            str(save_dir / f"{name}_raw.mp4"), fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
        track_writer = cv2.VideoWriter(
            str(save_dir / f"{name}_tracked.mp4"), fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))

    for i, frame in enumerate(frames):
        t, x_g, y_g, z_g, u_g, v_g, r_g = gt[i]
        pipeline._step(frame)

        if pipeline.state.center is not None:
            detections += 1
        if pipeline.latest_landing is not None:
            last_landing = (t, pipeline.latest_landing)

        # 保存帧
        if raw_writer is not None:
            # 原始帧 + 真值标注
            raw_frame = frame.copy()
            gt_center = (int(round(u_g)), int(round(v_g)))
            cv2.circle(raw_frame, gt_center, int(round(r_g)), (0, 255, 0), 2)
            cv2.putText(raw_frame, f"t={t:.2f}s z={z_g:.2f}m", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            raw_writer.write(raw_frame)

            # 跟踪器视角
            tracked = draw_tracker_overlay(frame, pipeline, gt[i], truth)
            track_writer.write(tracked)

    if raw_writer is not None:
        raw_writer.release()
        track_writer.release()
        print(f"  视频已保存: {save_dir / f'{name}_raw.mp4'}")
        print(f"              {save_dir / f'{name}_tracked.mp4'}")

    detect_rate = 100 * detections / len(frames)

    if last_landing is None:
        print(f"  FAIL: 未产生落点预测 (检测率 {detect_rate:.0f}%)")
        return None

    t_last, lp = last_landing
    px, py, pz = lp.pos

    # 关键: t_arrival 是从当前帧算起的剩余时间
    # 真值剩余时间 = t_truth - t_last
    remaining_truth = max(0, t_truth - t_last)
    t_arrival_err = abs(lp.t_arrival - remaining_truth)
    x_err = abs(px - x_truth)
    y_err = abs(py - y_truth)

    print(f"  检测率: {detect_rate:.0f}% ({detections}/{len(frames)})")
    print(f"  真值落点:  X={x_truth:.3f}m  Y={y_truth:.3f}m  "
          f"Z=0m  T_total={t_truth:.3f}s")
    print(f"  最终预测:  X={px:.3f}m  Y={py:.3f}m  "
          f"t_remain={lp.t_arrival:.3f}s  conf={lp.confidence:.3f}")
    print(f"  (帧时刻 t={t_last:.3f}s, 真值剩余={remaining_truth:.3f}s)")
    print(f"  误差:  dX={x_err*100:.0f}cm  dY={y_err*100:.0f}cm  "
          f"dt={t_arrival_err*1000:.0f}ms")

    # 评级
    if x_err < 0.20 and t_arrival_err < 0.10:
        grade = "GOOD"
    elif x_err < 0.50 and t_arrival_err < 0.25:
        grade = "OK"
    else:
        grade = "POOR"
    print(f"  评级: {grade}")

    return {
        "name": name, "grade": grade,
        "x_err": x_err, "y_err": y_err, "t_err": t_arrival_err,
        "detect_rate": detect_rate,
    }


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Synthetic video E2E test")
    parser.add_argument("--trajectory", choices=["lob", "drive", "drop", "all"],
                        default="all")
    parser.add_argument("--ball-diameter", type=float, default=0.093)
    parser.add_argument("--camera-height", type=float, default=0.258)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--calibration", default="configs/calibration.npz")
    parser.add_argument("--save", action="store_true",
                        help="Save synthetic videos to tests/output/")
    args = parser.parse_args()

    # 加载相机
    calib = load_calibration(PROJECT_ROOT / args.calibration)
    if calib is None:
        print("[WARN] calibration.npz not found, using default intrinsics")
        K = np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]], dtype=np.float64)
        calib = CameraIntrinsics(K=K, dist_coeffs=np.zeros(5))

    pose = CameraPose(height_m=args.camera_height,
                      pitch_deg=args.camera_pitch, yaw_deg=0.0)
    projector = Projector(calib, pose, args.ball_diameter)
    cfg = make_config(args.ball_diameter, args.camera_height,
                      args.camera_pitch, args.calibration)

    print("=" * 70)
    print("Synthetic End-to-End 3D Prediction Test")
    print(f"Calib: fx={calib.fx:.1f} fy={calib.fy:.1f}  "
          f"H={args.camera_height}m  pitch={args.camera_pitch}deg  "
          f"D={args.ball_diameter}m")
    print("=" * 70)

    trajectories = {
        "drop": ("近距离下落 (球大, 易检测)", make_trajectory_drop()),
        "lob": ("高吊球 (球小, 难度高)", make_trajectory_lob()),
        "drive": ("平击球 (球小/快, 最难)", make_trajectory_drive()),
    }

    if args.trajectory != "all":
        trajectories = {args.trajectory: trajectories[args.trajectory]}

    results = []
    for key, (desc, params) in trajectories.items():
        print(f"\n-- {desc} --")
        save_dir = PROJECT_ROOT / "tests" / "output" if args.save else None
        r = run_test(key, *params, projector, cfg, save_dir=save_dir)
        if r:
            results.append(r)

    print(f"\n{'=' * 70}")
    print(f"{'Trajectory':<16s} {'Grade':>6s}  {'dX(cm)':>8s}  {'dt(ms)':>8s}  {'detect%':>8s}")
    print("-" * 56)
    for r in results:
        print(f"{r['name']:<16s} {r['grade']:>6s}  {r['x_err']*100:8.1f}  "
              f"{r['t_err']*1000:8.0f}  {r['detect_rate']:8.0f}")

    good = sum(1 for r in results if r['grade'] == 'GOOD')
    ok = sum(1 for r in results if r['grade'] == 'OK')
    print(f"\nSummary: {good} GOOD, {ok} OK, "
          f"{len(results) - good - ok} POOR  (out of {len(results)})")

    # 关键说明
    print("""
注意事项:
  1. 合成视频使用 monotonic() 时间戳, 帧间 dt 很小,
     Kalman 使用默认 dt=16ms (60fps). 这会影响速度估计收敛速度.
  2. 早期帧球很小 (r<5px), HSV 轮廓检测不可靠,
     这正好反映真实场景中远距离网球难以精确检测的问题.
  3. 预测在球靠近后会快速收敛 — 看 t_arrival 的变化趋势.
  4. 单目深度估计本质上精度有限: 半径 1px 的误差 → ~10% 的深度误差.
""")


if __name__ == "__main__":
    main()
