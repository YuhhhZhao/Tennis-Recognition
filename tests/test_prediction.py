#!/usr/bin/env python3
"""3D 预测链路单元测试 — 纯数值, 无需摄像头/模型.

测试覆盖:
  1. 几何模块: 像素→相机→机器人坐标
  2. 轨迹滤波: 6D Kalman 收敛性
  3. 落点求解: 抛物线预测精度
  4. 端到端: detect_to_robot_3d → TrajectoryFilter → BallisticSolver

参数匹配当前 configs/app.yaml:
  - camera_pitch_deg: 0.0  (水平安装)
  - camera_height_m: 0.258
  - ball_diameter_m: 0.093

用法:
  cd Tennis-Recognition
  python tests/test_prediction.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_tracker.config import TrajectoryConfig
from tennis_tracker.prediction import (
    BallisticSolver,
    CameraIntrinsics,
    CameraPose,
    TrajectoryFilter,
    depth_from_ball_radius,
    detect_to_robot_3d,
    pixel_to_camera_frame,
)
from tennis_tracker.state import Detection3D


# ── 参数: 与 tracker.yaml 保持一致 ─────────────────────────────────────

REAL_DIAMETER = 0.093      # geometry.ball_diameter_m
CAMERA_HEIGHT = 0.258       # geometry.camera_height_m
CAMERA_PITCH   = 0.0        # geometry.camera_pitch_deg
CAMERA_YAW     = 0.0        # geometry.camera_yaw_deg


def _make_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=180.0) -> CameraIntrinsics:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return CameraIntrinsics(K=K, dist_coeffs=np.zeros(5, dtype=np.float64))


def _make_pose() -> CameraPose:
    return CameraPose(
        height_m=CAMERA_HEIGHT,
        pitch_deg=CAMERA_PITCH,
        yaw_deg=CAMERA_YAW,
    )


def _make_traj_cfg() -> TrajectoryConfig:
    return TrajectoryConfig(
        gravity=-9.81,
        min_samples_for_fit=6,
        max_history=60,
        target_height_m=0.0,
        process_noise_pos=0.01,
        process_noise_vel=0.5,
        measurement_noise=0.05,
        min_prediction_confidence=0.3,
    )


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


# ── Test 1: 深度估计 (修正球径 0.093m) ──────────────────────────────────

def test_depth_estimation():
    print("\n-- Test 1: depth_from_ball_radius (D=0.093m) --")
    K = _make_intrinsics()

    # 公式: Z = f * D / (2 * r)
    # r=20px → Z = 500 * 0.093 / 40 = 1.1625m
    d = depth_from_ball_radius(20.0, K, real_diameter_m=REAL_DIAMETER)
    expected = 500.0 * REAL_DIAMETER / 40.0
    check(f"depth ~{expected:.2f}m @20px", abs(d - expected) < 0.01, f"got {d:.3f}")

    # r=50px → Z = 500 * 0.093 / 100 = 0.465m
    d2 = depth_from_ball_radius(50.0, K, real_diameter_m=REAL_DIAMETER)
    expected2 = 500.0 * REAL_DIAMETER / 100.0
    check(f"depth ~{expected2:.2f}m @50px", abs(d2 - expected2) < 0.01, f"got {d2:.3f}")

    # 边界
    check("radius=0 -> inf", math.isinf(depth_from_ball_radius(0.0, K)))
    check("radius<0 -> inf", math.isinf(depth_from_ball_radius(-5.0, K)))


# ── Test 2: 像素 → 相机坐标系 ──────────────────────────────────────────

def test_pixel_to_camera():
    print("\n-- Test 2: pixel_to_camera_frame --")
    K = _make_intrinsics()

    # 图像中心 → Xc≈0, Yc≈0
    X, Y, Z = pixel_to_camera_frame(320.0, 180.0, 25.0, K, real_diameter_m=REAL_DIAMETER)
    check("center -> Xc~0", abs(X) < 0.01, f"X={X:.3f}")
    check("center -> Yc~0", abs(Y) < 0.01, f"Y={Y:.3f}")
    check("center -> Zc>0", Z > 0, f"Z={Z:.3f}")

    # 右侧 → Xc>0 (相机坐标系: X右)
    Xr, Yr, Zr = pixel_to_camera_frame(420.0, 180.0, 25.0, K, real_diameter_m=REAL_DIAMETER)
    check("right -> Xc>0", Xr > 0, f"X={Xr:.3f}")

    # 下方 → Yc>0 (相机坐标系: Y下)
    Xd, Yd, Zd = pixel_to_camera_frame(320.0, 260.0, 25.0, K, real_diameter_m=REAL_DIAMETER)
    check("bottom -> Yc>0", Yd > 0, f"Y={Yd:.3f}")


# ── Test 3: 相机 → 机器人坐标系 (水平相机, pitch=0) ────────────────────

def test_camera_to_robot():
    print("\n-- Test 3: camera_to_robot (pitch=0, 水平安装) --")
    K = _make_intrinsics()
    pose = _make_pose()

    # 水平相机 (pitch=0, yaw=0):
    #   坐标系映射: Xr=Zc(前), Yr=-Xc(左), Zr=-Yc+H(上)
    #   H = camera_height = 0.258m

    # --- 情况 A: 球在图像中心, 同一高度 ---
    # u=320, v=180 (cy=180) → Yc=0 → Zr = H = 0.258m
    # 球悬在相机同高度, 不是地面
    pos = detect_to_robot_3d(320.0, 180.0, 20.0, K, pose, real_diameter_m=REAL_DIAMETER)
    check("center ball -> pos not None", pos is not None)
    if pos:
        x, y, z = pos
        check(f"center ball Zr ~ H={CAMERA_HEIGHT}m", abs(z - CAMERA_HEIGHT) < 0.02,
              f"z={z:.3f}")
        print(f"      球在中心: ({x:.2f}, {y:.2f}, {z:.2f}) m  (同相机高度)")

    # --- 情况 B: 球在地面 ---
    # 水平相机看地面: 球在图像下半部, Yc > 0
    # 满足 Zr = H - Yc ≈ 0, 即 Yc ≈ H = 0.258
    # Yc = (v - cy) * Zc / fy
    # 取 r=15px → Zc = 500*0.093/30 = 1.55m
    # v ≈ 180 + 0.258*500/1.55 = 180 + 83.2 = 263
    r_px = 15.0
    Zc_expected = K.focal_mean * REAL_DIAMETER / (2.0 * r_px)  # = 1.55m
    v_ground = 180.0 + CAMERA_HEIGHT * K.fy / Zc_expected  # ≈ 263

    pos_g = detect_to_robot_3d(320.0, v_ground, r_px, K, pose, real_diameter_m=REAL_DIAMETER)
    check(f"ground ball (v={v_ground:.0f}) -> pos not None", pos_g is not None)
    if pos_g:
        x, y, z = pos_g
        check(f"Zr ~ 0 (地面)", abs(z) < 0.02, f"z={z:.3f}")
        check(f"Xr ~ Zc={Zc_expected:.2f}m (前方)", abs(x - Zc_expected) < 0.05,
              f"x={x:.3f}")
        print(f"      地面球: ({x:.2f}, {y:.2f}, {z:.2f}) m  (z~0=地面)")

    # --- 情况 C: 球在空中 (比相机低, 但未着地) ---
    # v=220, r=18px → Zc = 500*0.093/36 = 1.292m
    # Yc = (220-180)*1.292/500 = 0.103
    # Zr = 0.258 - 0.103 = 0.155m
    pos_mid = detect_to_robot_3d(320.0, 220.0, 18.0, K, pose, real_diameter_m=REAL_DIAMETER)
    check("mid-air ball -> pos not None", pos_mid is not None)
    if pos_mid:
        x, y, z = pos_mid
        check(f"mid-air Zr between 0 and H", 0 < z < CAMERA_HEIGHT,
              f"z={z:.3f}")
        print(f"      空中球: ({x:.2f}, {y:.2f}, {z:.2f}) m")

    # --- 情况 D: radius=0 -> None ---
    check("radius=0 -> None",
          detect_to_robot_3d(320.0, 180.0, 0.0, K, pose) is None)


# ── Test 4: 6D Kalman 轨迹滤波 ─────────────────────────────────────────

def test_trajectory_filter():
    print("\n-- Test 4: TrajectoryFilter (6D Kalman) --")
    cfg = _make_traj_cfg()
    kf = TrajectoryFilter(cfg)

    check("initially not ready", not kf.ready)

    # 模拟网球: 从 (2, 0, 1.5)m 以水平速度 (3, 0.5, 4)m/s 抛出
    # 重力 -9.81 在 z 方向
    x0, y0, z0 = 2.0, 0.0, 1.5
    vx0, vy0, vz0 = 3.0, 0.5, 4.0

    dt = 0.05
    for i in range(30):
        t = i * dt
        true_x = x0 + vx0 * t
        true_y = y0 + vy0 * t
        true_z = z0 + vz0 * t + 0.5 * (-9.81) * t * t

        noise = np.random.randn(3) * 0.02
        det = Detection3D(
            pos=(true_x + noise[0], true_y + noise[1], true_z + noise[2]),
            confidence=0.8, timestamp=t, radius_px=20.0,
        )
        kf.update(det)

    check("ready after 30 updates", kf.ready)

    px, py, pz = kf.position
    vx, vy, vz = kf.velocity

    # 收敛检查
    t_end = 29 * dt  # = 1.45s
    true_x_end = x0 + vx0 * t_end
    true_y_end = y0 + vy0 * t_end
    check(f"pos X ~ {true_x_end:.1f}m", abs(px - true_x_end) < 0.3, f"got {px:.3f}")
    check(f"pos Y ~ {true_y_end:.1f}m", abs(py - true_y_end) < 0.2, f"got {py:.3f}")
    check(f"vel X ~ {vx0:.1f} m/s", abs(vx - vx0) < 0.5, f"got {vx:.3f}")
    check(f"vel Y ~ {vy0:.1f} m/s", abs(vy - vy0) < 0.3, f"got {vy:.3f}")

    print(f"      估计位置: ({px:.3f}, {py:.3f}, {pz:.3f}) m")
    print(f"      估计速度: ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s")


# ── Test 5: 抛物线落点预测 ─────────────────────────────────────────────

def test_ballistic_solver():
    print("\n-- Test 5: BallisticSolver (落点预测) --")
    cfg = _make_traj_cfg()
    kf = TrajectoryFilter(cfg)
    solver = BallisticSolver(cfg)

    # 不够样本 -> None
    check("not ready -> None", solver.solve(kf) is None)

    # 球从 (4, 0, 1.5)m 以 (-2, 0, 3)m/s 运动
    # 解析解: 0.5*g*t² + vz*t + z = 0
    #   -4.905*t² + 3*t + 1.5 = 0
    #   discriminant = 9 + 4*4.905*1.5 = 38.43
    #   t = (3 + sqrt(38.43)) / 9.81 = (3+6.20)/9.81 = 0.938s
    #   x_land = 4 + (-2)*0.938 = 2.124m
    #   注: 因为 g=-9.81, sqrt_d = sqrt(vz² - 2*g*z) = sqrt(9 - 2*(-9.81)*1.5) = sqrt(38.43) = 6.20
    #        t1 = (-vz + sqrt_d)/g = (-3+6.20)/(-9.81) = -0.326 (负, 舍去)
    #        t2 = (-vz - sqrt_d)/g = (-3-6.20)/(-9.81) = 0.938 (正根)

    for i in range(10):
        t = i * 0.05
        x = 4.0 - 2.0 * t
        z = 1.5 + 3.0 * t + 0.5 * (-9.81) * t * t
        det = Detection3D(pos=(x, 0.0, z), confidence=0.8, timestamp=t, radius_px=20.0)
        kf.update(det)

    landing = solver.solve(kf)
    check("landing predicted", landing is not None)
    if landing:
        lx, ly, lz = landing.pos
        t_arr = landing.t_arrival
        check("landing X ~ 2.1m", abs(lx - 2.1) < 0.6, f"got {lx:.3f}")
        check("landing Z = 0 (地面)", abs(lz) < 0.01, f"got {lz:.3f}")
        check("t_arrival > 0", t_arr > 0)
        check("confidence > 0", landing.confidence > 0)
        print(f"      落点: ({lx:.3f}, {ly:.3f}, {lz:.3f}) m, t={t_arr:.3f}s, conf={landing.confidence:.3f}")


# ── Test 6: 边界情况 ───────────────────────────────────────────────────

def test_edge_cases():
    print("\n-- Test 6: 边界情况 --")
    cfg = _make_traj_cfg()

    # 重置
    kf = TrajectoryFilter(cfg)
    for i in range(5):
        det = Detection3D(pos=(1.0, 0.0, 1.0), confidence=0.8, timestamp=i*0.05, radius_px=20.0)
        kf.update(det)
    kf.reset()
    check("reset -> not ready", not kf.ready)
    check("reset -> position zeros", np.allclose(kf.position, (0, 0, 0)))

    # 标定文件不存在
    from tennis_tracker.prediction import load_calibration
    check("missing calibration -> None",
          load_calibration("nonexistent.npz") is None)

    # 间隔过长 → 自动重置
    kf2 = TrajectoryFilter(cfg)
    det1 = Detection3D(pos=(1.0, 0.0, 1.0), confidence=0.8, timestamp=0.0, radius_px=20.0)
    kf2.update(det1)
    det2 = Detection3D(pos=(2.0, 0.0, 1.5), confidence=0.8, timestamp=1.0, radius_px=25.0)  # dt=1s > 0.5s
    kf2.update(det2)
    px, py, pz = kf2.position
    # 应该被重新初始化到 det2 的位置附近
    check("long gap -> re-init at det2 pos", abs(px - 2.0) < 0.1, f"got px={px:.3f}")


# ── Test 7: 加载实际标定文件 ───────────────────────────────────────────

def test_load_actual_calibration():
    print("\n-- Test 7: 加载实际 calibration.npz --")
    from tennis_tracker.prediction import load_calibration

    calib_path = PROJECT_ROOT / "configs" / "calibration.npz"
    calib = load_calibration(calib_path)

    if calib is None:
        print("  [SKIP] calibration.npz not found or invalid")
        return

    check("calibration loaded", calib is not None)
    check("fx > 0", calib.fx > 0, f"fx={calib.fx:.1f}")
    check("fy > 0", calib.fy > 0, f"fy={calib.fy:.1f}")
    check("cx > 0", calib.cx > 0, f"cx={calib.cx:.1f}")
    check("cy > 0", calib.cy > 0, f"cy={calib.cy:.1f}")
    check("dist_coeffs not None", calib.dist_coeffs is not None)
    print(f"      内参: fx={calib.fx:.1f} fy={calib.fy:.1f} cx={calib.cx:.1f} cy={calib.cy:.1f}")
    print(f"      畸变: {calib.dist_coeffs.ravel()}")

    # 用实际内参跑一次端到端
    pose = _make_pose()
    pos = detect_to_robot_3d(320.0, 263.0, 15.0, calib, pose, real_diameter_m=REAL_DIAMETER)
    check("real calib: detect_to_robot_3d works", pos is not None)
    if pos:
        x, y, z = pos
        check("real calib: Xr > 0", x > 0, f"x={x:.3f}")
        print(f"      使用实际标定, 地面球坐标: ({x:.2f}, {y:.2f}, {z:.2f}) m")


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Tennis Ball 3D Prediction — Unit Tests")
    print(f"Config: pitch={CAMERA_PITCH}deg, H={CAMERA_HEIGHT}m, D={REAL_DIAMETER}m")
    print("=" * 60)

    test_depth_estimation()
    test_pixel_to_camera()
    test_camera_to_robot()
    test_trajectory_filter()
    test_ballistic_solver()
    test_edge_cases()
    test_load_actual_calibration()

    print(f"\n{'=' * 60}")
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed", end="")
    if FAIL > 0:
        print(f", {FAIL} FAILED")
        return False
    else:
        print(" -- all good!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
