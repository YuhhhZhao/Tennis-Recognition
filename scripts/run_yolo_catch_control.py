#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Optional
import numpy as np

import cv2


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_tracker.config import UartConfig, load_config
from tennis_tracker.control.uart_bridge import UartBridge
from tennis_tracker.detection import AlphaBetaFilter, YOLODetector, clamp_point
from tennis_tracker.prediction import (
    BallisticSolver,
    CameraPose,
    TrajectoryFilter,
    detect_to_robot_3d,
    load_calibration,
)
from tennis_tracker.state import Detection, Detection3D, LandingPoint, TrackState

# ── IMU + odometry localization (optional) ──
try:
    from tennis_robot_sim.estimation.odometry import OdomTracker
    from tennis_robot_sim.imu import ComplementaryLocalizer, WitMotionIMU
    _LOC_AVAILABLE = True
except ImportError:
    _LOC_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO-only catch integration: detect a 3D speed jump, predict landing, "
            "and optionally send TARGET to ESP32."
        )
    )
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="weights/best_v3.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--http-stream", action="store_true",
                        help="Serve MJPEG HTTP stream at --http-stream-port (no X11 needed).")
    parser.add_argument("--http-stream-port", type=int, default=8080,
                        help="Port for MJPEG HTTP stream.")
    parser.add_argument("--chase", action="store_true", help="Chase mode: go directly toward ball at max speed, skip landing prediction.")
    parser.add_argument("--flip-y", action="store_true", help="Negate Y in TARGET commands (if car moves opposite direction).")
    parser.add_argument("--enable-control", action="store_true", help="Actually send TARGET to ESP32.")
    parser.add_argument("--uart-port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--uart-handshake-timeout", type=float, default=0.5, help="Seconds to wait for ESP32 RDY after opening UART.")
    parser.add_argument("--trigger-speed", type=float, default=0.4, help="Current 3D speed threshold, m/s.")
    parser.add_argument("--trigger-delta", type=float, default=0.15, help="Speed jump threshold, m/s.")
    parser.add_argument("--trigger-window", type=int, default=12, help="Recent 3D samples used for speed-jump trigger.")
    parser.add_argument("--min-samples", type=int, default=4, help="Trajectory samples before landing prediction.")
    parser.add_argument("--seed-samples", type=int, default=4, help="Samples to seed trajectory after trigger.")
    parser.add_argument("--measurement-noise", type=float, default=0.15)
    parser.add_argument("--process-noise-vel", type=float, default=0.8)
    parser.add_argument("--target-height", type=float, default=None)
    parser.add_argument("--send-interval", type=float, default=0.12)
    parser.add_argument("--max-target-distance", type=float, default=10.0)
    parser.add_argument("--min-arrival-time", type=float, default=0.05)
    parser.add_argument("--active-max-missing", type=int, default=30, help="Active-mode frames without usable 3D before re-arm (dead-reckoning enabled).")
    parser.add_argument("--rearm-grace", type=float, default=0.25, help="Seconds after predicted landing arrival before STOP and re-arm.")
    parser.add_argument("--rearm-cooldown", type=float, default=0.8, help="Seconds to ignore new triggers after STOP/re-arm.")
    parser.add_argument("--print-interval", type=float, default=0.10)
    parser.add_argument("--trail", type=int, default=60)
    parser.add_argument("--raw-3d", action="store_true", help="Use raw YOLO box for 3D instead of filtered 2D.")

    # ── Phase 1: 3D outlier filtering ──
    parser.add_argument("--min-z", type=float, default=-9.0,
                        help="Minimum valid Z in robot frame (m).")
    parser.add_argument("--max-x", type=float, default=9.0,
                        help="Maximum valid abs(X) in robot frame (m).")
    parser.add_argument("--max-y", type=float, default=9.0,
                        help="Maximum valid abs(Y) in robot frame (m).")
    parser.add_argument("--max-3d-speed", type=float, default=99.0,
                        help="Maximum physically plausible inter-frame 3D speed (m/s).")
    parser.add_argument("--max-frame-speed", type=float, default=50.0,
                        help="Reject single-frame position jumps faster than this (m/s, catches false positives).")
    parser.add_argument("--radius-median-window", type=int, default=5,
                        help="Window size for radius median filter.")

    # ── Phase 2: trigger confirmation ──
    parser.add_argument("--trigger-confirm-frames", type=int, default=2,
                        help="Consecutive frames meeting trigger condition required to activate.")
    parser.add_argument("--trigger-calm-speed", type=float, default=0.45,
                        help="Baseline speed must be below this to allow trigger (m/s, prevents hand-wave false triggers).")
    parser.add_argument("--trigger-fast-speed", type=float, default=1.0,
                        help="If current speed exceeds this, trigger immediately without calm check (m/s, for balls already in flight).")
    parser.add_argument("--speed-ema-alpha", type=float, default=0.3,
                        help="EMA alpha for speed smoothing in trigger detection.")

    # ── Phase 4: reachability ──
    parser.add_argument("--reachable-max-speed", type=float, default=1.8,
                        help="Maximum robot speed (m/s) for reachability check.")

    # ── Phase 5: latency ──
    parser.add_argument("--yolo-every-n", type=int, default=1,
                        help="Run YOLO detection every N frames (1=every frame).")

    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else str(Path(source))


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_device(value: str):
    if value == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return int(value) if value.isdigit() else value


def open_capture(source, cfg):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        fourcc = cfg.camera.fourcc.strip().upper()
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    else:
        cap = cv2.VideoCapture(source)

    if cfg.camera.buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.camera.buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        "[CAM] "
        f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f}fps fourcc={fourcc_text}",
        flush=True,
    )
    return cap


def tracked_detection_from_state(detection: Detection, state: TrackState) -> Detection:
    return Detection(
        center=state.center if state.center is not None else detection.center,
        bbox=detection.bbox,
        confidence=detection.confidence,
        source="YOLO_TRACK",
        timestamp=detection.timestamp,
        radius=state.radius if state.radius > 0 else detection.radius,
    )


def detection_to_3d(detection: Detection, cfg, calib, camera_pose: CameraPose,
                    dist_coeffs=None, K_mat=None):
    if calib is None:
        return None
    u, v = detection.center
    # Apply lens undistortion to pixel coordinates
    if dist_coeffs is not None and K_mat is not None:
        pts = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float32),
            K_mat, dist_coeffs, P=K_mat,
        )
        u, v = pts[0, 0]
    return detect_to_robot_3d(
        u, v,
        detection.radius,
        calib,
        camera_pose,
        real_diameter_m=cfg.geometry.ball_diameter_m,
    )


def speed_stats(
    samples: Deque[Detection3D],
    ema_alpha: float = 0.3,
) -> tuple[float, float, float]:
    """Compute speed statistics with EMA smoothing for noise reduction.

    Returns (baseline_speed, current_speed, speed_delta).
    baseline_speed: average of first half of EMA-smoothed speeds
    current_speed: average of second half of EMA-smoothed speeds
    speed_delta: current_speed - baseline_speed
    """
    if len(samples) < 4:
        return 0.0, 0.0, 0.0

    # Compute raw inter-sample speeds
    speeds: list = []
    items = list(samples)
    for prev, cur in zip(items, items[1:]):
        dt = cur.timestamp - prev.timestamp
        if dt <= 1e-6:
            continue
        dx = cur.pos[0] - prev.pos[0]
        dy = cur.pos[1] - prev.pos[1]
        dz = cur.pos[2] - prev.pos[2]
        speeds.append(math.sqrt(dx * dx + dy * dy + dz * dz) / dt)

    if len(speeds) < 3:
        return 0.0, 0.0, 0.0

    # EMA smooth the speeds to suppress single-frame noise spikes
    smoothed = [speeds[0]]
    for s in speeds[1:]:
        smoothed.append(ema_alpha * s + (1.0 - ema_alpha) * smoothed[-1])

    split = max(1, len(smoothed) // 2)
    baseline = sum(smoothed[:split]) / len(smoothed[:split])
    current = sum(smoothed[split:]) / len(smoothed[split:])

    return baseline, current, current - baseline


def update_trajectory(
    det_3d: Detection3D,
    traj_filter: TrajectoryFilter,
    ballistic: BallisticSolver,
    state: TrackState,
    max_history: int,
) -> Optional[LandingPoint]:
    state.pos_3d = det_3d.pos
    state.last_radius_px = det_3d.radius_px
    state.history_3d.append(det_3d)
    if len(state.history_3d) > max_history:
        state.history_3d = state.history_3d[-max_history:]
    traj_filter.update(det_3d)
    vx, vy, vz = traj_filter.velocity
    state.vel_3d = (float(vx), float(vy), float(vz))
    return ballistic.solve(traj_filter)


def is_safe_landing(lp: LandingPoint, args: argparse.Namespace) -> bool:
    x, y, _ = lp.pos
    dist = math.hypot(x, y)
    return dist <= args.max_target_distance and lp.t_arrival >= args.min_arrival_time


# ── Phase 1: 3D outlier filtering ──────────────────────────────────────

def is_valid_3d_position(
    pos: tuple,
    prev_pos: Optional[tuple],
    dt: float,
    args: argparse.Namespace,
) -> bool:
    """Reject physically impossible 3D positions.

    Checks:
      1. z >= min_z (ball cannot be significantly below ground)
      2. abs(x) <= max_x (ball within reasonable forward range)
      3. abs(y) <= max_y (ball within reasonable lateral range)
      4. Inter-frame speed < max_3d_speed (no teleportation)
    """
    x, y, z = pos
    if z < args.min_z:
        return False
    if abs(x) > args.max_x:
        return False
    if abs(y) > args.max_y:
        return False
    if prev_pos is not None and dt > 1e-6:
        px, py, pz = prev_pos
        dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        speed = dist / dt
        if speed > args.max_3d_speed:
            return False
    return True


def median_radius(new_radius: float, window: Deque[float]) -> float:
    """Maintain a sliding window of raw radii, return the median value."""
    window.append(new_radius)
    n = len(window)
    if n == 0:
        return new_radius
    sorted_vals = sorted(window)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])


# ── Phase 4: reachability check ────────────────────────────────────────

def is_reachable(lp: LandingPoint, max_robot_speed: float = 1.5) -> bool:
    """Check if landing point is reachable given robot max speed.

    Estimates required speed as Euclidean distance / t_arrival.
    Adds 15% margin for acceleration/deceleration.
    """
    x, y, _ = lp.pos
    dist = math.hypot(x, y)
    if dist <= 0.02:  # effectively at target, no movement needed
        return True
    required_speed = dist / max(0.05, lp.t_arrival)
    return required_speed <= max_robot_speed * 0.85


# ── Phase 5: latency helpers ───────────────────────────────────────────

def grab_latest_frame(cap) -> tuple:
    """Drain V4L2 buffer queue, return the most recent frame.

    Uses cap.read() in a tight loop to drain all buffered frames,
    then returns the last one. Compatible with Jetson V4L2 where
    cap.grab() may block unexpectedly.
    """
    ok, frame = cap.read()
    if not ok:
        return False, None
    # Drain any additional frames accumulated in the buffer
    for _ in range(4):
        ret, f = cap.read()
        if not ret:
            break
        ok, frame = ret, f
    return ok, frame


# ── HTTP MJPEG stream server ──────────────────────────────────────────

_shared_frame: Any = None
_shared_frame_lock = threading.Lock()


def _make_mjpeg_handler() -> type:
    """Create a handler class with access to the shared frame."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                with _shared_frame_lock:
                    if _shared_frame is None:
                        time.sleep(0.01)
                        continue
                    _, jpeg = cv2.imencode(".jpg", _shared_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.03)

        def log_message(self, *args) -> None:
            pass  # suppress HTTP request logs

    return _Handler


def start_mjpeg_server(port: int) -> HTTPServer:
    handler = _make_mjpeg_handler()
    server = HTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def draw_overlay(
    frame,
    detection: Optional[Detection],
    state: TrackState,
    landing: Optional[LandingPoint],
    trail: Deque[tuple[int, int]],
    active: bool,
    fps: float,
    infer_ms: float,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    if detection is not None:
        x1, y1, x2, y2 = detection.bbox
        cx, cy = clamp_point(detection.center, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), max(3, int(detection.radius)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if len(trail) >= 2:
        for p1, p2 in zip(trail, list(trail)[1:]):
            cv2.line(frame, p1, p2, (255, 0, 255), 2)

    status = "ACTIVE: sending landing" if active else "ARMED: waiting speed jump"
    lines = [
        status,
        f"conf={0.0 if detection is None else detection.confidence:.2f} "
        f"samples={len(state.history_3d)} miss={state.missing_frames}",
        f"FPS={fps:.1f} infer={infer_ms:.1f}ms",
    ]
    if state.pos_3d is not None:
        x, y, z = state.pos_3d
        vx, vy, vz = state.vel_3d
        lines.append(f"3D=({x:.2f},{y:.2f},{z:.2f}) v=({vx:.2f},{vy:.2f},{vz:.2f})")
    if landing is not None:
        lx, ly, lz = landing.pos
        lines.append(f"Land=({lx:.2f},{ly:.2f},{lz:.2f}) t={landing.t_arrival:.2f}s")

    for i, text in enumerate(lines[:5]):
        color = (0, 255, 0) if active and i == 0 else (0, 255, 255)
        cv2.putText(frame, text, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> None:
    args = parse_args()
    args.active_max_missing = max(0, args.active_max_missing)
    args.rearm_grace = max(0.0, args.rearm_grace)
    args.rearm_cooldown = max(0.0, args.rearm_cooldown)
    args.uart_handshake_timeout = max(0.0, args.uart_handshake_timeout)
    cfg = load_config(resolve_project_path(args.config))
    cfg.yolo.enabled = True
    cfg.yolo.model_path = str(resolve_project_path(args.model))
    cfg.yolo.confidence = args.conf
    cfg.yolo.imgsz = args.imgsz
    cfg.yolo.device = parse_device(args.device)
    cfg.display.enabled = not args.headless
    cfg.display.trail_length = args.trail
    cfg.trajectory.min_samples_for_fit = max(2, args.min_samples)
    cfg.trajectory.measurement_noise = max(1e-6, args.measurement_noise)
    cfg.trajectory.process_noise_vel = max(1e-6, args.process_noise_vel)
    if args.target_height is not None:
        cfg.trajectory.target_height_m = args.target_height

    print(
        "[MODE] YOLO-only catch control "
        f"control={'ENABLED' if args.enable_control else 'DRY-RUN'}",
        flush=True,
    )
    print(
        f"[TRIGGER] speed>={args.trigger_speed:.2f}m/s "
        f"delta>={args.trigger_delta:.2f}m/s window={args.trigger_window}",
        flush=True,
    )
    print(
        f"[PRED] min_samples={cfg.trajectory.min_samples_for_fit} "
        f"measurement_noise={cfg.trajectory.measurement_noise} "
        f"process_noise_vel={cfg.trajectory.process_noise_vel}",
        flush=True,
    )
    print(
        f"[SAFE] active_max_missing={args.active_max_missing} "
        f"rearm_grace={args.rearm_grace:.2f}s "
        f"rearm_cooldown={args.rearm_cooldown:.2f}s",
        flush=True,
    )

    uart: Optional[UartBridge] = None
    if args.enable_control:
        port = args.uart_port or cfg.uart.port
        baudrate = args.baudrate or cfg.uart.baudrate
        uart = UartBridge(
            UartConfig(
                port=port,
                baudrate=baudrate,
                timeout_s=cfg.uart.timeout_s,
                handshake_timeout_s=args.uart_handshake_timeout,
                enabled=True,
            )
        )
        if not uart.open():
            raise RuntimeError(f"Cannot open ESP32 UART: {port}")
        uart.send_stop()
        print(f"[UART] control armed on {port} @ {baudrate}", flush=True)
    else:
        print("[UART] dry-run: no TARGET will be sent. Add --enable-control to move car.", flush=True)

    # ── IMU + Odometry localization ──
    odom: Optional[Any] = None
    imu: Optional[Any] = None
    localizer: Optional[Any] = None
    _loc_last_t: float = 0.0
    _last_odom_query: float = 0.0
    _robot_x: float = 0.0
    _robot_y: float = 0.0
    _robot_yaw: float = 0.0
    if _LOC_AVAILABLE:
        from tennis_robot_sim.data import ControlCommand as SimCmd, RobotState as SimState
        odom = OdomTracker()
        localizer = ComplementaryLocalizer({
            "robot": {"start_pose": [0.0, 0.0, 0.0]},
            "imu": {"yaw_complementary_alpha": 0.92},
        })
        localizer.reset()
        try:
            imu = WitMotionIMU("/dev/ttyACM0", 115200)
            if imu.open():
                _loc_last_t = time.monotonic()
                print("[LOC] IMU + odometry localization enabled", flush=True)
            else:
                print("[LOC] IMU open failed, localization disabled", flush=True)
                imu = None
                localizer = None
        except Exception as e:
            print(f"[LOC] IMU init error: {e}", flush=True)
            imu = None
            localizer = None

    # ── HTTP MJPEG stream ──
    http_server: Optional[HTTPServer] = None
    if args.http_stream:
        http_server = start_mjpeg_server(args.http_stream_port)
        # Use HTTP stream instead of local imshow
        cfg.display.enabled = False
        print(
            f"[STREAM] MJPEG http://{os.uname().nodename}:{args.http_stream_port}/  "
            f"(or http://<jetson-ip>:{args.http_stream_port}/)",
            flush=True,
        )

    detector = YOLODetector(cfg.yolo)
    detector.load()
    cap = open_capture(parse_source(args.source), cfg)

    calib = load_calibration(resolve_project_path(cfg.geometry.calibration_path))
    if calib is None:
        raise RuntimeError(f"Calibration not loaded: {cfg.geometry.calibration_path}")
    # Extract raw matrices for undistortion
    _dist_coeffs = calib.dist_coeffs if calib.dist_coeffs is not None else None
    _K_mat = calib.K.copy() if calib.K is not None else None
    camera_pose = CameraPose(
        height_m=cfg.geometry.camera_height_m,
        pitch_deg=cfg.geometry.camera_pitch_deg,
        yaw_deg=cfg.geometry.camera_yaw_deg,
        offset_x_m=cfg.geometry.camera_offset_x_m,
        offset_y_m=cfg.geometry.camera_offset_y_m,
        offset_z_m=cfg.geometry.camera_offset_z_m,
    )

    filter_2d = AlphaBetaFilter(cfg.filter.alpha, cfg.filter.beta)
    traj_filter = TrajectoryFilter(cfg.trajectory)
    ballistic = BallisticSolver(cfg.trajectory)
    state = TrackState()
    trigger_samples: Deque[Detection3D] = deque(maxlen=max(4, args.trigger_window))
    trail: Deque[tuple[int, int]] = deque(maxlen=max(0, args.trail))

    active = False
    landing: Optional[LandingPoint] = None
    landing_count = 0
    last_send_t = 0.0
    last_print_t = 0.0
    last_frame_t = 0.0
    missing_after_active = 0
    landing_deadline_t: Optional[float] = None
    rearm_until_t = 0.0
    fps = 0.0

    # Phase 1: outlier filtering state
    radius_window: Deque[float] = deque(maxlen=max(3, args.radius_median_window))
    last_valid_3d_pos: Optional[tuple] = None
    last_valid_3d_t: float = 0.0

    # Phase 2: trigger confirmation
    trigger_confirm_count: int = 0

    # Bounce tracking
    bounce_count: int = 0
    _prev_vz: float = 0.0
    _bounce_cooldown: int = 0  # frames to suppress re-triggering same bounce

    # Phase 5: YOLO frame skip
    yolo_frame_counter: int = 0

    # Track consecutive "landing too soon" to avoid premature rearm
    landing_too_soon_streak: int = 0

    # Debug: track rejected 3D positions to avoid log spam
    _reject_count: int = 0
    _last_reject_print: float = 0.0

    # Stuck detection: if landing point barely moves for too long, it's a false positive
    _last_landing_pos: Optional[tuple] = None
    _stuck_frames: int = 0

    _chase_target: Optional[tuple] = None

    # 3D position EMA smoothing (display only, trigger still uses raw)
    smooth_pos_3d: Optional[tuple] = None
    _pos_ema_alpha: float = 0.4  # lower = smoother, higher = more responsive

    def rearm(reason: str) -> None:
        nonlocal active
        nonlocal landing
        nonlocal landing_count
        nonlocal last_send_t
        nonlocal last_print_t
        nonlocal missing_after_active
        nonlocal landing_deadline_t
        nonlocal rearm_until_t
        nonlocal trigger_confirm_count
        nonlocal yolo_frame_counter
        nonlocal landing_too_soon_streak

        stopped = False
        if uart is not None:
            stopped = uart.send_stop()

        active = False
        landing = None
        landing_count = 0
        last_send_t = 0.0
        last_print_t = 0.0
        missing_after_active = 0
        landing_deadline_t = None
        rearm_until_t = time.monotonic() + args.rearm_cooldown
        trigger_confirm_count = 0
        yolo_frame_counter = 0
        landing_too_soon_streak = 0
        nonlocal _reject_count
        nonlocal _last_reject_print
        _reject_count = 0
        _last_reject_print = 0.0
        nonlocal _last_landing_pos
        nonlocal _stuck_frames
        _last_landing_pos = None
        _stuck_frames = 0
        nonlocal bounce_count
        nonlocal _prev_vz
        nonlocal _bounce_cooldown
        bounce_count = 0
        _prev_vz = 0.0
        _bounce_cooldown = 0
        nonlocal _chase_target
        _chase_target = None
        traj_filter.reset()
        state.reset()
        trigger_samples.clear()
        trail.clear()
        radius_window.clear()
        nonlocal last_valid_3d_pos
        nonlocal last_valid_3d_t
        last_valid_3d_pos = None
        last_valid_3d_t = 0.0
        nonlocal smooth_pos_3d
        smooth_pos_3d = None

        stop_status = "STOP sent" if stopped else "STOP dry-run" if uart is None else "STOP failed"
        print(f"\n[SAFE] {reason}; {stop_status}; re-armed", flush=True)

    try:
        while True:
            # ── Phase 5a: grab latest frame (drain stale buffers) ──
            ok, frame = grab_latest_frame(cap)
            if not ok:
                break
            now_t = time.monotonic()
            if last_frame_t > 0:
                instant = 1.0 / max(1e-6, now_t - last_frame_t)
                fps = instant if fps <= 0 else 0.85 * fps + 0.15 * instant
            last_frame_t = now_t

            # ── Localization update (IMU + odometry) ──
            _robot_dx, _robot_dy = 0.0, 0.0
            if localizer is not None and imu is not None:
                dt_loc = now_t - _loc_last_t
                _loc_last_t = now_t
                if dt_loc > 0 and dt_loc < 0.5:
                    # 1. IMU: gyro → yaw rate
                    sample = imu.get_sample()
                    if sample is not None:
                        localizer.predict(SimCmd(v=0, omega=sample.yaw_rate_radps), dt_loc)
                        localizer.update_imu(sample, dt_loc)
                # 2. Odometry: ping ESP32 every ~100ms
                if uart is not None and now_t - _last_odom_query > 0.1:
                    steps = uart.send_ping()
                    _last_odom_query = now_t
                    if steps is not None and odom is not None:
                        odom.update(steps.to_tuple())
                        ox, oy, oyaw = odom.pose
                        localizer.update_odometry(
                            SimState(x=ox, y=oy, yaw=oyaw, v=0, omega=0, timestamp=now_t),
                            weight=0.15,
                        )
                # 3. Track robot displacement since last frame
                loc_state = localizer.get_state()
                prev_x, prev_y = _robot_x, _robot_y
                _robot_x, _robot_y, _robot_yaw = loc_state.x, loc_state.y, loc_state.yaw
                _robot_dx = _robot_x - prev_x
                _robot_dy = _robot_y - prev_y

            got_usable_3d = False
            in_rearm_cooldown = now_t < rearm_until_t
            infer_ms = 0.0  # default when YOLO is skipped

            # ── Phase 5b: optional YOLO frame skip ──
            yolo_frame_counter += 1
            run_yolo = yolo_frame_counter >= args.yolo_every_n

            detection = None
            if run_yolo:
                t0 = time.monotonic()
                detection = detector.detect(frame)
                infer_ms = (time.monotonic() - t0) * 1000.0
                yolo_frame_counter = 0
            # else: frame skipped, detection stays None, infer_ms=0.0

            if in_rearm_cooldown:
                if now_t - last_print_t >= args.print_interval:
                    print(
                        f"[COOLDOWN] ignoring trigger for {rearm_until_t - now_t:.2f}s",
                        flush=True,
                    )
                    last_print_t = now_t
            elif detection is not None:
                state = filter_2d.update(state, detection)
                trail.append(clamp_point(state.center, frame.shape[1], frame.shape[0]))
                det_for_3d = detection if args.raw_3d else tracked_detection_from_state(detection, state)

                # ── Phase 1: median radius filter + 3D outlier check ──
                median_r = median_radius(det_for_3d.radius, radius_window)
                # Undistort pixel coords before 3D conversion
                u_raw, v_raw = det_for_3d.center
                if _dist_coeffs is not None and _K_mat is not None:
                    pts = cv2.undistortPoints(
                        np.array([[[u_raw, v_raw]]], dtype=np.float32),
                        _K_mat, _dist_coeffs, P=_K_mat,
                    )
                    u_undist, v_undist = pts[0, 0]
                else:
                    u_undist, v_undist = u_raw, v_raw
                pos_raw = detect_to_robot_3d(
                    u_undist, v_undist, det_for_3d.radius, calib, camera_pose,
                    real_diameter_m=cfg.geometry.ball_diameter_m,
                )
                if pos_raw is not None and median_r > 0:
                    # Recompute 3D with median-smoothed radius for stability
                    pos = detect_to_robot_3d(
                        u_undist, v_undist, median_r, calib, camera_pose,
                        real_diameter_m=cfg.geometry.ball_diameter_m,
                    )
                    # Fall back to raw position if median-recomputed fails
                    if pos is None:
                        pos = pos_raw
                else:
                    pos = pos_raw

                if pos is not None:
                    # Phase 1: outlier rejection + teleport filter
                    dt_3d = (det_for_3d.timestamp - last_valid_3d_t) if last_valid_3d_pos is not None else 0.0
                    # Check for massive single-frame jump (false positive teleport)
                    frame_jump_ok = True
                    if last_valid_3d_pos is not None and dt_3d > 1e-6:
                        px, py, pz = last_valid_3d_pos
                        dist = math.sqrt((pos[0]-px)**2 + (pos[1]-py)**2 + (pos[2]-pz)**2)
                        frame_jump_ok = (dist / dt_3d) <= args.max_frame_speed
                    if is_valid_3d_position(pos, last_valid_3d_pos, dt_3d, args) and frame_jump_ok:
                        got_usable_3d = True
                        last_valid_3d_pos = pos
                        last_valid_3d_t = det_for_3d.timestamp
                        _reject_count = 0

                        # Compensate for robot motion since last frame
                        pos_comp = (pos[0] + _robot_dx, pos[1] + _robot_dy, pos[2])

                        # EMA smooth for display (use compensated pos)
                        if smooth_pos_3d is None:
                            smooth_pos_3d = pos_comp
                        else:
                            sx, sy, sz = smooth_pos_3d
                            smooth_pos_3d = (
                                _pos_ema_alpha * pos_comp[0] + (1.0 - _pos_ema_alpha) * sx,
                                _pos_ema_alpha * pos_comp[1] + (1.0 - _pos_ema_alpha) * sy,
                                _pos_ema_alpha * pos_comp[2] + (1.0 - _pos_ema_alpha) * sz,
                            )
                        det_3d = Detection3D(
                            pos=pos_comp,
                            confidence=det_for_3d.confidence,
                            timestamp=det_for_3d.timestamp,
                            radius_px=median_r,
                        )
                        trigger_samples.append(det_3d)

                        # ── Phase 2a: EMA-smoothed speed stats ──
                        prev_speed, current_speed, speed_delta = speed_stats(
                            trigger_samples, ema_alpha=args.speed_ema_alpha,
                        )

                        if not active:
                            # ── Phase 2b: trigger with confirmation counter ──
                            # Two trigger paths:
                            # 1. Normal: ball was calm, then accelerated (calm baseline + speed jump)
                            # 2. Fast: ball already in flight at high speed (skip calm check)
                            normal_trigger = (
                                len(trigger_samples) >= max(4, args.trigger_window // 2)
                                and prev_speed <= args.trigger_calm_speed
                                and current_speed >= args.trigger_speed
                                and speed_delta >= args.trigger_delta
                            )
                            fast_trigger = (
                                len(trigger_samples) >= 3  # fewer samples needed for fast balls
                                and current_speed >= args.trigger_fast_speed
                            )
                            condition_met = normal_trigger or fast_trigger
                            if condition_met:
                                trigger_confirm_count += 1
                            else:
                                trigger_confirm_count = max(0, trigger_confirm_count - 1)

                            if trigger_confirm_count >= args.trigger_confirm_frames:
                                active = True
                                missing_after_active = 0
                                landing_deadline_t = None
                                landing_count = 0
                                last_send_t = 0.0
                                bounce_count = 0
                                _prev_vz = 0.0
                                _bounce_cooldown = 0
                                _chase_target = None
                                seed = list(trigger_samples)[-max(2, args.seed_samples):]
                                trigger_rejected = False
                                if not args.chase:
                                    traj_filter.reset()
                                    state.history_3d.clear()
                                    for sample in seed:
                                        landing = update_trajectory(
                                            sample, traj_filter, ballistic, state, cfg.trajectory.max_history
                                        )
                                    if len(seed) >= 2:
                                        dt_seed = seed[1].timestamp - seed[0].timestamp
                                        if dt_seed > 1e-6:
                                            traj_filter._x[3, 0] = (seed[1].pos[0] - seed[0].pos[0]) / dt_seed
                                            traj_filter._x[4, 0] = (seed[1].pos[1] - seed[0].pos[1]) / dt_seed
                                            traj_filter._x[5, 0] = (seed[1].pos[2] - seed[0].pos[2]) / dt_seed
                                    if landing is not None:
                                        landing_deadline_t = now_t + landing.t_arrival
                                        if landing.t_arrival < args.min_arrival_time:
                                            trigger_rejected = True
                                            rearm(f"landing too soon t={landing.t_arrival:.2f}s")
                                if not trigger_rejected:
                                    trig_mode = "fast" if current_speed >= args.trigger_fast_speed else "normal"
                                    print(
                                        f"\n[TRIGGER] speed jump confirmed ({trig_mode}) "
                                        f"prev={prev_speed:.2f} current={current_speed:.2f} "
                                        f"delta={speed_delta:.2f}m/s seed={len(seed)} "
                                        f"confirm={trigger_confirm_count}",
                                        flush=True,
                                    )
                        else:
                            if args.chase:
                                # ── Chase mode: go directly toward ball, no prediction ──
                                landing = None  # no ballistic prediction
                                landing_is_late = False
                                _chase_target = pos_comp  # use compensated ball position
                            else:
                                # ── Normal mode: Kalman + ballistic prediction ──
                                landing = update_trajectory(
                                    det_3d, traj_filter, ballistic, state, cfg.trajectory.max_history
                                )
                                _chase_target = None
                            if landing is not None:
                                estimated_arrival_t = now_t + landing.t_arrival
                                landing_deadline_t = estimated_arrival_t
                                if landing.t_arrival < args.min_arrival_time:
                                    # Don't rearm immediately — Kalman may still be converging.
                                    landing_too_soon_streak += 1
                                else:
                                    landing_too_soon_streak = 0

                                # ── Bounce detection: two methods ──
                                _, _, kf_z = traj_filter.position
                                _, _, kf_vz = traj_filter.velocity
                                _bounce_cooldown = max(0, _bounce_cooldown - 1)
                                # Method 1: vz reversal near ground (ball falling → rising)
                                near_ground = kf_z < 0.35
                                was_falling = _prev_vz < -0.3
                                now_rising_or_slow = kf_vz > -0.3
                                bounce1 = (_bounce_cooldown <= 0 and near_ground
                                           and was_falling and now_rising_or_slow)
                                # Method 2: z crossed below ground threshold with downward velocity
                                prev_z = traj_filter._x[2, 0]
                                z_crossed_ground = (kf_z < 0.08 and _prev_vz < -0.2)
                                bounce2 = (_bounce_cooldown <= 0 and z_crossed_ground
                                           and bounce_count == 0)  # first bounce only for method 2
                                # Debug: print bounce status every second
                                if near_ground and now_t - _last_reject_print > 0.5:
                                    print(
                                        f"[BOUNCE?] z={kf_z:.3f} vz={kf_vz:.2f}(prev={_prev_vz:.2f}) "
                                        f"fall={was_falling} slow={now_rising_or_slow} "
                                        f"b1={bounce1} b2={bounce2} cd={_bounce_cooldown}",
                                        flush=True,
                                    )
                                    _last_reject_print = now_t
                                if bounce1 or bounce2:
                                    bounce_count += 1
                                    _bounce_cooldown = 5
                                    # Dampen all velocity components (ball loses energy on bounce)
                                    bounce_vz = max(abs(_prev_vz) * 0.55, 0.5)
                                    traj_filter._x[5, 0] = bounce_vz
                                    traj_filter._x[3, 0] *= 0.7  # horizontal drag
                                    traj_filter._x[4, 0] *= 0.7
                                    # Don't force z to 0.03 — use actual Kalman position
                                    landing = ballistic.solve(traj_filter)
                                    if landing is not None:
                                        landing_deadline_t = now_t + landing.t_arrival
                                    print(
                                        f"\n[BOUNCE#{bounce_count}] "
                                        f"z={kf_z:.3f} vz={_prev_vz:.2f}→{traj_filter._x[5,0]:.2f} "
                                        + (f"land=({landing.pos[0]:.2f},{landing.pos[1]:.2f}) t={landing.t_arrival:.2f}s" if landing else ""),
                                        flush=True,
                                    )
                            _prev_vz = traj_filter.velocity[2]

                        if not active and now_t - last_print_t >= args.print_interval:
                            sx, sy, sz = smooth_pos_3d if smooth_pos_3d is not None else pos
                            calm_ok = prev_speed <= args.trigger_calm_speed
                            fast_ok = current_speed >= args.trigger_fast_speed
                            trig_type = "fast" if fast_ok else ("normal" if calm_ok else "none")
                            loc_str = f" loc=({_robot_x:+.2f},{_robot_y:+.2f})" if localizer is not None else ""
                            print(
                                f"[WAIT] pos=({sx:+.2f},{sy:+.2f},{sz:+.2f}) "
                                f"speed={current_speed:.2f} base={prev_speed:.2f} "
                                f"trig={trig_type} delta={speed_delta:.2f} confirm={trigger_confirm_count}/{args.trigger_confirm_frames}{loc_str}",
                                flush=True,
                            )
                            last_print_t = now_t
                    else:
                        # Outlier or teleport rejected — print reason periodically
                        _reject_count += 1
                        if _reject_count <= 3 or now_t - _last_reject_print > 1.0:
                            x, y, z = pos
                            reasons = []
                            if not frame_jump_ok:
                                reasons.append(f"teleport {dist/dt_3d:.0f}m/s>{args.max_frame_speed}")
                            if z < args.min_z:
                                reasons.append(f"z={z:.3f}<{args.min_z}")
                            if abs(x) > args.max_x:
                                reasons.append(f"|x|={abs(x):.2f}>{args.max_x}")
                            if abs(y) > args.max_y:
                                reasons.append(f"|y|={abs(y):.2f}>{args.max_y}")
                            if last_valid_3d_pos is not None and dt_3d > 1e-6 and not frame_jump_ok:
                                pass  # already reported as teleport
                            elif last_valid_3d_pos is not None and dt_3d > 1e-6:
                                px, py, pz = last_valid_3d_pos
                                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
                                spd = dist / dt_3d
                                if spd > args.max_3d_speed:
                                    reasons.append(f"speed={spd:.1f}>{args.max_3d_speed}")
                            if not reasons:
                                reasons.append("unknown")
                            print(
                                f"[DROP] rejected 3D: pos=({x:+.2f},{y:+.2f},{z:+.2f}) "
                                f"reason={'; '.join(reasons)}",
                                flush=True,
                            )
                            _last_reject_print = now_t
            else:
                if detection is None:
                    state.mark_missing()

            if active:
                if got_usable_3d:
                    missing_after_active = 0
                else:
                    missing_after_active += 1
                    # ── Dead-reckon: propagate Kalman without measurement ──
                    if not args.chase and traj_filter._initialized:
                        dt_since_last = now_t - traj_filter._last_t
                        if 0 < dt_since_last <= 0.5:
                            traj_filter.propagate(dt_since_last)
                            # ── Bounce check in dead-reckoning ──
                            kf_z = traj_filter._x[2, 0]
                            kf_vz = traj_filter._x[5, 0]
                            _bounce_cooldown = max(0, _bounce_cooldown - 1)
                            if kf_z < 0.25 and now_t - _last_reject_print > 0.5:
                                print(
                                    f"[BOUNCE?] dead-reckon z={kf_z:.3f} vz={kf_vz:.2f} "
                                    f"prev_vz={_prev_vz:.2f} cd={_bounce_cooldown}",
                                    flush=True,
                                )
                                _last_reject_print = now_t
                            if (_bounce_cooldown <= 0 and kf_z < 0.10
                                    and _prev_vz < -0.3 and kf_vz < -0.3):
                                bounce_count += 1
                                _bounce_cooldown = 5
                                traj_filter._x[5, 0] = max(abs(_prev_vz) * 0.55, 0.5)
                                traj_filter._x[3, 0] *= 0.7
                                traj_filter._x[4, 0] *= 0.7
                                print(
                                    f"\n[BOUNCE#{bounce_count}] propagate "
                                    f"z={kf_z:.3f} vz={_prev_vz:.2f}→{traj_filter._x[5,0]:.2f}",
                                    flush=True,
                                )
                            _prev_vz = traj_filter._x[5, 0]
                            landing = ballistic.solve(traj_filter)
                            if landing is not None:
                                landing_deadline_t = now_t + landing.t_arrival
                    # Only rearm if ball lost for long AND past landing deadline
                    if missing_after_active > args.active_max_missing:
                        if landing_deadline_t is not None and now_t < landing_deadline_t:
                            pass  # still waiting for landing, keep dead-reckoning
                        else:
                            rearm(f"lost ball for {missing_after_active} active frames")

                # ── Persistent "landing too soon" check ──
                if landing_too_soon_streak > 5:
                    rearm(f"persistent landing too soon ({landing_too_soon_streak} frames)")

            chase_ready = args.chase and _chase_target is not None
            if active and (landing is not None or chase_ready) and now_t - last_print_t >= args.print_interval:
                landing_count += 1
                if chase_ready:
                    lx, ly, lz = _chase_target
                else:
                    lx, ly, lz = landing.pos

                # ── Stuck / timeout detection ──
                # 1. Landing point barely moving → false positive
                if _last_landing_pos is not None:
                    plx, ply, _ = _last_landing_pos
                    if math.hypot(lx - plx, ly - ply) < 0.10:
                        _stuck_frames += 1
                    else:
                        _stuck_frames = max(0, _stuck_frames - 2)
                _last_landing_pos = (lx, ly, lz)
                if _stuck_frames > 30:  # ~1s
                    rearm(f"stuck landing (false positive) for {_stuck_frames} frames")
                    continue
                # 2. ACTIVE for too long without ball moving → timeout
                if landing_count > 150:  # ~5s, ball should have landed by now
                    rearm(f"active timeout ({landing_count} landings)")
                    continue
                vx, vy, vz = traj_filter.velocity if not args.chase else (0.0, 0.0, 0.0)

                # ── Phase 4: safety / reachability ──
                if args.chase:
                    safe = True
                    should_send = True
                    send_t = 0.1  # dummy, will be overridden
                else:
                    landing_is_late = landing.t_arrival < args.min_arrival_time
                    safe = is_safe_landing(landing, args) and not landing_is_late
                    should_send = safe
                    send_t = landing.t_arrival

                best_x, best_y = lx, ly

                sent = False
                if should_send and uart is not None and now_t - last_send_t >= args.send_interval:
                    send_yy = -best_y if args.flip_y else best_y
                    abs_x, abs_y = abs(best_x), abs(send_yy)
                    if abs_x < 0.02 and abs_y < 0.02:
                        pass  # too close
                    else:
                        # Compute max-speed target, preserving direction ratio.
                        # Firmware: vx=clamp(x/t,1.8), vy=clamp(y/t,1.5). t=0.01 < MIN_T → uses signs.
                        ratio_val = abs_y / max(abs_x, 0.001)
                        if ratio_val <= 1.5 / 1.8:
                            vx = 1.8 * (1.0 if best_x > 0 else -1.0)
                            vy = vx * (send_yy / max(abs(best_x), 0.001))
                        else:
                            vy = 1.5 * (1.0 if send_yy > 0 else -1.0)
                            vx = vy * (best_x / max(abs(send_yy), 0.001))
                        target_x = vx * 0.01
                        target_y = vy * 0.01
                        sent = uart.send_target(target_x, target_y, 0.01)
                        best_x, best_y = target_x, target_y
                        send_t = 0.01
                    last_send_t = now_t
                    # ── Phase 5c: decouple UART read (every 4th send) ──
                    if landing_count % 4 == 0:
                        resp = uart.read_response()
                        if resp:
                            print(f"[UART] {resp}", flush=True)

                if args.chase:
                    status = "chase"
                elif landing_is_late:
                    status = f"late(streak={landing_too_soon_streak})"
                elif safe:
                    status = "safe"
                else:
                    status = "unsafe"
                target_str = ""
                target_y_display = -best_y if args.flip_y else best_y
                if args.flip_y or (not args.chase and send_t != landing.t_arrival):
                    parts = [f"target=({best_x:+.2f},{target_y_display:+.2f})"]
                    if not args.chase and send_t != landing.t_arrival:
                        parts.append(f"t={send_t:.2f}s")
                    target_str = " ".join(parts)
                label = "CHASE" if args.chase else "LAND"
                vel_str = f"vel=({vx:+.2f},{vy:+.2f},{vz:+.2f}) " if not args.chase else ""
                t_str = f"t={landing.t_arrival:.2f}s conf={landing.confidence:.2f} " if not args.chase else ""
                print(
                    f"[{label}#{landing_count:04d}] "
                    f"pos=({lx:+.2f},{ly:+.2f},{lz:+.2f}) "
                    f"{t_str}{vel_str}"
                    f"{'SENT' if sent else 'DRY' if uart is None else 'HELD'} "
                    f"{status} {target_str}",
                    flush=True,
                )
                last_print_t = now_t

            # "landing window passed" rearm disabled — stay active until ball lost

            if cfg.display.enabled:
                draw_overlay(frame, detection, state, landing, trail, active, fps, infer_ms)
                cv2.imshow(cfg.display.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Share frame to HTTP MJPEG stream ──
            if http_server is not None:
                overlay_frame = frame.copy()
                draw_overlay(overlay_frame, detection, state, landing, trail, active, fps, infer_ms)
                with _shared_frame_lock:
                    global _shared_frame
                    _shared_frame = overlay_frame

    finally:
        cap.release()
        if uart is not None:
            for _ in range(3):
                uart.send_stop()
                time.sleep(0.03)
            uart.close()
        if imu is not None:
            imu.close()
        if http_server is not None:
            http_server.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
