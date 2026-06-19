#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_tracker.config import load_config
from tennis_tracker.detection import AlphaBetaFilter, YOLODetector, clamp_point
from tennis_tracker.prediction import (
    BallisticSolver,
    CameraPose,
    TrajectoryFilter,
    detect_to_robot_3d,
    load_calibration,
)
from tennis_tracker.state import Detection, Detection3D, LandingPoint, TrackState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe YOLO-only preview. No HSV, UART, IMU, or lower-controller output."
    )
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--model", default="weights/best_v3.pt", help="YOLO .pt/.engine path.")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence override.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO image size override.")
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO device: auto, cpu, 0, cuda:0, etc. Default: auto.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--trail", type=int, default=None, help="2D trail length.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum YOLO 3D samples before landing prediction. Default: 6.",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=0.08,
        help="3D Kalman measurement noise. Higher is smoother. Default: 0.08.",
    )
    parser.add_argument(
        "--process-noise-vel",
        type=float,
        default=1.2,
        help="3D velocity process noise. Higher follows fast throws sooner. Default: 1.2.",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=None,
        help="Landing/catch height in meters. Defaults to config trajectory target height.",
    )
    parser.add_argument(
        "--raw-3d",
        action="store_true",
        help="Use raw YOLO center/radius for 3D instead of filtered 2D track.",
    )
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


def update_3d(
    detection: Optional[Detection],
    cfg,
    calib,
    camera_pose: CameraPose,
    traj_filter: TrajectoryFilter,
    ballistic: BallisticSolver,
    state: TrackState,
) -> Optional[LandingPoint]:
    if detection is None or calib is None:
        return None

    u, v = detection.center
    radius_px = detection.radius
    state.last_radius_px = radius_px
    pos_3d = detect_to_robot_3d(
        u,
        v,
        radius_px,
        calib,
        camera_pose,
        real_diameter_m=cfg.geometry.ball_diameter_m,
    )
    if pos_3d is None:
        return None

    det_3d = Detection3D(
        pos=pos_3d,
        confidence=detection.confidence,
        timestamp=detection.timestamp,
        radius_px=radius_px,
    )
    state.pos_3d = pos_3d
    state.history_3d.append(det_3d)
    if len(state.history_3d) > cfg.trajectory.max_history:
        state.history_3d = state.history_3d[-cfg.trajectory.max_history :]

    traj_filter.update(det_3d)
    vx, vy, vz = traj_filter.velocity
    state.vel_3d = (float(vx), float(vy), float(vz))
    return ballistic.solve(traj_filter)


def tracked_detection_from_state(
    detection: Detection,
    state: TrackState,
) -> Detection:
    return Detection(
        center=state.center if state.center is not None else detection.center,
        bbox=detection.bbox,
        confidence=detection.confidence,
        source="YOLO_TRACK",
        timestamp=detection.timestamp,
        radius=state.radius if state.radius > 0 else detection.radius,
    )


def draw_overlay(
    frame,
    detection: Optional[Detection],
    state: TrackState,
    landing: Optional[LandingPoint],
    trail: Deque[tuple[int, int]],
    fps: float,
    infer_ms: float,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    if detection is not None:
        x1, y1, x2, y2 = detection.bbox
        cx, cy = clamp_point(detection.center, width, height)
        radius = max(3, int(round(detection.radius)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if len(trail) >= 2:
        points = list(trail)
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, (255, 0, 255), 2)

    lines = [
        f"YOLO-only conf={0.0 if detection is None else detection.confidence:.2f} "
        f"miss={state.missing_frames} samples={len(state.history_3d)}",
        f"FPS={fps:.1f} infer={infer_ms:.1f}ms",
    ]
    if state.pos_3d is not None:
        x, y, z = state.pos_3d
        lines.append(f"3D: ({x:.2f}, {y:.2f}, {z:.2f}) m r={state.last_radius_px:.1f}px")
    if landing is not None:
        lx, ly, lz = landing.pos
        lines.append(f"Land: ({lx:.2f}, {ly:.2f}, {lz:.2f})m t={landing.t_arrival:.2f}s")

    for i, text in enumerate(lines):
        color = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)][i % 4]
        cv2.putText(
            frame,
            text,
            (12, 28 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    cfg.control.enabled = False
    cfg.uart.enabled = False
    cfg.yolo.enabled = True
    cfg.yolo.model_path = str(resolve_project_path(args.model))
    cfg.yolo.device = parse_device(args.device)
    if args.conf is not None:
        cfg.yolo.confidence = args.conf
    if args.imgsz is not None:
        cfg.yolo.imgsz = args.imgsz
    if args.headless:
        cfg.display.enabled = False
    if args.trail is not None:
        cfg.display.trail_length = args.trail
    cfg.trajectory.min_samples_for_fit = max(2, args.min_samples)
    cfg.trajectory.measurement_noise = max(1e-6, args.measurement_noise)
    cfg.trajectory.process_noise_vel = max(1e-6, args.process_noise_vel)
    if args.target_height is not None:
        cfg.trajectory.target_height_m = args.target_height

    print("[SAFE] UART/control/IMU are not initialized in this script.", flush=True)
    print(
        f"[YOLO] model={cfg.yolo.model_path} conf={cfg.yolo.confidence} "
        f"imgsz={cfg.yolo.imgsz} device={cfg.yolo.device}",
        flush=True,
    )
    print(
        "[PRED] "
        f"min_samples={cfg.trajectory.min_samples_for_fit} "
        f"measurement_noise={cfg.trajectory.measurement_noise} "
        f"process_noise_vel={cfg.trajectory.process_noise_vel} "
        f"target_height={cfg.trajectory.target_height_m} "
        f"source={'raw-yolo' if args.raw_3d else 'filtered-yolo'}",
        flush=True,
    )

    detector = YOLODetector(cfg.yolo)
    detector.load()
    cap = open_capture(parse_source(args.source), cfg)

    calib = load_calibration(resolve_project_path(cfg.geometry.calibration_path))
    if calib is None:
        print("[3D] calibration not loaded; 3D trajectory prediction disabled.", flush=True)
    camera_pose = CameraPose(
        height_m=cfg.geometry.camera_height_m,
        pitch_deg=cfg.geometry.camera_pitch_deg,
        yaw_deg=cfg.geometry.camera_yaw_deg,
        offset_x_m=cfg.geometry.camera_offset_x_m,
        offset_y_m=cfg.geometry.camera_offset_y_m,
        offset_z_m=cfg.geometry.camera_offset_z_m,
    )
    traj_filter = TrajectoryFilter(cfg.trajectory)
    ballistic = BallisticSolver(cfg.trajectory)
    state = TrackState()
    filter_2d = AlphaBetaFilter(cfg.filter.alpha, cfg.filter.beta)
    trail: Deque[tuple[int, int]] = deque(maxlen=max(0, cfg.display.trail_length))

    last_frame_t = 0.0
    fps = 0.0
    landing: Optional[LandingPoint] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_t = time.monotonic()
            if last_frame_t > 0:
                instant = 1.0 / max(1e-6, frame_t - last_frame_t)
                fps = instant if fps <= 0 else 0.85 * fps + 0.15 * instant
            last_frame_t = frame_t

            t0 = time.monotonic()
            detection = detector.detect(frame)
            infer_ms = (time.monotonic() - t0) * 1000.0

            if detection is not None:
                state = filter_2d.update(state, detection)
                trail.append(clamp_point(state.center, frame.shape[1], frame.shape[0]))
                detection_for_3d = (
                    detection if args.raw_3d else tracked_detection_from_state(detection, state)
                )
                landing = update_3d(
                    detection_for_3d,
                    cfg,
                    calib,
                    camera_pose,
                    traj_filter,
                    ballistic,
                    state,
                )
            else:
                state.mark_missing()
                if state.missing_frames > cfg.filter.max_missing_frames:
                    state.reset()
                    traj_filter.reset()
                    trail.clear()
                    landing = None

            if cfg.display.enabled:
                draw_overlay(frame, detection, state, landing, trail, fps, infer_ms)
                cv2.imshow(cfg.display.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
