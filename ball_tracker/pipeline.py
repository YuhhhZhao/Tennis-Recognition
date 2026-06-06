from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .async_yolo import AsyncYOLOWorker
from .config import AppConfig
from .control import CarController
from .filters import AlphaBetaFilter, clamp_point
from .hsv_tracker import HSVTracker
from .prediction import (
    BallisticSolver,
    CameraIntrinsics,
    CameraPose,
    TrajectoryFilter,
    detect_to_robot_3d,
    load_calibration,
)
from .state import Detection, Detection3D, LandingPoint, TrackState, now
from .yolo_detector import YOLODetector


class TrackerPipeline:
    def __init__(self, cfg: AppConfig, source: str):
        self.cfg = cfg
        self.source = self._parse_source(source)
        self.state = TrackState()
        self.filter = AlphaBetaFilter(
            alpha=cfg.filter.alpha,
            beta=cfg.filter.beta,
        )
        self.hsv_tracker = HSVTracker(cfg.hsv, cfg.roi)
        self.yolo_worker: Optional[AsyncYOLOWorker] = None
        if cfg.yolo.enabled:
            self.yolo_worker = AsyncYOLOWorker(YOLODetector(cfg.yolo))
        self.last_yolo_request = 0.0
        self.controller = CarController(
            cfg.control,
            frame_width=cfg.camera.width,
            frame_height=cfg.camera.height,
        )

        # ── 3D pipeline ──────────────────────────────────────────────
        calib_path = PROJECT_ROOT / cfg.geometry.calibration_path
        self.calib: Optional[CameraIntrinsics] = load_calibration(calib_path)
        self.camera_pose = CameraPose(
            height_m=cfg.geometry.camera_height_m,
            pitch_deg=cfg.geometry.camera_pitch_deg,
            yaw_deg=cfg.geometry.camera_yaw_deg,
        )
        self.traj_filter = TrajectoryFilter(cfg.trajectory)
        self.ballistic = BallisticSolver(cfg.trajectory)
        self.latest_landing: Optional[LandingPoint] = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.camera.fps)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")

        if self.yolo_worker is not None:
            self.yolo_worker.start()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                self._step(frame)

                if self.cfg.display.enabled:
                    self._draw(frame)
                    cv2.imshow(self.cfg.display.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        finally:
            cap.release()
            if self.yolo_worker is not None:
                self.yolo_worker.stop()
            cv2.destroyAllWindows()

    # ── step ─────────────────────────────────────────────────────────

    def _step(self, frame) -> None:
        yolo_detection = self._pop_yolo()
        if yolo_detection is not None:
            self._accept_detection(yolo_detection)

        hsv_detection = None
        allow_hsv_init = self.yolo_worker is None
        if self.state.ready or allow_hsv_init:
            hsv_detection = self.hsv_tracker.track(frame, self.state)

        if hsv_detection is not None:
            self._accept_detection(hsv_detection)
        else:
            self.state.mark_missing()

        if self.state.missing_frames > self.cfg.filter.max_missing_frames:
            self.state.reset()

        # ── 3D 估计 ──────────────────────────────────────────────
        self._update_3d(hsv_detection)

        self._maybe_request_yolo(frame)
        self._send_control()

    def _update_3d(self, detection: Optional[Detection]) -> None:
        """从 2D 检测更新 3D 轨迹和落点预测."""
        if detection is None or self.calib is None:
            # 标定未加载时不更新 3D, 但仍然衰减置信度
            self.latest_landing = None
            return

        u, v = detection.center
        radius_px = detection.radius
        self.state.last_radius_px = radius_px
        pos_3d = detect_to_robot_3d(
            u, v, radius_px, self.calib, self.camera_pose,
            real_diameter_m=self.cfg.geometry.ball_diameter_m,
        )

        if pos_3d is not None:
            det_3d = Detection3D(
                pos=pos_3d,
                confidence=detection.confidence,
                timestamp=detection.timestamp,
                radius_px=radius_px,
            )
            self.state.pos_3d = pos_3d
            self.state.history_3d.append(det_3d)
            # 限制历史长度
            if len(self.state.history_3d) > self.cfg.trajectory.max_history:
                self.state.history_3d = self.state.history_3d[
                    -self.cfg.trajectory.max_history :
                ]

            # Kalman 更新
            self.traj_filter.update(det_3d)
            # 更新速度估计
            vel = self.traj_filter.velocity
            self.state.vel_3d = (float(vel[0]), float(vel[1]), float(vel[2]))

            # 落点预测
            self.latest_landing = self.ballistic.solve(self.traj_filter)

    # ── detection ────────────────────────────────────────────────────

    def _accept_detection(self, detection: Detection) -> None:
        self.state = self.filter.update(self.state, detection)

    # ── YOLO ─────────────────────────────────────────────────────────

    def _maybe_request_yolo(self, frame) -> None:
        if self.yolo_worker is None:
            return

        interval_s = self.cfg.yolo.periodic_interval_ms / 1000.0
        confidence_low = (
            self.state.confidence < self.cfg.yolo.request_when_confidence_below
        )
        periodic_due = now() - self.last_yolo_request > interval_s
        should_request = (not self.state.ready) or confidence_low or periodic_due

        if should_request and self.yolo_worker.request(frame):
            self.last_yolo_request = now()

    def _pop_yolo(self) -> Optional[Detection]:
        if self.yolo_worker is None:
            return None
        return self.yolo_worker.pop_latest()

    # ── control ──────────────────────────────────────────────────────

    def _send_control(self) -> None:
        if not self.state.ready:
            return
        if self.latest_landing is not None:
            self.controller.send_landing(self.latest_landing)
        else:
            # 降级: 2D 像素目标
            target = self.filter.predict_latency(
                self.state, self.cfg.filter.prediction_latency_ms
            )
            self.controller.send_target(target)

    # ── draw ─────────────────────────────────────────────────────────

    def _draw(self, frame) -> None:
        height, width = frame.shape[:2]

        # --- status bar background ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        if self.state.center is None:
            cv2.putText(
                frame, "searching ...", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            return

        cx, cy = clamp_point(self.state.center, width, height)
        pred = self.filter.predict_latency(
            self.state, self.cfg.filter.prediction_latency_ms
        )
        px, py = clamp_point(pred, width, height)

        # 2D tracking circle
        color = (0, 255, 0) if self.state.source == "HSV" else (255, 0, 0)
        cv2.circle(frame, (cx, cy), max(3, int(self.state.radius)), color, 2)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx, cy), (px, py), (0, 0, 255), 2)

        # status text — line 1
        lines = [
            f"{self.state.source} conf={self.state.confidence} miss={self.state.missing_frames}",
        ]
        # debug: raw radius + depth formula
        if self.state.last_radius_px > 0 and self.calib is not None:
            r = self.state.last_radius_px
            zc_raw = self.calib.focal_mean * self.cfg.geometry.ball_diameter_m / (2.0 * r)
            lines.append(f"r={r:.1f}px  Zc=f*D/(2r)={zc_raw:.3f}m")
        # 3D position
        if self.state.pos_3d is not None:
            x, y, z = self.state.pos_3d
            lines.append(f"3D: ({x:.2f}, {y:.2f}, {z:.2f}) m")
        # landing prediction
        if self.latest_landing is not None:
            lp = self.latest_landing
            lx, ly, lz = lp.pos
            lines.append(
                f"Land: ({lx:.2f}, {ly:.2f}, {lz:.2f})m t={lp.t_arrival:.2f}s"
            )
        # draw text lines
        y0 = 28
        for i, txt in enumerate(lines):
            c = [(0, 255, 0), (255, 255, 0), (0, 255, 255)][i % 3]
            cv2.putText(
                frame, txt, (12, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2,
            )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_source(source: str):
        if source.isdigit():
            return int(source)
        return str(Path(source))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
