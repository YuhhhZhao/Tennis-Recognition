from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from .async_yolo import AsyncYOLOWorker
from .config import AppConfig
from .control import CarController
from .filters import AlphaBetaFilter, clamp_point
from .hsv_tracker import HSVTracker
from .state import Detection, TrackState, now
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

        self._maybe_request_yolo(frame)
        self._send_control()

    def _accept_detection(self, detection: Detection) -> None:
        self.state = self.filter.update(self.state, detection)

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

    def _send_control(self) -> None:
        if not self.state.ready:
            return
        target = self.filter.predict_latency(
            self.state, self.cfg.filter.prediction_latency_ms
        )
        command = self.controller.command_from_target(target)
        self.controller.send(command)

    def _draw(self, frame) -> None:
        height, width = frame.shape[:2]
        if self.state.center is None:
            cv2.putText(
                frame,
                "searching YOLO...",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            return

        cx, cy = clamp_point(self.state.center, width, height)
        pred = self.filter.predict_latency(
            self.state, self.cfg.filter.prediction_latency_ms
        )
        px, py = clamp_point(pred, width, height)

        color = (0, 255, 0) if self.state.source == "HSV" else (255, 0, 0)
        cv2.circle(frame, (cx, cy), max(3, int(self.state.radius)), color, 2)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx, cy), (px, py), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{self.state.source} conf={self.state.confidence} miss={self.state.missing_frames}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    def _parse_source(self, source: str):
        if source.isdigit():
            return int(source)
        path = Path(source)
        return str(path)
