from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import YoloConfig
from ..state import Detection, now


class YOLODetector:
    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        self.model = None
        self.class_index: Optional[int] = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not Path(self.cfg.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.cfg.model_path}. "
                "Train/export a model or edit configs/app.yaml."
            )

        from ultralytics import YOLO

        self.model = YOLO(self.cfg.model_path)
        self.class_index = self._resolve_class_index()

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        self.load()
        assert self.model is not None

        results = self.model.predict(
            source=frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        best = None
        best_conf = -1.0
        for box in boxes:
            cls = int(box.cls[0].item()) if box.cls is not None else -1
            if self.class_index is not None and cls != self.class_index:
                continue
            conf = float(box.conf[0].item())
            if conf > best_conf:
                best_conf = conf
                best = box

        if best is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = 0.25 * ((x2 - x1) + (y2 - y1))
        return Detection(
            center=(cx, cy),
            bbox=(x1, y1, x2, y2),
            confidence=best_conf,
            source="YOLO",
            timestamp=now(),
            radius=radius,
        )

    def _resolve_class_index(self) -> Optional[int]:
        assert self.model is not None
        names = getattr(self.model, "names", None)
        if not names:
            return None

        target = self.cfg.class_name.strip().lower()
        items = names.items() if hasattr(names, "items") else enumerate(names)
        for idx, name in items:
            if str(name).strip().lower() == target:
                return int(idx)
        if target in {"ball", "tennis_ball", "tennis ball"} and len(names) == 1:
            if hasattr(names, "keys"):
                return int(next(iter(names.keys())))
            return 0
        return None
