from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from tennis_robot_sim.data import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[Detection]:
        raise NotImplementedError


class ColorBallDetector(Detector):
    def __init__(self, cfg: dict):
        self.cfg = cfg["detector"] if "detector" in cfg else cfg

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg["hsv_lower"], dtype=np.uint8)
        upper = np.array(self.cfg["hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        erode_iter = int(self.cfg.get("erode_iterations", 1))
        dilate_iter = int(self.cfg.get("dilate_iterations", 2))
        kernel = np.ones((3, 3), dtype=np.uint8)
        if erode_iter > 0:
            mask = cv2.erode(mask, kernel, iterations=erode_iter)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Detection] = None
        best_score = -1.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.get("min_area", 0.0) or area > self.cfg.get("max_area", float("inf")):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.cfg.get("min_circularity", 0.0):
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.cfg.get("min_radius_px", 0.0) or radius > self.cfg.get("max_radius_px", float("inf")):
                continue
            fill_ratio = area / max(np.pi * radius * radius, 1e-6)
            score = min(1.0, 0.55 * circularity + 0.45 * min(1.0, fill_ratio))
            if score > best_score:
                best_score = float(score)
                best = Detection(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    center_px=(float(x), float(y)),
                    radius_px=float(radius),
                    confidence=float(score),
                    source="color",
                )
        return best


class ModelAdapterStub(Detector):
    """Placeholder for future neural models; intentionally does no work."""

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[Detection]:
        return None

