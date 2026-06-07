from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import HSVConfig, ROIConfig
from ..state import BBox, Detection, TrackState, now


@dataclass
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class HSVTracker:
    def __init__(self, hsv_cfg: HSVConfig, roi_cfg: ROIConfig):
        self.hsv_cfg = hsv_cfg
        self.roi_cfg = roi_cfg
        self.lower = np.array(hsv_cfg.lower, dtype=np.uint8)
        self.upper = np.array(hsv_cfg.upper, dtype=np.uint8)

    def track(self, frame: np.ndarray, state: TrackState) -> Optional[Detection]:
        roi = self._make_roi(frame, state)
        crop = frame[roi.y1 : roi.y2, roi.x1 : roi.x2]
        if crop.size == 0:
            return None

        mask = self._make_mask(crop)
        candidate = self._best_candidate(mask)
        if candidate is None:
            return None

        x, y, w, h, area, circularity, fill_ratio, contour = candidate
        if not self._passes_shape_checks(w, h, area, circularity, fill_ratio):
            return None

        cx = roi.x1 + x + w / 2.0
        cy = roi.y1 + y + h / 2.0
        bbox: BBox = (roi.x1 + x, roi.y1 + y, roi.x1 + x + w, roi.y1 + y + h)

        # 用 fitEllipse 做最小二乘椭圆拟合 — 对膨胀噪声比 minEnclosingCircle 更鲁棒
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            # ellipse = ((cx, cy), (major_axis, minor_axis), angle)
            major, minor = ellipse[1]
            radius = (major + minor) / 4.0
        else:
            radius = 0.25 * (w + h)
        confidence = min(0.99, 0.45 + circularity * 0.35 + fill_ratio * 0.20)

        return Detection(
            center=(cx, cy),
            bbox=bbox,
            confidence=confidence,
            source="HSV",
            timestamp=now(),
            radius=radius,
        )

    def _make_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # close: dilate→erode, 填充掩码内部空洞并扩张至球边缘
        if self.hsv_cfg.close_iterations > 0:
            mask = cv2.dilate(mask, None, iterations=self.hsv_cfg.close_iterations)
            mask = cv2.erode(mask, None, iterations=self.hsv_cfg.close_iterations)
        if self.hsv_cfg.erode_iterations > 0:
            mask = cv2.erode(mask, None, iterations=self.hsv_cfg.erode_iterations)
        if self.hsv_cfg.dilate_iterations > 0:
            mask = cv2.dilate(mask, None, iterations=self.hsv_cfg.dilate_iterations)
        return mask

    def _best_candidate(
        self, mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int, float, float, float, np.ndarray]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best = None
        best_contour = None
        best_score = -1.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            fill_ratio = area / max(1.0, float(w * h))
            score = area * (0.5 + circularity) * (0.5 + fill_ratio)
            if score > best_score:
                best_score = score
                best = (x, y, w, h, area, circularity, fill_ratio)
                best_contour = contour

        if best is None:
            return None
        return (*best, best_contour)

    def _passes_shape_checks(
        self,
        width: int,
        height: int,
        area: float,
        circularity: float,
        fill_ratio: float,
    ) -> bool:
        cfg = self.hsv_cfg
        if area < cfg.min_area or area > cfg.max_area:
            return False
        aspect = max(width, height) / max(1.0, float(min(width, height)))
        if aspect > cfg.max_aspect_ratio:
            return False
        if circularity < cfg.min_circularity:
            return False
        if fill_ratio < cfg.min_mask_fill_ratio:
            return False
        return True

    def _make_roi(self, frame: np.ndarray, state: TrackState) -> ROI:
        height, width = frame.shape[:2]
        if not self.roi_cfg.enabled or state.center is None:
            return ROI(0, 0, width, height)

        cx, cy = state.center
        vx, vy = state.velocity
        velocity_margin = (
            (abs(vx) + abs(vy)) * 0.016 * self.roi_cfg.velocity_margin_scale
        )
        size = int(
            max(
                self.roi_cfg.min_size_px,
                2 * state.radius + self.roi_cfg.base_margin_px + velocity_margin,
            )
        )
        size = min(size, self.roi_cfg.max_size_px)
        half = size // 2

        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(width, int(cx) + half)
        y2 = min(height, int(cy) + half)
        return ROI(x1, y1, x2, y2)
