from __future__ import annotations

import math
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


@dataclass
class Candidate:
    center: Tuple[float, float]
    radius: float
    bbox: BBox
    area: float
    circularity: float
    fill_ratio: float
    circle_fill_ratio: float
    touches_border: bool
    score: float


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
        predicted = self._predicted_center_in_roi(roi, state)
        candidate = self._best_candidate(mask, predicted)
        if candidate is None:
            return None

        cx = roi.x1 + candidate.center[0]
        cy = roi.y1 + candidate.center[1]
        x1, y1, x2, y2 = candidate.bbox
        bbox: BBox = (roi.x1 + x1, roi.y1 + y1, roi.x1 + x2, roi.y1 + y2)
        confidence = min(
            0.99,
            0.45 + candidate.circularity * 0.35 + candidate.fill_ratio * 0.20,
        )

        return Detection(
            center=(cx, cy),
            bbox=bbox,
            confidence=confidence,
            source="HSV",
            timestamp=now(),
            radius=candidate.radius,
        )

    def _make_mask(self, bgr: np.ndarray) -> np.ndarray:
        # Small blur reduces sensor noise and JPEG block artifacts before thresholding.
        bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # close: 填充掩码内部空洞并尽量保持圆形边界.
        if self.hsv_cfg.close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.hsv_cfg.close_iterations,
            )
        if self.hsv_cfg.erode_iterations > 0:
            mask = cv2.erode(mask, kernel, iterations=self.hsv_cfg.erode_iterations)
        if self.hsv_cfg.dilate_iterations > 0:
            mask = cv2.dilate(mask, kernel, iterations=self.hsv_cfg.dilate_iterations)
        mask = cv2.medianBlur(mask, 3)
        return mask

    def _best_candidate(
        self,
        mask: np.ndarray,
        predicted_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Candidate]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best: Optional[Candidate] = None
        best_score = -1.0
        mask_shape = mask.shape[:2]
        for contour in contours:
            candidate = self._candidate_from_contour(
                contour, mask_shape, predicted_center
            )
            if candidate is None:
                continue
            if candidate.score > best_score:
                best_score = candidate.score
                best = candidate

        return best

    def _candidate_from_contour(
        self,
        contour: np.ndarray,
        mask_shape: Tuple[int, int],
        predicted_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Candidate]:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return None

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        fill_ratio = area / max(1.0, float(w * h))

        (circle_cx, circle_cy), enclosing_radius = cv2.minEnclosingCircle(contour)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cx, cy = ellipse[0]
            major, minor = ellipse[1]
            radius = max(1.0, (major + minor) / 4.0)
        else:
            cx, cy = circle_cx, circle_cy
            radius = enclosing_radius

        enclosing_radius = max(1.0, float(enclosing_radius))
        circle_fill_ratio = area / max(np.pi * enclosing_radius * enclosing_radius, 1.0)
        mask_h, mask_w = mask_shape
        margin = max(0, int(self.hsv_cfg.border_margin_px))
        touches_border = (
            x <= margin
            or y <= margin
            or x + w >= mask_w - margin
            or y + h >= mask_h - margin
        )
        if not self._passes_shape_checks(
            w,
            h,
            area,
            circularity,
            fill_ratio,
            circle_fill_ratio,
            radius,
            touches_border,
        ):
            return None

        score = (
            area
            * (0.5 + circularity)
            * (0.5 + fill_ratio)
            * (0.5 + circle_fill_ratio)
        )
        if predicted_center is not None:
            px, py = predicted_center
            dist = math.hypot(cx - px, cy - py)
            # Bias toward the predicted track without making it a hard gate.
            distance_scale = max(25.0, 4.0 * radius)
            distance_score = 1.0 / (1.0 + dist / distance_scale)
            score *= 0.5 + distance_score

        return Candidate(
            center=(float(cx), float(cy)),
            radius=float(radius),
            bbox=(x, y, x + w, y + h),
            area=area,
            circularity=float(circularity),
            fill_ratio=float(fill_ratio),
            circle_fill_ratio=float(circle_fill_ratio),
            touches_border=touches_border,
            score=float(score),
        )

    def _passes_shape_checks(
        self,
        width: int,
        height: int,
        area: float,
        circularity: float,
        fill_ratio: float,
        circle_fill_ratio: float,
        radius: float,
        touches_border: bool,
    ) -> bool:
        cfg = self.hsv_cfg
        if area < cfg.min_area or area > cfg.max_area:
            return False
        if radius < cfg.min_radius_px or radius > cfg.max_radius_px:
            return False
        aspect = max(width, height) / max(1.0, float(min(width, height)))
        if aspect > cfg.max_aspect_ratio:
            return False
        if circularity < cfg.min_circularity:
            return False
        if fill_ratio < cfg.min_mask_fill_ratio:
            return False
        if fill_ratio > cfg.max_mask_fill_ratio:
            return False
        if circle_fill_ratio < cfg.min_circle_fill_ratio:
            return False
        if cfg.reject_border_touch and touches_border:
            return False
        return True

    def _make_roi(self, frame: np.ndarray, state: TrackState) -> ROI:
        height, width = frame.shape[:2]
        if not self.roi_cfg.enabled or state.center is None:
            return ROI(0, 0, width, height)

        cx, cy = self._predicted_center(state)
        vx, vy = state.velocity
        age_s = self._state_age_s(state)
        travel_margin = (abs(vx) + abs(vy)) * age_s * self.roi_cfg.velocity_margin_scale
        missing_margin = state.missing_frames * max(6.0, state.radius * 0.35)
        size = int(
            max(
                self.roi_cfg.min_size_px,
                2 * state.radius
                + self.roi_cfg.base_margin_px
                + travel_margin
                + missing_margin,
            )
        )
        size = min(size, self.roi_cfg.max_size_px)
        half = size // 2

        cx = max(0.0, min(float(width - 1), cx))
        cy = max(0.0, min(float(height - 1), cy))
        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(width, int(cx) + half)
        y2 = min(height, int(cy) + half)
        return ROI(x1, y1, x2, y2)

    def _predicted_center_in_roi(
        self, roi: ROI, state: TrackState
    ) -> Optional[Tuple[float, float]]:
        if state.center is None:
            return None
        cx, cy = self._predicted_center(state)
        return (cx - roi.x1, cy - roi.y1)

    def _predicted_center(self, state: TrackState) -> Tuple[float, float]:
        cx, cy = state.center or (0.0, 0.0)
        vx, vy = state.velocity
        age_s = self._state_age_s(state)
        return (cx + vx * age_s, cy + vy * age_s)

    @staticmethod
    def _state_age_s(state: TrackState) -> float:
        if state.last_update <= 0.0:
            return 0.0
        return max(0.0, min(0.5, now() - state.last_update))
