from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import PinholeCamera


def render_court_background(camera: PinholeCamera, court_cfg: dict) -> np.ndarray:
    frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    frame[:] = (42, 116, 58)
    horizon = int(camera.height * 0.34)
    frame[:horizon] = (185, 190, 185)
    cv2.rectangle(frame, (0, horizon), (camera.width - 1, camera.height - 1), (42, 125, 58), -1)
    for x in np.linspace(court_cfg["x_min"], court_cfg["x_max"], 7):
        p0 = camera.world_to_pixel((float(x), court_cfg["y_min"], 0.0))
        p1 = camera.world_to_pixel((float(x), court_cfg["y_max"], 0.0))
        if p0 and p1:
            cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (215, 215, 215), 1)
    for y in np.linspace(court_cfg["y_min"], court_cfg["y_max"], 5):
        p0 = camera.world_to_pixel((court_cfg["x_min"], float(y), 0.0))
        p1 = camera.world_to_pixel((court_cfg["x_max"], float(y), 0.0))
        if p0 and p1:
            cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (210, 210, 210), 1)
    return frame


def render_frame(
    camera: PinholeCamera,
    ball_position: Tuple[float, float, float],
    cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    ball_radius_m = float(cfg["ball"]["radius_m"])
    frame = render_court_background(camera, cfg["court"])
    projected = camera.world_to_pixel(ball_position, radius_m=ball_radius_m)
    if projected is not None:
        u, v, radius_px = projected
        if -50 <= u < camera.width + 50 and -50 <= v < camera.height + 50 and radius_px > 0.5:
            center = (int(round(u)), int(round(v)))
            radius = max(2, int(round(radius_px)))
            cv2.circle(frame, center, radius, (48, 226, 230), -1)
            cv2.circle(frame, center, radius, (28, 160, 180), 1)
            cv2.circle(frame, (center[0] - max(1, radius // 3), center[1] - max(1, radius // 3)), max(1, radius // 4), (150, 255, 255), -1)

    noise_std = float(cfg.get("sim", {}).get("render_noise", 0.0))
    if rng is not None and noise_std > 0:
        noise = rng.normal(0.0, noise_std, frame.shape)
        frame = np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)
    return frame


def save_video(path: str | Path, frames: list[np.ndarray], fps: int) -> bool:
    if not frames:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True
