from __future__ import annotations

from typing import Optional, Tuple

from tennis_robot_sim.data import Detection
from tennis_robot_sim.sim.camera import PinholeCamera


def world_to_pixel(camera: PinholeCamera, point_world: tuple[float, float, float], radius_m: float = 0.0) -> Optional[Tuple[float, float, float]]:
    return camera.world_to_pixel(point_world, radius_m if radius_m > 0 else None)


def pixel_to_ground(camera: PinholeCamera, pixel: tuple[float, float]) -> Optional[Tuple[float, float]]:
    return camera.pixel_to_ground(pixel)


def detection_to_world(camera: PinholeCamera, detection: Detection, ball_radius_m: float) -> Optional[tuple[float, float, float]]:
    return camera.pixel_radius_to_world(detection.center_px, detection.radius_px, ball_radius_m)

