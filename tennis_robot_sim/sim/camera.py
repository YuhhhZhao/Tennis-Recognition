from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "PinholeCamera":
        cam = cfg["camera"]
        return cls(
            width=int(cam["width"]),
            height=int(cam["height"]),
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            height_m=float(cam["height_m"]),
            pitch_deg=float(cam.get("pitch_deg", 0.0)),
            yaw_deg=float(cam.get("yaw_deg", 0.0)),
        )

    @property
    def focal_mean(self) -> float:
        return 0.5 * (self.fx + self.fy)

    @property
    def center_world(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.height_m], dtype=float)

    def _rotation_world_to_camera(self) -> np.ndarray:
        base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float)
        pitch = np.deg2rad(self.pitch_deg)
        yaw = np.deg2rad(self.yaw_deg)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=float)
        r_yaw_world = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return r_pitch @ base @ r_yaw_world.T

    def world_to_camera(self, point_world: Vec3) -> np.ndarray:
        rel = np.asarray(point_world, dtype=float) - self.center_world
        return self._rotation_world_to_camera() @ rel

    def camera_to_world(self, point_camera: Vec3) -> np.ndarray:
        return self._rotation_world_to_camera().T @ np.asarray(point_camera, dtype=float) + self.center_world

    def world_to_pixel(self, point_world: Vec3, radius_m: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
        pc = self.world_to_camera(point_world)
        if pc[2] <= 1e-6:
            return None
        u = self.fx * pc[0] / pc[2] + self.cx
        v = self.fy * pc[1] / pc[2] + self.cy
        radius_px = 0.0
        if radius_m is not None:
            radius_px = self.focal_mean * radius_m / pc[2]
        return (float(u), float(v), float(radius_px))

    def pixel_to_ray_world(self, pixel: Vec2) -> np.ndarray:
        u, v = pixel
        ray_camera = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0], dtype=float)
        ray_camera /= np.linalg.norm(ray_camera)
        ray_world = self._rotation_world_to_camera().T @ ray_camera
        return ray_world / np.linalg.norm(ray_world)

    def pixel_to_ground(self, pixel: Vec2, ground_z: float = 0.0) -> Optional[Tuple[float, float]]:
        ray = self.pixel_to_ray_world(pixel)
        if abs(ray[2]) < 1e-9:
            return None
        t = (ground_z - self.height_m) / ray[2]
        if t <= 0:
            return None
        point = self.center_world + t * ray
        return (float(point[0]), float(point[1]))

    def pixel_radius_to_world(self, pixel: Vec2, radius_px: float, ball_radius_m: float) -> Optional[Vec3]:
        if radius_px <= 0:
            return None
        depth = self.focal_mean * ball_radius_m / radius_px
        u, v = pixel
        point_camera = ((u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth)
        world = self.camera_to_world(point_camera)
        return (float(world[0]), float(world[1]), float(world[2]))

