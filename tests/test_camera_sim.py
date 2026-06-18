from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.sim.camera import PinholeCamera


def test_projection_and_ground_inverse():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    point = (2.5, 0.7, 0.0)
    pixel = camera.world_to_pixel(point)
    assert pixel is not None
    ground = camera.pixel_to_ground((pixel[0], pixel[1]))
    assert ground is not None
    assert abs(ground[0] - point[0]) < 1e-6
    assert abs(ground[1] - point[1]) < 1e-6


def test_projected_default_ball_inside_frame():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    pixel = camera.world_to_pixel((1.2, 0.4, 1.0), cfg["ball"]["radius_m"])
    assert pixel is not None
    u, v, r = pixel
    assert 0 <= u < camera.width
    assert 0 <= v < camera.height
    assert r > 1

