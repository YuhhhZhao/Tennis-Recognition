from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import Detection
from tennis_robot_sim.estimation.geometry import detection_to_world, pixel_to_ground, world_to_pixel
from tennis_robot_sim.sim.camera import PinholeCamera


def test_detection_to_world_round_trip():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    point = (2.0, -0.4, 0.9)
    projected = world_to_pixel(camera, point, cfg["ball"]["radius_m"])
    assert projected is not None
    det = Detection(0, 0.0, (projected[0], projected[1]), projected[2], 1.0)
    estimated = detection_to_world(camera, det, cfg["ball"]["radius_m"])
    assert estimated is not None
    assert abs(estimated[0] - point[0]) < 1e-6
    assert abs(estimated[1] - point[1]) < 1e-6
    assert abs(estimated[2] - point[2]) < 1e-6


def test_pixel_to_ground_center_corner():
    cfg = load_config()
    camera = PinholeCamera.from_config(cfg)
    for point in [(2.0, 0.0, 0.0), (4.0, 1.5, 0.0)]:
        projected = world_to_pixel(camera, point)
        assert projected is not None
        ground = pixel_to_ground(camera, (projected[0], projected[1]))
        assert ground is not None
        assert abs(ground[0] - point[0]) < 1e-6
        assert abs(ground[1] - point[1]) < 1e-6

