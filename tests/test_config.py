from __future__ import annotations

import pytest

from tennis_robot_sim.config import ConfigError, load_config, validate_config


def test_load_config_required_sections():
    cfg = load_config()
    for section in ["camera", "detector", "ball", "court", "robot", "controller", "imu", "sim", "logging", "safety"]:
        assert section in cfg


def test_invalid_config_raises_clear_error():
    cfg = load_config()
    del cfg["camera"]["width"]
    with pytest.raises(ConfigError, match="camera"):
        validate_config(cfg)

