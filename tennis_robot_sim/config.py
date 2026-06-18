from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_sim.yaml"


class ConfigError(ValueError):
    """Raised when the simulation config is missing required data."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "width": 640,
        "height": 360,
        "fps": 60,
        "fx": 520.0,
        "fy": 520.0,
        "cx": 320.0,
        "cy": 180.0,
        "height_m": 1.20,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
    },
    "detector": {
        "hsv_lower": [24, 60, 80],
        "hsv_upper": [48, 255, 255],
        "min_area": 12.0,
        "max_area": 50000.0,
        "min_circularity": 0.45,
        "min_radius_px": 2.0,
        "max_radius_px": 80.0,
        "erode_iterations": 1,
        "dilate_iterations": 2,
    },
    "ball": {
        "radius_m": 0.0335,
        "gravity_mps2": -9.81,
        "bounce_coefficient": 0.62,
    },
    "court": {
        "length_m": 12.0,
        "width_m": 8.0,
        "x_min": 0.0,
        "x_max": 12.0,
        "y_min": -4.0,
        "y_max": 4.0,
    },
    "robot": {
        "start_pose": [0.0, 0.0, 0.0],
        "max_speed_mps": 4.0,
        "max_angular_rate_radps": 5.0,
        "max_accel_mps2": 6.0,
        "wheel_base_m": 0.32,
        "catch_offset_m": [0.0, 0.0],
        "position_tolerance_m": 0.15,
    },
    "controller": {
        "linear_gain": 2.4,
        "angular_gain": 5.0,
        "yaw_tolerance_rad": 0.08,
        "timeout_s": 4.0,
    },
    "trajectory": {
        "min_samples_for_fit": 6,
        "max_history": 80,
        "max_landing_time_s": 3.0,
        "max_clean_error_m": 0.5,
    },
    "imu": {
        "accel_noise_std": 0.04,
        "gyro_noise_std": 0.01,
        "accel_bias": [0.0, 0.0, 0.0],
        "gyro_bias": [0.0, 0.0, 0.0],
        "timestamp_jitter_std": 0.0,
        "seed": 123,
        "yaw_complementary_alpha": 0.92,
    },
    "sim": {
        "scenario": "default",
        "seed": 7,
        "frames": 90,
        "dt": 0.0166666667,
        "render_noise": 0.0,
        "save_overlay_video": True,
    },
    "logging": {
        "output_dir": "outputs/smoke",
        "trajectory_csv": "trajectory.csv",
        "detections_csv": "detections.csv",
        "tracks_csv": "tracks.csv",
        "robot_path_csv": "robot_path.csv",
        "imu_csv": "imu.csv",
        "commands_csv": "commands.csv",
        "metrics_json": "metrics.json",
        "top_down_png": "top_down.png",
        "overlay_video": "overlay.mp4",
        "overlay_image": "overlay_last.png",
    },
    "safety": {
        "enable_real_hardware": False,
        "require_hardware_flag": True,
        "max_command_speed_mps": 4.0,
        "max_command_omega_radps": 5.0,
        "dry_run": True,
    },
}

REQUIRED_SECTIONS = (
    "camera",
    "detector",
    "ball",
    "court",
    "robot",
    "controller",
    "imu",
    "sim",
    "logging",
    "safety",
)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_keys(section: Mapping[str, Any], keys: Iterable[str], name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"config section '{name}' is missing keys: {', '.join(missing)}")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    missing_sections = [name for name in REQUIRED_SECTIONS if name not in cfg]
    if missing_sections:
        raise ConfigError(f"config is missing sections: {', '.join(missing_sections)}")

    _require_keys(cfg["camera"], ("width", "height", "fps", "fx", "fy", "cx", "cy", "height_m"), "camera")
    _require_keys(cfg["detector"], ("hsv_lower", "hsv_upper", "min_area", "max_area"), "detector")
    _require_keys(cfg["ball"], ("radius_m", "gravity_mps2", "bounce_coefficient"), "ball")
    _require_keys(cfg["court"], ("x_min", "x_max", "y_min", "y_max"), "court")
    _require_keys(cfg["robot"], ("start_pose", "max_speed_mps", "max_angular_rate_radps", "max_accel_mps2"), "robot")
    _require_keys(cfg["controller"], ("linear_gain", "angular_gain"), "controller")
    _require_keys(cfg["imu"], ("accel_noise_std", "gyro_noise_std", "seed"), "imu")
    _require_keys(cfg["sim"], ("scenario", "seed", "frames", "dt"), "sim")
    _require_keys(cfg["logging"], ("output_dir", "metrics_json", "top_down_png"), "logging")
    _require_keys(cfg["safety"], ("enable_real_hardware", "max_command_speed_mps", "max_command_omega_radps"), "safety")

    if cfg["camera"]["width"] <= 0 or cfg["camera"]["height"] <= 0:
        raise ConfigError("camera width and height must be positive")
    if cfg["ball"]["radius_m"] <= 0:
        raise ConfigError("ball.radius_m must be positive")
    if cfg["sim"]["dt"] <= 0:
        raise ConfigError("sim.dt must be positive")
    if cfg["robot"]["max_speed_mps"] <= 0:
        raise ConfigError("robot.max_speed_mps must be positive")
    return cfg


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"config file must contain a mapping: {config_path}")
        cfg = _deep_merge(cfg, loaded)
    elif path is not None:
        raise ConfigError(f"config file does not exist: {config_path}")
    return validate_config(cfg)
