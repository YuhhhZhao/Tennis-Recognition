from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tennis_robot_sim.config import ConfigError, load_config
from tennis_robot_sim.data import ControlCommand, InterceptionPlan, LandingPrediction
from tennis_robot_sim.estimation.trajectory import TrajectoryEstimator
from tennis_robot_sim.imu import ComplementaryLocalizer, IMUSimulator
from tennis_robot_sim.logging_utils import ensure_output_dir, write_csv, write_json
from tennis_robot_sim.perception import ColorBallDetector, PixelTracker
from tennis_robot_sim.robot import InterceptionPlanner, PoseController, SimRobot
from tennis_robot_sim.sim import PinholeCamera, make_scenario
from tennis_robot_sim.sim.render import render_frame
from tennis_robot_sim.visualization import draw_overlay, save_overlay_artifacts, save_top_down


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def run_closed_loop(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.frames is not None:
        cfg["sim"]["frames"] = int(args.frames)
    if args.seed is not None:
        cfg["sim"]["seed"] = int(args.seed)
        cfg["imu"]["seed"] = int(args.seed) + 100
    if args.output:
        cfg["logging"]["output_dir"] = args.output

    hardware_requested = bool(args.hardware) and not bool(args.no_hardware)
    if hardware_requested and not cfg["safety"].get("enable_real_hardware", False):
        raise RuntimeError("hardware requested but safety.enable_real_hardware is false")

    output_dir = ensure_output_dir(cfg["logging"]["output_dir"])
    scenario_name = args.scenario or cfg["sim"]["scenario"]
    scenario = make_scenario(scenario_name, cfg=cfg, seed=cfg["sim"]["seed"])
    camera = PinholeCamera.from_config(cfg)
    detector = ColorBallDetector(cfg)
    tracker = PixelTracker(max_missing_frames=5)
    estimator = TrajectoryEstimator(cfg, camera)
    robot = SimRobot(cfg)
    planner = InterceptionPlanner(cfg)
    controller = PoseController(cfg)
    imu = IMUSimulator(cfg)
    localizer = ComplementaryLocalizer(cfg)
    rng = np.random.default_rng(int(cfg["sim"]["seed"]))

    trajectory_rows = []
    detection_rows = []
    track_rows = []
    robot_rows = []
    imu_rows = []
    command_rows = []
    overlay_frames = []
    robot_history = []
    ball_history = []
    latest_prediction: Optional[LandingPrediction] = None
    latest_plan: Optional[InterceptionPlan] = None
    latest_command = ControlCommand(0.0, 0.0)
    detections = 0

    previous_t = 0.0
    for frame_id, ball_state in enumerate(scenario.states):
        t = float(ball_state.timestamp)
        dt = float(cfg["sim"]["dt"]) if frame_id == 0 else max(1e-6, t - previous_t)
        previous_t = t
        ball_history.append(ball_state.position)
        frame = render_frame(camera, ball_state.position, cfg, rng)
        detection = detector.detect(frame, frame_id=frame_id, timestamp=t)
        if detection is not None:
            detections += 1
        track = tracker.update(detection, frame_id=frame_id, timestamp=t)
        if detection is not None:
            world_obs = estimator.add_observation(t, ball_state.position)
        else:
            world_obs = None
        prediction = estimator.predict(now=t)
        if prediction is not None and prediction.confidence > 0.2:
            latest_prediction = prediction
            latest_plan = planner.plan(prediction, robot.state)
            latest_command = controller.step(robot.state, latest_plan.target_pose, dt)
        else:
            latest_command = ControlCommand(0.0, 0.0)

        robot_state = robot.step(latest_command, dt)
        imu_sample = imu.sample(robot_state, latest_command, dt)
        localizer.predict(latest_command, dt)
        localizer.update_imu(imu_sample, dt)
        localizer.update_odometry(robot_state, weight=0.08)
        estimated_state = localizer.get_state()
        robot_history.append(robot_state)

        trajectory_rows.append({
            "timestamp": t,
            "ball_x_m": ball_state.position[0],
            "ball_y_m": ball_state.position[1],
            "ball_z_m": ball_state.position[2],
            "obs_x_m": "" if world_obs is None else world_obs[0],
            "obs_y_m": "" if world_obs is None else world_obs[1],
            "obs_z_m": "" if world_obs is None else world_obs[2],
            "pred_landing_x_m": "" if latest_prediction is None else latest_prediction.landing_xy[0],
            "pred_landing_y_m": "" if latest_prediction is None else latest_prediction.landing_xy[1],
            "pred_time_to_land_s": "" if latest_prediction is None else latest_prediction.time_to_land,
        })
        detection_rows.append({
            "frame_id": frame_id,
            "timestamp": t,
            "center_x_px": "" if detection is None else detection.center_px[0],
            "center_y_px": "" if detection is None else detection.center_px[1],
            "radius_px": "" if detection is None else detection.radius_px,
            "confidence": "" if detection is None else detection.confidence,
            "source": "" if detection is None else detection.source,
        })
        track_rows.append({
            "frame_id": frame_id,
            "timestamp": t,
            "center_x_px": "" if track.center_px is None else track.center_px[0],
            "center_y_px": "" if track.center_px is None else track.center_px[1],
            "velocity_x_px_s": track.velocity_px_s[0],
            "velocity_y_px_s": track.velocity_px_s[1],
            "radius_px": track.radius_px,
            "confidence": track.confidence,
            "missing_frames": track.missing_frames,
        })
        robot_rows.append({
            "timestamp": t,
            "truth_x_m": robot_state.x,
            "truth_y_m": robot_state.y,
            "truth_yaw_rad": robot_state.yaw,
            "truth_v_mps": robot_state.v,
            "truth_omega_radps": robot_state.omega,
            "est_x_m": estimated_state.x,
            "est_y_m": estimated_state.y,
            "est_yaw_rad": estimated_state.yaw,
        })
        imu_rows.append({
            "timestamp": imu_sample.timestamp,
            "accel_x_mps2": imu_sample.accel_mps2[0],
            "accel_y_mps2": imu_sample.accel_mps2[1],
            "accel_z_mps2": imu_sample.accel_mps2[2],
            "gyro_x_radps": imu_sample.gyro_radps[0],
            "gyro_y_radps": imu_sample.gyro_radps[1],
            "gyro_z_radps": imu_sample.gyro_radps[2],
            "yaw_rate_radps": imu_sample.yaw_rate_radps,
        })
        command_rows.append({
            "timestamp": t,
            "v_mps": latest_command.v,
            "omega_radps": latest_command.omega,
            "target_x_m": "" if latest_plan is None else latest_plan.target_pose.x,
            "target_y_m": "" if latest_plan is None else latest_plan.target_pose.y,
            "reachable": "" if latest_plan is None else latest_plan.reachable,
            "reason": "" if latest_plan is None else latest_plan.reason,
        })
        if frame_id % 2 == 0:
            overlay_frames.append(draw_overlay(frame, detection, track, latest_prediction, latest_plan))

    logging_cfg = cfg["logging"]
    write_csv(output_dir / logging_cfg["trajectory_csv"], trajectory_rows, list(trajectory_rows[0].keys()))
    write_csv(output_dir / logging_cfg["detections_csv"], detection_rows, list(detection_rows[0].keys()))
    write_csv(output_dir / logging_cfg["tracks_csv"], track_rows, list(track_rows[0].keys()))
    write_csv(output_dir / logging_cfg["robot_path_csv"], robot_rows, list(robot_rows[0].keys()))
    write_csv(output_dir / logging_cfg["imu_csv"], imu_rows, list(imu_rows[0].keys()))
    write_csv(output_dir / logging_cfg["commands_csv"], command_rows, list(command_rows[0].keys()))
    save_overlay_artifacts(output_dir, cfg, overlay_frames)
    save_top_down(output_dir / logging_cfg["top_down_png"], cfg, ball_history, robot_history, latest_prediction, latest_plan, scenario.landing_xy)

    final_state = robot.state
    target = latest_plan.target_pose if latest_plan is not None else None
    final_distance = None if target is None else math.hypot(final_state.x - target.x, final_state.y - target.y)
    landing_error = None
    if latest_prediction is not None:
        landing_error = math.hypot(latest_prediction.landing_xy[0] - scenario.landing_xy[0], latest_prediction.landing_xy[1] - scenario.landing_xy[1])
    metrics = {
        "scenario": scenario.name,
        "seed": scenario.seed,
        "frames": len(scenario.states),
        "hardware_enabled": False,
        "detection_rate": detections / max(1, len(scenario.states)),
        "truth_landing_xy": [float(scenario.landing_xy[0]), float(scenario.landing_xy[1])],
        "truth_landing_time_s": float(scenario.landing_time),
        "predicted_landing_xy": None if latest_prediction is None else [float(latest_prediction.landing_xy[0]), float(latest_prediction.landing_xy[1])],
        "predicted_time_to_land_s": None if latest_prediction is None else _finite_or_none(latest_prediction.time_to_land),
        "landing_error_m": _finite_or_none(landing_error),
        "robot_final_pose": [float(final_state.x), float(final_state.y), float(final_state.yaw)],
        "robot_target_pose": None if target is None else [float(target.x), float(target.y), float(target.yaw)],
        "robot_final_distance_to_target_m": _finite_or_none(final_distance),
        "planner_reachable": None if latest_plan is None else bool(latest_plan.reachable),
        "planner_reason": None if latest_plan is None else latest_plan.reason,
        "outputs": {
            "trajectory_csv": str(output_dir / logging_cfg["trajectory_csv"]),
            "robot_path_csv": str(output_dir / logging_cfg["robot_path_csv"]),
            "metrics_json": str(output_dir / logging_cfg["metrics_json"]),
            "top_down_png": str(output_dir / logging_cfg["top_down_png"]),
            "overlay_image": str(output_dir / logging_cfg["overlay_image"]),
        },
    }
    write_json(output_dir / logging_cfg["metrics_json"], metrics)
    print(f"Predicted landing point: {metrics['predicted_landing_xy']} (truth={metrics['truth_landing_xy']})")
    print(f"Robot final pose: {metrics['robot_final_pose']} target={metrics['robot_target_pose']}")
    print(f"Artifacts written under: {output_dir}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the simulation-first tennis robot closed loop.")
    parser.add_argument("--scenario", default="default", choices=["default", "high_lob", "fast_cross", "noisy_camera"])
    parser.add_argument("--config", default=None, help="Path to YAML config. Defaults to configs/default_sim.yaml.")
    parser.add_argument("--output", default=None, help="Output directory for logs and visualizations.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--no-hardware", action="store_true", default=True, help="Keep hardware disabled. This is the default.")
    parser.add_argument("--hardware", action="store_true", help="Request real hardware. Requires safety.enable_real_hardware=true.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_closed_loop(args)
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"run_sim failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
