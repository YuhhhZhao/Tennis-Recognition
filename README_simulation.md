# Tennis Robot Simulation

This repository now includes a simulation-first closed loop for tennis ball recognition and robot interception. It runs without a camera, serial port, GPIO, motor driver, GPU, or Jetson-specific services.

## Architecture

```text
synthetic scenario
  -> pinhole camera renderer
  -> HSV tennis-ball detector
  -> pixel tracker
  -> trajectory and landing estimator
  -> interception planner
  -> pose controller
  -> simulated robot chassis
  -> IMU simulator and complementary localizer
  -> CSV/JSON logs and visualizations
```

The existing `tennis_tracker/` package remains available for real camera and YOLO work. The simulation path lives in `tennis_robot_sim/` and is the safe default.

## Setup

```bash
cd /home/nvidia/Documents/tennis-recognition/Tennis-Recognition
bash scripts/setup_env.sh
```

If conda is available, the setup script creates or updates an environment named `tennis-robot-sim`. Without conda, it falls back to `.venv`. The simulation needs `numpy`, `opencv-python`, `PyYAML`, `matplotlib`, and `pytest`. `torch`, CUDA, and `ultralytics` are optional for the simulation path.

Check the active environment:

```bash
python scripts/check_env.py
```

## Run Smoke Simulation

```bash
python -m tennis_robot_sim.run_sim --scenario default --no-hardware --output outputs/smoke
```

The command prints a predicted landing point and robot final pose. It writes:

```text
outputs/smoke/metrics.json
outputs/smoke/trajectory.csv
outputs/smoke/detections.csv
outputs/smoke/tracks.csv
outputs/smoke/robot_path.csv
outputs/smoke/imu.csv
outputs/smoke/commands.csv
outputs/smoke/top_down.png
outputs/smoke/overlay_last.png
outputs/smoke/overlay.mp4
```

## Validate

```bash
bash scripts/validate_sim.sh
```

Validation runs the environment check, all tests, a smoke simulation, and replay plot generation. Artifacts are written under `outputs/validation/`.

## Render And Replay

Render synthetic frames:

```bash
python -m tennis_robot_sim.tools.render_synthetic --frames 30 --output outputs/synthetic/default.mp4
```

Regenerate a top-down plot from logs:

```bash
python -m tennis_robot_sim.tools.replay --input outputs/smoke --output outputs/replay_test
```

Process a real video without moving hardware:

```bash
python -m tennis_robot_sim.run_video --input path/to/video.mp4 --output outputs/video_test
```

If the file is missing, the command exits with a clear error. Real video mode only runs detector/tracker logging and overlay generation.

## Module Responsibilities

- `tennis_robot_sim.config`: loads `configs/default_sim.yaml`, supplies defaults, and validates required sections.
- `tennis_robot_sim.sim`: projectile dynamics, seeded scenarios, camera projection, and synthetic rendering.
- `tennis_robot_sim.perception`: baseline HSV detector and pixel tracker.
- `tennis_robot_sim.estimation`: image/world geometry and parabolic landing prediction.
- `tennis_robot_sim.robot`: simulated unicycle chassis, planner, pose controller, and hardware safety interfaces.
- `tennis_robot_sim.imu`: noisy IMU samples and complementary pose localization.
- `tennis_robot_sim.visualization`: overlay images/videos and top-down court plots.
- `tennis_robot_sim.logging_utils`: structured CSV and JSON output helpers.

## Coordinate Frames

- Image frame: pixels `(u, v)`, origin at top-left, `u` right, `v` down.
- Camera frame: OpenCV convention, `X` right, `Y` down, `Z` forward.
- World/court frame: `x` forward from the robot/camera baseline, `y` left, `z` up, ground plane `z = 0`.
- Robot body frame: `x` forward, `y` left, yaw positive counter-clockwise in the world plane.
- IMU frame: simulated as body-aligned; gyro `z` is yaw rate in rad/s.

## IMU Limitations

The IMU simulator generates accelerometer and gyro readings from the simulated robot state with configurable bias and Gaussian noise. The localizer is a lightweight complementary filter; it does not infer absolute position from IMU acceleration and will drift without odometry or external correction.

## Hardware Safety

The simulation runner defaults to no hardware:

```bash
python -m tennis_robot_sim.run_sim --no-hardware
```

Passing `--hardware` is rejected unless `safety.enable_real_hardware: true` is set in the config. The `RealRobotInterface` also refuses to instantiate without that explicit flag. Tests and validation never open serial, GPIO, cameras, or motor devices.

To move toward real hardware safely, add a transport implementation behind `RealRobotInterface`, keep `dry_run: true` until bench-tested, clamp speeds in config, and verify emergency stop behavior before any motor is powered.

## Jetson Notes

Simulation is CPU-only. On Jetson systems, `nvidia-smi` may fail or report many N/A fields for Orin/nvgpu; this does not block simulation. Optional monitoring:

```bash
bash scripts/jetson_monitor.sh
```

By default it samples `tegrastats` for 5 seconds and exits. Set `JETSON_MONITOR_SECONDS=30` for a longer sample.

## Detector Next Steps

The HSV detector is intentionally simple and deterministic for synthetic tests. For real tennis data, collect calibrated video, tune HSV thresholds with `scripts/calibrate_hsv.py`, add camera calibration, then add a neural detector adapter behind the existing detector interface without changing the no-hardware simulation default.
