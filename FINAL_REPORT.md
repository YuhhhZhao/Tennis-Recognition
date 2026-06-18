# Final Integration Report

## What Changed

- Added the `tennis_robot_sim` package for a simulation-first closed loop.
- Added `configs/default_sim.yaml` with camera, detector, ball, court, robot, controller, trajectory, IMU, logging, and safety settings.
- Added safe CLI entry points:
  - `python -m tennis_robot_sim.run_sim`
  - `python -m tennis_robot_sim.tools.render_synthetic`
  - `python -m tennis_robot_sim.tools.replay`
  - `python -m tennis_robot_sim.run_video`
- Added environment and validation scripts:
  - `scripts/setup_env.sh`
  - `scripts/check_env.py`
  - `scripts/validate_sim.sh`
  - `scripts/jetson_monitor.sh`
- Added unit and end-to-end tests for simulation, detection, trajectory, robot control, IMU, localization, hardware safety, and smoke outputs.
- Added `README_simulation.md`, `docs/data_contracts.md`, and `environment_report.md`.

## Run

Smoke simulation:

```bash
python -m tennis_robot_sim.run_sim --scenario default --no-hardware --output outputs/smoke
```

Validation:

```bash
bash scripts/validate_sim.sh
```

Replay:

```bash
python -m tennis_robot_sim.tools.replay --input outputs/smoke --output outputs/replay_test
```

## Validation

Last validation run:

```text
bash scripts/validate_sim.sh
[1/4] Checking Python environment
[2/4] Running unit tests
33 passed in 3.14s
[3/4] Running smoke simulation
Predicted landing point: [3.239194614414162, 0.15758837210218282]
Robot final pose: [2.105818222578065, 0.09733465446314339, 0.05195332824925991]
[4/4] Replaying saved logs
Validation complete. Artifacts are under outputs/validation/
```

Smoke metrics from `outputs/validation/smoke/metrics.json`:

- Detection rate: `0.9888888888888889`
- Landing error: `8.049116928532385e-16 m`
- Hardware enabled: `false`
- Robot final distance to target: `0.114402097344796 m`
- Planner status: `unreachable before landing; best effort target selected`

## Artifacts

- `outputs/smoke/metrics.json`
- `outputs/smoke/trajectory.csv`
- `outputs/smoke/detections.csv`
- `outputs/smoke/tracks.csv`
- `outputs/smoke/robot_path.csv`
- `outputs/smoke/imu.csv`
- `outputs/smoke/commands.csv`
- `outputs/smoke/top_down.png`
- `outputs/smoke/overlay_last.png`
- `outputs/smoke/overlay.mp4`
- `outputs/validation/smoke/*`
- `outputs/validation/replay/top_down_replay.png`
- `outputs/synthetic/default.mp4`

## Limitations

- Synthetic simulation uses known 3D ball state as the depth observation once the detector finds the ball. Real video uses image radius depth estimation and needs camera calibration for accurate landing prediction.
- The robot model is a simple unicycle chassis with bounded speed, angular rate, and acceleration.
- The IMU localizer is a complementary filter. It reduces yaw drift from biased odometry in tests but does not estimate absolute position from acceleration.
- `nvidia-smi` failed in this Jetson environment, but the simulation path is CPU-only.
- `scipy` and `torch` are not installed; they are optional for the simulation path.

## Safety

- `run_sim` defaults to no hardware.
- `--hardware` is rejected unless `safety.enable_real_hardware=true`.
- `RealRobotInterface` refuses to instantiate without the same safety flag.
- Tests and validation do not open serial, GPIO, camera devices, or motor interfaces.

## Next Actions

- Add calibrated real-video examples and verify radius-based depth against measured court points.
- Implement a real transport behind `RealRobotInterface` only after bench testing with `dry_run: true`.
- Add a neural detector adapter for real tennis video while keeping `ColorBallDetector` as the CPU-only baseline.
- Extend planner/controller for the actual chassis geometry if it is not unicycle-like.
