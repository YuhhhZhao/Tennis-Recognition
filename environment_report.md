# Environment Report

Generated for the simulation-first goal in `goal.csv`.

## Git State

- Branch: `main`
- Remote: `origin https://github.com/YuhhhZhao/Tennis-Recognition.git`
- Latest commits:
  - `6b2a281 update firmware`
  - `13def89 Add 3D prediction unit tests and synthetic E2E tests`
  - `e3a56a7 Add YOLO training scripts and weights`
  - `1684717 reorganize: ball_tracker -> tennis_tracker`
  - `d10bcc9 完成了畸变的计算，根据测量结果对网球直径进行线性修正`
- `git pull --ff-only` was skipped because the worktree was not clean: `goal.csv` is currently untracked.
- Safety note: no destructive git commands were run.

## Runtime

- Python: `Python 3.10.20`
- pip: `pip 26.1.2` from `/home/nvidia/miniforge3/envs/tennis/lib/python3.10/site-packages/pip`
- conda: `conda 26.3.2`
- Platform: `Linux nvidia-desktop 5.15.148-tegra #1 SMP PREEMPT Thu Sep 18 15:08:33 PDT 2025 aarch64`
- CUDA/GPU: `nvidia-smi` is present but failed to communicate with the NVIDIA driver in this environment. The simulation path is CPU-only and does not require GPU access.
- Installed/verified core packages during this task: `pytest 9.1.0`, `matplotlib 3.10.9`, `numpy 2.2.6`, `PyYAML 6.0.3`, `opencv-python 4.13.0`.
- Optional packages not present in the active environment: `scipy`, `torch`.

## Project Inventory

- Existing package: `tennis_tracker/`
  - `detection/`: HSV, YOLO, async detector, filters
  - `prediction/`: monocular geometry, calibration, trajectory prediction
  - `control/`: controller and UART bridge
  - `pipeline.py`: camera/video runtime pipeline
- New simulation package: `tennis_robot_sim/`
  - `sim/`: projectile, scenarios, pinhole camera, synthetic renderer
  - `perception/`: detector and tracker
  - `estimation/`: geometry helpers and landing estimator
  - `robot/`: simulated robot, planner, controller, hardware interface guard
  - `imu/`: IMU simulator and complementary localizer
  - `run_sim.py`: no-hardware closed-loop runner
- Config files:
  - Existing real tracker config: `configs/app.yaml`
  - New simulation config: `configs/default_sim.yaml`
- Entry points:
  - Existing: `scripts/run_tracker.py`
  - New: `python -m tennis_robot_sim.run_sim`
  - New tools: `python -m tennis_robot_sim.tools.render_synthetic`, `python -m tennis_robot_sim.tools.replay`, `python -m tennis_robot_sim.run_video`

## Hardware Risk Notes

- Existing UART code lives in `tennis_tracker/control/uart_bridge.py` and can open serial only when the old app config enables UART.
- The new simulation runner defaults to no hardware and rejects `--hardware` unless `safety.enable_real_hardware=true`.
- Tests and validation scripts do not access serial, GPIO, camera devices, or motors.

## Validation Summary

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q`: `33 passed in 3.17s`
- `python -m tennis_robot_sim.run_sim --scenario default --no-hardware --output outputs/smoke`: completed and wrote metrics/logs/visualizations.
- `python -m tennis_robot_sim.tools.render_synthetic --frames 30 --output outputs/synthetic/default.mp4`: completed.
- `python -m tennis_robot_sim.tools.replay --input outputs/smoke --output outputs/replay_test`: completed.
- `bash scripts/validate_sim.sh`: completed with `33 passed in 3.14s`, smoke outputs under `outputs/validation/smoke`, replay plot under `outputs/validation/replay/top_down_replay.png`.
