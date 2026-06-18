# Simulation Data Contracts

All coordinates use meters, seconds, radians, and pixels unless otherwise stated.

## Frames

- Image frame: `(u_px, v_px)`, origin top-left, `u` right, `v` down.
- Camera frame: OpenCV pinhole frame, `X` right, `Y` down, `Z` forward.
- World/court frame: `x_m` forward, `y_m` left, `z_m` up, ground plane `z_m = 0`.
- Robot body frame: `x` forward, `y` left, yaw positive counter-clockwise.
- IMU frame: body-aligned; gyro `z` is yaw rate.

## Dataclasses

`Detection`

| Field | Units | Meaning |
| --- | --- | --- |
| `frame_id` | index | Video/simulation frame number |
| `timestamp` | s | Frame time |
| `center_px` | px | `(u, v)` detected ball center |
| `radius_px` | px | Detected image radius |
| `confidence` | 0-1 | Detector confidence |
| `source` | text | Detector source, e.g. `color` |

`TrackState`

| Field | Units | Meaning |
| --- | --- | --- |
| `frame_id` | index | Last processed frame |
| `timestamp` | s | Last update time |
| `center_px` | px | Smoothed center or empty when lost |
| `velocity_px_s` | px/s | Smoothed image-plane velocity |
| `radius_px` | px | Smoothed radius |
| `confidence` | 0-1 | Track confidence |
| `missing_frames` | frames | Consecutive misses |

`LandingPrediction`

| Field | Units | Meaning |
| --- | --- | --- |
| `landing_xy` | m | Predicted ground-plane point `(x, y)` |
| `time_to_land` | s | Time from current frame to ground contact |
| `uncertainty` | m | Mean trajectory residual |
| `confidence` | 0-1 | Confidence derived from residual |
| `debug_metrics` | mixed | Fit details such as samples and velocity |

`RobotState`

| Field | Units | Meaning |
| --- | --- | --- |
| `x` | m | World x |
| `y` | m | World y |
| `yaw` | rad | Heading |
| `v` | m/s | Linear velocity |
| `omega` | rad/s | Angular velocity |
| `timestamp` | s | State time |

`ControlCommand`

| Field | Units | Meaning |
| --- | --- | --- |
| `v` | m/s | Commanded linear speed |
| `omega` | rad/s | Commanded yaw rate |

`InterceptionPlan`

| Field | Units | Meaning |
| --- | --- | --- |
| `target_pose` | m/rad | Robot target pose |
| `reachable` | bool | Whether target is reachable before landing |
| `eta` | s | Minimum travel time estimate |
| `distance` | m | Distance to requested target |
| `reason` | text | Planner status |

`IMUSample`

| Field | Units | Meaning |
| --- | --- | --- |
| `timestamp` | s | Sensor sample time |
| `accel_mps2` | m/s^2 | Body-frame acceleration `(x, y, z)` |
| `gyro_radps` | rad/s | Body-frame angular velocity `(x, y, z)` |
| `yaw_rate_radps` | rad/s | Convenience copy of gyro z |
| `ground_truth_pose` | state | Optional simulated truth pose |

## Log Columns

`trajectory.csv`

`timestamp`, `ball_x_m`, `ball_y_m`, `ball_z_m`, `obs_x_m`, `obs_y_m`, `obs_z_m`, `pred_landing_x_m`, `pred_landing_y_m`, `pred_time_to_land_s`

`detections.csv`

`frame_id`, `timestamp`, `center_x_px`, `center_y_px`, `radius_px`, `confidence`, `source`

`tracks.csv`

`frame_id`, `timestamp`, `center_x_px`, `center_y_px`, `velocity_x_px_s`, `velocity_y_px_s`, `radius_px`, `confidence`, `missing_frames`

`robot_path.csv`

`timestamp`, `truth_x_m`, `truth_y_m`, `truth_yaw_rad`, `truth_v_mps`, `truth_omega_radps`, `est_x_m`, `est_y_m`, `est_yaw_rad`

`imu.csv`

`timestamp`, `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`, `gyro_x_radps`, `gyro_y_radps`, `gyro_z_radps`, `yaw_rate_radps`

`commands.csv`

`timestamp`, `v_mps`, `omega_radps`, `target_x_m`, `target_y_m`, `reachable`, `reason`

`metrics.json`

Contains scenario metadata, detection rate, truth and predicted landing point, landing error, final robot pose, target pose, planner reachability, and artifact paths.

