# Tennis Ball Jetson Tracker

面向 `NVIDIA Jetson + 相机 + 小车 + 挡网垃圾桶` 的网球检测、3D 轨迹预测与跟踪示例工程。

核心策略：

```text
YOLO 全图检测：慢但稳，用于热启动、丢失重定位、周期校正
HSV ROI 跟踪：快但脆，用于连续毫秒级跟踪
Alpha-Beta Filter：估计速度并预测下一帧位置，抵消推理和控制延迟
单目 3D 估计：利用已知网球直径 (6.7cm) 估算深度 → 机器人坐标系
Kalman + 重力模型：6D 状态滤波 → 抛物线落点预测
```

## 目录结构

```text
tennis-ball-jetson-tracker/
  ball_tracker/
    prediction/             # ★ 3D 落点预测子包
      __init__.py           #     公开 API 导出
      geometry.py           #     单目 3D 几何 (像素→相机→机器人)
      calibration.py        #     相机标定 (棋盘格) 加载/保存
      trajectory.py         #     3D Kalman 轨迹滤波 + 抛物线落点求解
    async_yolo.py           # 后台 YOLO 推理线程
    config.py               # YAML 配置加载
    control.py              # 小车控制接口 (支持 2D 降级 + 3D 落点指令)
    filters.py              # Alpha-Beta 位置/速度滤波
    hsv_tracker.py          # HSV + ROI + 轮廓筛选
    pipeline.py             # 主状态机 (2D → 3D → 轨迹 → 落点 → 控制)
    state.py                # Detection / Detection3D / LandingPoint / TrackState
    yolo_detector.py        # Ultralytics YOLO 封装
  configs/
    tracker.yaml            # 默认参数 (含 camera / geometry / trajectory)
  scripts/
    run_tracker.py          # 运行入口
    calibrate_hsv.py        # HSV 阈值标定工具
    calibrate_camera.py     # ★ 棋盘格相机标定工具
    export_yolo_tensorrt.py
  tools/
    train_yolo.md           # 数据采集和 YOLO 微调说明
```

## 安装

建议先在普通电脑上调通，再部署到 Jetson。

```bash
cd /Users/zzzyhh123/projects/tennis-ball-jetson-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jetson 上如果已经装了系统版 OpenCV，先不要用 pip 覆盖 OpenCV。

## 运行

使用摄像头：

```bash
python scripts/run_tracker.py --config configs/tracker.yaml --source 0
```

只测 HSV，不加载 YOLO：

```bash
python scripts/run_tracker.py --config configs/tracker.yaml --source 0 --no-yolo
```

使用视频文件：

```bash
python scripts/run_tracker.py --config configs/tracker.yaml --source data/samples/test.mp4
```

如果你已经有 YOLO 权重，把 `configs/tracker.yaml` 里的路径改掉：

```yaml
yolo:
  model_path: "weights/best.engine"
```

也可以临时指定：

```bash
python scripts/run_tracker.py --source 0 --yolo-model weights/best.engine
```

Jetson 推荐使用 TensorRT engine：

```text
best.pt -> best.onnx -> best.engine
```

也可以使用 Ultralytics 直接导出：

```bash
python scripts/export_yolo_tensorrt.py --weights weights/best.pt --imgsz 640
```

## 工作逻辑

1. 启动时没有目标，主循环把当前帧交给 YOLO worker。
2. YOLO 找到网球后，得到初始 bbox、中心点、半径估计。
3. HSV tracker 只在预测位置附近裁剪 ROI，做颜色阈值、形态学、轮廓筛选。
4. HSV 有效时高速更新目标状态，并发送控制指令。
5. HSV 失败或置信度下降时，触发 YOLO 全图重定位。
6. YOLO 也可以按固定间隔做低频校正，防止 HSV 漂移。
7. **★ 3D 管线**: 2D 检测 → 单目深度估计 → 机器人坐标系 → Kalman 滤波 → 落点预测 → 控制指令。

## 3D 落点预测

### 原理

```
Camera Frame
    ↓
[HSV/YOLO Detection] → (u, v, radius_px)
    ↓
[pixel_to_camera_frame] → (Xc, Yc, Zc) ← depth_from_ball_radius (已知网球直径 6.7cm)
    ↓
[camera_to_robot] → (Xr, Yr, Zr) ← CameraPose (相机高度 + 俯仰角)
    ↓
[TrajectoryFilter] → 6D Kalman (x, y, z, vx, vy, vz) 带重力模型
    ↓
[BallisticSolver] → 解 z(t) == target_height → 落点 (x, y, z, t_arrival)
    ↓
[CarController] → turn, forward 指令
```

### 使能 3D 预测

**Step 1 — 相机标定**（只需一次）

```bash
# 生成棋盘格 PNG，A4 打印贴在平板上
python -c "from ball_tracker.prediction import generate_chessboard_png; generate_chessboard_png('chessboard.png')"

# 运行交互式标定
python scripts/calibrate_camera.py --source 1 --pattern 9x6 --square 0.025
```

- 将棋盘格在相机视野内改变位置/角度
- 看到彩色 overlay 时按 **空格** 保存
- 采集 15-30 张后按 **q** 退出
- 结果保存在 `configs/calibration.npz`

**Step 2 — 配置相机位姿**

在 `configs/tracker.yaml` 的 `geometry` 段填入真实值：

```yaml
geometry:
  calibration_path: "configs/calibration.npz"
  ball_diameter_m: 0.067      # 标准网球
  camera_height_m: 0.30       # 相机距地面高度 (米)
  camera_pitch_deg: 20.0      # 俯仰角 (正值 = 向下)
  camera_yaw_deg: 0.0         # 偏航角, 0 = 正前方
```

**Step 3 — 运行**

```bash
python scripts/run_tracker.py --source 1 --no-yolo
```

界面左上角会显示 3D 位置和落点预测信息。无标定文件时 3D 模块自动降级为 no-op，2D 跟踪不受影响。

### 配置项说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trajectory.gravity` | -9.81 | 重力加速度 (m/s²) |
| `trajectory.target_height_m` | 0.0 | 目标接球高度 (0=地面) |
| `trajectory.min_samples_for_fit` | 6 | 至少收集 6 帧才开始落点预测 |
| `trajectory.min_prediction_confidence` | 0.3 | 置信度阈值 |

## HSV 标定

先用标定工具调阈值：

```bash
python scripts/calibrate_hsv.py --source 0
```

按 `q` 退出后，把显示出来的 HSV 范围写回 `configs/tracker.yaml`。

## 小车控制接口

默认的 `ball_tracker/control.py` 支持两种模式：

- **3D 落点指令**（`send_landing`）: 基于预测的 3D 落点 (x, y, z) 生成 turn/forward
- **2D 像素指令**（`send_target`）: 3D 不可用时降级为 2D 图像目标

实际项目里可以替换为：

```text
串口 UART / CAN / ROS2 topic / UDP / GPIO PWM
```

建议控制程序接收的是"预测目标点"，不是当前检测点：

```text
predicted_center = current_center + velocity * control_latency
```

## 训练 YOLO

见 [tools/train_yolo.md](tools/train_yolo.md)。
