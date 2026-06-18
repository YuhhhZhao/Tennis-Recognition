# Tennis Ball Jetson Tracker

面向 `NVIDIA Jetson + 相机 + IMU + 麦轮小车` 的网球检测、3D 轨迹预测、IMU 定位与机器人拦截系统。

## 当前进度

| 模块 | 状态 | 说明 |
|------|------|------|
| 相机+YOLO检测 | ✅ 已测试 | GPU 推理 30FPS, best_v3.pt, 89%置信度 |
| 3D单目定位 | ✅ 已测试 | 静态检测准确, 方向正确 |
| ESP32小车控制 | ✅ 已测试 | VEL/TARGET/STOP 正常, 带符号步数里程计 |
| IMU驱动 | ✅ 已测试 | WitMotion 1000Hz, 加速度/角速度/姿态角 |
| 里程计追踪 | ✅ 已测试 | 步数→正解算→世界坐标, 直线0.5m误差<1cm |
| IMU+里程计融合 | ✅ 已测试 | 互补滤波, 转向误差<6° |
| 静态截击 | ✅ 已测试 | 检测球位置→小车移动到目标, 误差2cm |
| 运动学校准 | ✅ 已完成 | L_SUM=0.173 (IMU标定, 物理实测0.193) |
| **轨迹预测+接球** | ⚠️ 开发中 | 抛球检测+抛物线落点预测待调试 |

**待完成 (Codex):**
1. `test_catch.py` 抛球检测调通 — 速度阈值触发、滤波器重置、落点预测
2. 接球联调 — 手持球→抛出→小车移动到预测落点
3. 端到端测试 — 完整网球拦截管线

## 硬件连接

```
Jetson USB-A
  ├── DCXIN Camera    → /dev/video0        (UVC 相机, 640×360)
  ├── ESP32 (CH340)   → /dev/ttyCH341USB0  (小车控制, 115200)
  └── WitMotion IMU   → /dev/ttyACM0       (IMU姿态, 115200)
```

| 设备 | USB VID:PID | 设备节点 | 协议 |
|------|-------------|----------|------|
| 相机 DCXIN | `1bcf:2d50` | `/dev/video0` | UVC |
| ESP32 小车 | `1a86:7523` | `/dev/ttyCH341USB0` | UART 115200 |
| WitMotion IMU | `19f5:5740` | `/dev/ttyACM0` | USB CDC ACM 115200 |

### CH340 驱动问题 (Jetson)

Jetson 上 `brltty` 服务会抢占 CH340 串口设备。解决：

```bash
sudo systemctl mask brltty --now
sudo systemctl mask brltty-udev.service
```

CH340 变体 `1a86:7523` 不被内核 ch341 驱动识别，需写入 `new_id`：

```bash
sudo sh -c 'echo "1a86 7523" > /sys/bus/usb/drivers/usb_ch341/new_id'
```

项目包含 WCH 官方编译好的驱动 `CH341SER_LINUX/driver/ch341.ko`，可加载：

```bash
sudo rmmod ch341
sudo insmod CH341SER_LINUX/driver/ch341.ko
```

### IMU 权限

```bash
sudo chmod 666 /dev/ttyCH341USB0 /dev/ttyACM0
```

## 目录结构

```text
tennis-recognition/
  tennis_tracker/
    detection/              # 2D 识别 (HSV + YOLO + 滤波)
    prediction/             # 3D 落点预测 (单目几何 + Kalman)
    control/                # 电控 + 通信 (CarController + UART)
    pipeline.py             # 主状态机
    state.py                # 全部数据结构
    config.py               # YAML 配置加载
  tennis_robot_sim/
    estimation/
      odometry.py           # 麦轮里程计 (步数→位移正解算)
      trajectory.py         # 落点预测
      geometry.py           # 图像/世界坐标转换
    imu/
      witmotion.py          # WitMotion 真实 IMU 驱动
      localization.py       # 互补滤波器 (IMU+里程计融合)
      sim_imu.py            # 仿真 IMU
    robot/                  # 运动学/规划/控制
    sim/                    # 仿真环境 (球/相机/场景)
  configs/
    app.yaml                # 真实硬件配置
    default_sim.yaml        # 仿真配置
  firmware/
    mecanum_controller/     # ESP32 麦轮控制固件
  scripts/
    run_tracker.py          # 完整跟踪管线入口
    test_car_move.py        # 小车运动测试
    test_imu.py             # IMU 数据读取测试
    test_imu_localization.py # IMU+定位器集成测试
    test_imu_vs_odometry.py  # IMU vs 里程计对比测试
    diag_serial.py          # 串口诊断工具
    calibrate_hsv.py        # HSV 阈值标定
    calibrate_camera.py     # 棋盘格相机标定
  weights/                  # YOLO 模型权重
    best_v3.pt              # 推荐使用
    best.pt
    yolov8n.pt
```

## 安装

### Conda 环境 (推荐)

```bash
conda activate tennis
pip install -r requirements.txt

# Jetson GPU PyTorch (系统预装)
# 在 conda 环境中添加系统 torch 路径:
echo "/usr/local/lib/python3.10/dist-packages" > $(python -c "import site; print(site.getsitepackages()[0])")/system-torch.pth
```

> Jetson 系统预装了 NVIDIA GPU 版 PyTorch (`/usr/local/lib/python3.10/dist-packages/`)。
> conda 环境需降级 numpy 以兼容: `pip install 'numpy<2'`

## 核心策略

```text
YOLO 全图检测：慢但稳，用于热启动、丢失重定位、周期校正
HSV ROI 跟踪：快但脆，用于连续毫秒级跟踪
Alpha-Beta Filter：估计速度并预测下一帧位置，抵消推理和控制延迟
单目 3D 估计：利用已知网球直径 (6.7cm) 估算深度 → 机器人坐标系
Kalman + 重力模型：6D 状态滤波 → 抛物线落点预测
IMU 互补滤波：陀螺仪修正里程计航向漂移
麦轮里程计：ESP32 步数 → 正解算 → 车体位移 (x, y, yaw)
```

## 运行

### 完整管线

```bash
python scripts/run_tracker.py --config configs/app.yaml --source 0 --no-yolo
```

- `--source 0` = 相机 `/dev/video0`
- `--no-yolo` = 纯 HSV 检测 (更快)
- 不加 `--no-yolo` = YOLO + HSV 混合
- 按 `q` 退出

### 仅测试小车运动

```bash
python scripts/test_car_move.py
```

### 测试 IMU

```bash
python scripts/test_imu.py                          # 实时显示姿态
python scripts/test_imu_localization.py              # IMU+定位器集成
python scripts/test_imu_vs_odometry.py               # IMU vs 里程计对比
```

### 串口诊断

```bash
python scripts/diag_serial.py
```

## 小车控制协议

ESP32 固件 `mecanum_controller.ino` (v3.1):

```
Jetson → ESP32:   VEL <vx> <vy> <w>\n     (速度指令 m/s, rad/s)
                   TARGET <x> <y> <t>\n     (落点指令 m, s)
                   STOP\n                   (紧急停车)
                   PING\n                   (查询步数)
                   STAT\n                   (诊断状态)
                   RESET_STEPS\n            (步数归零)

ESP32 → Jetson:   OK\n
                   PONG s=<s0>,<s1>,<s2>,<s3>\n  (四轮带符号步数)
                   STOP_OK STEPS=<s0>,<s1>,<s2>,<s3>\n
```

### 里程计

步数 → 位移在 Jetson 端 (`tennis_robot_sim/estimation/odometry.py`) 计算：

```
步数增量 × (2πR / 1600) = 轮位移 (m)
四轮位移 → 麦轮正解算 → 车体位移 (dx, dy, dyaw)
→ 旋转到世界坐标系 → 累积 (x, y, yaw)
```

### IMU 定位

WitMotion IMU (`tennis_robot_sim/imu/witmotion.py`) 持续输出：

| 数据类型 | 更新率 | 用途 |
|----------|--------|------|
| 角速度 (gyro Z) | ~1000 Hz | 航向修正 |
| 加速度 | ~1000 Hz | 静止检测 |
| 姿态角 (roll/pitch/yaw) | ~1000 Hz | 辅助参考 |

互补滤波器 (`ComplementaryLocalizer`):

```
融合航向 = 0.92 × IMU积分航向 + 0.08 × 里程计航向
```

## 固件烧录

用 Arduino IDE 打开 `firmware/mecanum_controller/mecanum_controller.ino`：

1. 选择开发板: ESP32 Dev Module
2. 端口: 对应 CH340 串口
3. 点击上传

## 3D 落点预测

```
Camera Frame
    ↓
[HSV/YOLO Detection] → (u, v, radius_px)
    ↓
[pixel_to_camera_frame] → (Xc, Yc, Zc) ← depth_from_ball_radius
    ↓
[camera_to_robot] → (Xr, Yr, Zr) ← CameraPose
    ↓
[TrajectoryFilter] → 6D Kalman 带重力模型
    ↓
[BallisticSolver] → 落点 (x, y, t_arrival)
    ↓
[CarController] → TARGET 指令 → ESP32
```

## 训练 YOLO

见 [tools/train_yolo.md](tools/train_yolo.md)。

## Wiki

- [README_simulation.md](README_simulation.md) — 仿真系统说明
- [docs/data_contracts.md](docs/data_contracts.md) — 数据结构与坐标约定
- [environment_report.md](environment_report.md) — 环境配置报告
- [FINAL_REPORT.md](FINAL_REPORT.md) — 集成报告
