# Tennis Ball Jetson Tracker

面向 `NVIDIA Jetson + 相机 + IMU + 麦轮小车` 的网球检测、3D 轨迹预测、IMU 定位与机器人拦截系统。

## Agent 接手说明

本节给后续 Codex/Claude/其他智能体快速接手用。先读这里, 再看后面的完整说明。

### 当前实机结论

- 工作目录: `/home/nvidia/Documents/tennis-recognition/Tennis-Recognition`
- Python 环境: `conda activate tennis`; 不要用 `base` 环境跑项目。
- 摄像头: DCXIN UVC, 通常是 `/dev/video0`。
- ESP32 小车控制: CH340, 当前正确端口是 `/dev/ttyCH341USB0`。
- `/dev/ttyACM0` 不是小车控制口; 当前枚举为 `NATIONS N32L40x`, 会输出类似 `UQ/UR/US/UT` 的二进制/乱码文本。
- YOLO 权重: `weights/best_v3.pt`。
- CUDA: Jetson `tennis` 环境中已验证 `torch.cuda.is_available() == True`, YOLO 命令使用 `--device 0`。

### 当前主线脚本

| 目标 | 脚本 | 是否会动小车 |
|------|------|--------------|
| 摄像头+YOLO安全预览 | `scripts/run_yolo_preview.py` | 否 |
| YOLO落点预测+ESP32联调 | `scripts/run_yolo_catch_control.py` | 只有加 `--enable-control` 才会 |
| ESP32串口交互测试 | `scripts/test_uart.py --port /dev/ttyCH341USB0 --mode interactive` | 会, 取决于输入命令 |
| 完整旧管线 | `scripts/run_tracker.py` | 可能会, 取决于配置 |

### 现在推荐的真实联调命令

```bash
cd /home/nvidia/Documents/tennis-recognition/Tennis-Recognition
conda activate tennis

python scripts/run_yolo_catch_control.py \
  --source 0 \
  --model weights/best_v3.pt \
  --device 0 \
  --conf 0.25 \
  --imgsz 640 \
  --trigger-speed 0.6 \
  --trigger-delta 0.35 \
  --active-max-missing 8 \
  --rearm-grace 0.25 \
  --rearm-cooldown 0.8 \
  --min-arrival-time 0.08 \
  --uart-handshake-timeout 0.2 \
  --enable-control \
  --uart-port /dev/ttyCH341USB0
```

先不动小车时, 去掉最后两行 `--enable-control` 和 `--uart-port ...`。

### 已改过的关键逻辑

- `tennis_tracker/prediction/geometry.py`
  - `CameraPose` 已支持完整三维相机光心偏移。
  - 当前相机光心相对机器人原点: `x=-0.08`, `y=+0.10`, `z=+0.08`。
  - 转换逻辑是 `robot_point = R * camera_point + camera_center_offset`。
- `configs/app.yaml`
  - `geometry.camera_offset_x_m/y_m/z_m` 已写入上述偏移。
  - `uart.port` 是 `/dev/ttyCH341USB0`。
- `tennis_tracker/control/uart_bridge.py`
  - UART 打开过程会打印 `opening/port opened/waiting RDY`。
  - 读取串口时忽略非 ASCII 坏字节, 避免 IMU/启动乱码导致崩溃。
  - 支持 `handshake_timeout_s`; 联调脚本参数为 `--uart-handshake-timeout`。
- `scripts/run_yolo_catch_control.py`
  - YOLO-only, 不走 HSV。
  - 速度突变后才进入 active。
  - active 后连续丢失可用 3D 点会 `STOP` 并 re-arm。
  - 预测落点窗口结束会 `STOP` 并 re-arm。
  - re-arm 后有 cooldown, 防止同一个球反复触发。
  - `t_arrival < --min-arrival-time` 会被认为太晚, 直接安全停止。
- `tennis_tracker/detection/hsv_tracker.py`
  - HSV 曾做过更严格的形状/半径/边界约束, 但当前接球联调主线建议用 YOLO-only。

### tmux/日志排查经验

- `tmux1` 之前出现的大量 `UTUURUS...` 不是 Python 异常, 是把 `/dev/ttyACM0` 这类非小车文本协议串口当 ESP32 串口读了。
- `KeyboardInterrupt` 通常是用户按了 `Ctrl+C`, 不是程序本身崩溃。
- 看到 `[LAND#...] ... SENT safe` 说明上位机确实发送了 `TARGET`。
- 看到 `[UART] OK` 或 `[UART] STOP_OK STEPS=...` 说明下位机确实收到了命令。
- 看到 `[SAFE] landing too soon ...` 说明预测已太晚, 安全逻辑主动 STOP/re-arm。

### 已知问题和下一步

- 近距离约 `1m` 直上直下抛球时, 球落地窗口很短, 小车经常只能短距离动一下。
- YOLO 框半径反推单目深度仍会抖, 日志里可能出现 `x/y/z` 跳变和假速度突变。
- 下一步优先级:
  1. ~~在 `run_yolo_catch_control.py` 增加 3D 离群点过滤~~ ✅ 已添加 `--min-z/--max-x/--max-y/--max-3d-speed`
  2. ~~加可达性判断~~ ✅ 已添加 `--reachable-max-speed`，不可达时打印 `unreachable` 不发 TARGET
  3. 如果要让小车稳定追落点, 固件侧 `TARGET` 需要改成真正的位置闭环目标保持, 或上位机改成持续发送规划速度 `VEL`。
  4. ~~降低触发阈值方便测试~~ ✅ 默认值已调低: trigger-speed=0.6, trigger-delta=0.35, min-samples=4
  5. ~~landing too soon 不应立即 rearm~~ ✅ active 模式下允许 Kalman 继续收敛，连续 5 帧太短才 rearm

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
| **YOLO落点联调** | ⚠️ 开发中 | YOLO-only 脚本已接入速度突变触发、落点输出、UART下发和安全re-arm |
| **轨迹预测+接球** | ⚠️ 开发中 | 直上直下近距离抛球窗口过短, 需继续调深度/可达性过滤 |

**待完成 (Codex):**
1. 给 YOLO-only 联调增加更严格的 3D 离群点过滤和可达性判断
2. 调整抛球测试方式: 优先用更高/更慢的抛球验证落点预测, 再做真实接球
3. 如果要让小车明显追落点, 需要把 `TARGET` 语义改成更稳定的位置闭环或延长控制窗口

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

当前实机枚举结论:

```text
/dev/ttyCH341USB0  VID:PID=1a86:7523  QinHeng CH340  → ESP32 小车控制
/dev/ttyACM0       VID:PID=19f5:5740  NATIONS N32L40x → 不是小车控制口
```

不要把 `--uart-port` 指向 `/dev/ttyACM0`。该口会输出类似 `UQ/UR/US/UT` 的二进制/乱码数据, 不是小车协议文本。

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
    run_yolo_preview.py     # YOLO-only 安全预览, 不启用UART/IMU/下位机
    run_yolo_catch_control.py # YOLO-only 落点预测+ESP32联调入口
    test_car_move.py        # 小车运动测试
    test_uart.py            # ESP32 UART 交互/自动/监视工具
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
cd /home/nvidia/Documents/tennis-recognition/Tennis-Recognition
conda activate tennis
pip install -r requirements.txt

# Jetson GPU PyTorch (系统预装)
# 在 conda 环境中添加系统 torch 路径:
echo "/usr/local/lib/python3.10/dist-packages" > $(python -c "import site; print(site.getsitepackages()[0])")/system-torch.pth
```

> Jetson 系统预装了 NVIDIA GPU 版 PyTorch (`/usr/local/lib/python3.10/dist-packages/`)。
> conda 环境需降级 numpy 以兼容: `pip install 'numpy<2'`

检查当前环境是否能启动项目：

```bash
conda activate tennis
python scripts/check_env.py
```

当前仓库已验证 `tennis` 环境可用；不要直接用 `base` 环境运行，`base` 里可能缺少 `cv2/numpy/yaml/matplotlib`。

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

### 启动前确认硬件

```bash
cd /home/nvidia/Documents/tennis-recognition/Tennis-Recognition
conda activate tennis

# 查看 Jetson 是否识别到摄像头、ESP32、IMU
ls -l /dev/video* /dev/ttyCH341USB0 /dev/ttyACM0
v4l2-ctl --list-devices
```

如果看不到 `/dev/video0`，说明当前系统没有枚举到摄像头。先检查 USB 线、供电、摄像头是否插在 Jetson 上，再重新插拔后执行：

```bash
dmesg | tail -50
v4l2-ctl --list-devices
```

### 完整管线

```bash
python scripts/run_tracker.py --config configs/app.yaml --source 0 --no-yolo
```

- `--source 0` = 相机 `/dev/video0`
- `--no-yolo` = 纯 HSV 检测 (更快)
- 不加 `--no-yolo` = YOLO + HSV 混合
- 按 `q` 退出

这条命令会打开 OpenCV 窗口 `tennis-ball-tracker`，窗口里会实时显示摄像头画面，并在识别到网球时画出圆圈、预测点、3D 位置和落点信息。

注意：`configs/app.yaml` 里 `control.enabled: true` 且 `uart.enabled: true`。如果 ESP32 已连接，完整管线可能向小车发送 `TARGET` 指令。只想先看摄像头和球的位置时，用下面的“只看画面，不控制小车”方式。

### 只看画面，不控制小车

先生成一份临时配置，把小车控制和 UART 关掉：

```bash
python - <<'PY'
from pathlib import Path
import yaml

src = Path("configs/app.yaml")
dst = Path("/tmp/tennis-view-only.yaml")
data = yaml.safe_load(src.read_text())
data["control"]["enabled"] = False
data["uart"]["enabled"] = False
dst.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(dst)
PY
```

启动实时预览和网球检测：

```bash
python scripts/run_tracker.py --config /tmp/tennis-view-only.yaml --source 0 --no-yolo
```

如果需要启用 YOLO + HSV 混合检测：

```bash
python scripts/run_tracker.py --config /tmp/tennis-view-only.yaml --source 0
```

### Linux 桌面端查看 Jetson 摄像头画面

摄像头接在 Jetson 上时，画面窗口必须显示在 Jetson 的图形桌面里。常用方式有三种：

1. 直接在 Jetson 桌面上打开终端运行：

```bash
cheese
```

或用 GStreamer 低延迟预览：

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  video/x-raw,width=640,height=360,framerate=30/1 ! \
  videoconvert ! autovideosink sync=false
```

2. 从 Linux 桌面 SSH 到 Jetson，并把窗口转发回本机：

```bash
ssh -X nvidia@<jetson-ip>
cd /home/nvidia/Documents/tennis-recognition/Tennis-Recognition
conda activate tennis
python scripts/run_tracker.py --config /tmp/tennis-view-only.yaml --source 0 --no-yolo
```

如果 `/tmp/tennis-view-only.yaml` 不存在，先执行上面“只看画面，不控制小车”里的临时配置生成命令。

如果 `ssh -X` 显示很卡，可以用 `ssh -Y`，或者改用 NoMachine/VNC 连接 Jetson 桌面后在远程桌面里运行程序。

3. 只验证摄像头是否工作，不跑识别程序：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
cheese
```

只要 `cheese` 或 GStreamer 能看到实时画面，`run_tracker.py --source 0` 就应该能读到同一个摄像头。识别窗口里出现绿色圆圈时，说明网球已经被映射到摄像头画面坐标；顶部状态栏会显示当前检测源、置信度、半径、3D 坐标和落点预测。

### HSV 阈值标定

如果窗口有画面但识别不到球，先用 HSV 标定工具调阈值：

```bash
python scripts/calibrate_hsv.py --source 0
```

拖动滑条直到画面里只剩网球区域，按 `q` 退出，终端会打印 `lower` 和 `upper`。把这两个值填回 `configs/app.yaml` 的 `hsv.lower` 和 `hsv.upper`。

### 仅测试小车运动

```bash
python scripts/test_car_move.py
```

更推荐先用交互式串口工具确认 ESP32 控制口:

```bash
python scripts/test_uart.py --port /dev/ttyCH341USB0 --mode interactive
```

进入后依次输入:

```text
PING
STOP
VEL 0.2 0 0
STOP
TARGET 0.5 0.0 2.0
STOP
```

如果返回 `OK` / `STOP_OK STEPS=...`, 且小车能前进/停止, 说明下位机通信与底盘运动基本可用。

### YOLO-only 安全预览

只看 YOLO 检测、3D 坐标和落点预测, 不初始化 UART/IMU/下位机:

```bash
python scripts/run_yolo_preview.py \
  --source 0 \
  --model weights/best_v3.pt \
  --device 0 \
  --conf 0.25 \
  --imgsz 640
```

### YOLO-only 接球联调

先 dry-run, 只看终端输出和画面, 不让小车动:

```bash
python scripts/run_yolo_catch_control.py \
  --source 0 \
  --model weights/best_v3.pt \
  --device 0 \
  --conf 0.25 \
  --imgsz 640 \
  --trigger-speed 0.6 \
  --trigger-delta 0.35 \
  --active-max-missing 8 \
  --rearm-grace 0.25 \
  --rearm-cooldown 0.8 \
  --min-arrival-time 0.08 \
  --uart-handshake-timeout 0.2
```

确认 dry-run 的 `[LAND#...]` 落点稳定后, 再启用小车控制:

```bash
python scripts/run_yolo_catch_control.py \
  --source 0 \
  --model weights/best_v3.pt \
  --device 0 \
  --conf 0.25 \
  --imgsz 640 \
  --trigger-speed 0.6 \
  --trigger-delta 0.35 \
  --active-max-missing 8 \
  --rearm-grace 0.25 \
  --rearm-cooldown 0.8 \
  --min-arrival-time 0.08 \
  --uart-handshake-timeout 0.2 \
  --enable-control \
  --uart-port /dev/ttyCH341USB0
```

输出含义:

```text
[WAIT]     已检测到球, 但还没满足速度突变触发
[TRIGGER] 速度突变触发, 开始收集3D轨迹并预测落点
[LAND#]   当前预测落点, SENT=已发TARGET, HELD=发送间隔未到, DRY=未启用控制
[LATE]    t_arrival 太短 (Kalman 可能还在收敛), 不发送 TARGET 但继续跟踪
[SAFE]    安全保护触发, 发送STOP并重新等待下一次触发
[COOLDOWN] re-arm 后冷却中, 暂时忽略新触发
```

当前安全保护:

- active 后连续 `--active-max-missing` 帧没有可用 3D 点, 发送 `STOP` 并 re-arm。
- 预测落点到达窗口结束后 `--rearm-grace` 秒, 发送 `STOP` 并 re-arm。
- re-arm 后 `--rearm-cooldown` 秒内忽略新触发, 避免同一个球反复触发。
- `t_arrival < --min-arrival-time` 的落点会被认为太晚, 直接停止并 re-arm。

已知现象:

- 近距离 `1m` 直上直下抛球时, 落地时间很短, 系统常输出 `landing too soon`。这是保护逻辑在阻止过晚预测继续驱动车。
- YOLO 框半径用于单目深度估计, 当框抖动时会出现 `x/y/z` 跳变和假速度突变。后续需要继续加入 3D 离群点过滤。
- ESP32 当前固件里的 `TARGET x y t` 会直接转换成 `vx=x/t, vy=y/t`, 再由上位机的 `STOP` 停车；如果 `t` 很小, 小车只能短距离移动一下。

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

当前 `mecanum_controller.ino` 对 `TARGET` 的处理是速度式:

```text
vx = clamp(x / t, ±1.5 m/s)
vy = clamp(y / t, ±1.0 m/s)
```

它不是完整的位置闭环目标保持。上位机频繁更新 `TARGET` 并在安全条件触发时发送 `STOP`, 因此接球联调中如果预测 `t_arrival` 太短, 小车只会出现很短的前进或横移。

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

当前真实相机外参偏移配置在 `configs/app.yaml`:

```yaml
geometry:
  camera_offset_x_m: -0.08   # 机器人前方为正
  camera_offset_y_m: 0.10    # 机器人左方为正
  camera_offset_z_m: 0.08    # 机器人上方为正
```

`camera_to_robot()` 使用完整平移:

```text
robot_point = R * camera_point + camera_center_offset
```

## 训练 YOLO

见 [tools/train_yolo.md](tools/train_yolo.md)。

## Wiki

- [README_simulation.md](README_simulation.md) — 仿真系统说明
- [docs/data_contracts.md](docs/data_contracts.md) — 数据结构与坐标约定
- [environment_report.md](environment_report.md) — 环境配置报告
- [FINAL_REPORT.md](FINAL_REPORT.md) — 集成报告
