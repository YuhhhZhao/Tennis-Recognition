# Tennis Ball Jetson Tracker

面向 `NVIDIA Jetson + 相机 + 小车 + 挡网垃圾桶` 的网球检测与跟踪示例工程。

核心策略：

```text
YOLO 全图检测：慢但稳，用于热启动、丢失重定位、周期校正
HSV ROI 跟踪：快但脆，用于连续毫秒级跟踪
Alpha-Beta Filter：估计速度并预测下一帧位置，抵消推理和控制延迟
```

## 目录结构

```text
tennis-ball-jetson-tracker/
  ball_tracker/
    async_yolo.py       # 后台 YOLO 推理线程
    config.py           # YAML 配置加载
    control.py          # 小车控制接口占位
    filters.py          # Alpha-Beta 位置/速度滤波
    hsv_tracker.py      # HSV + ROI + 轮廓筛选
    pipeline.py         # 主状态机
    state.py            # Detection / TrackState 数据结构
    yolo_detector.py    # Ultralytics YOLO 封装
  configs/
    tracker.yaml        # 默认参数
  scripts/
    run_tracker.py      # 运行入口
    calibrate_hsv.py    # HSV 阈值标定工具
    export_yolo_tensorrt.py
  tools/
    train_yolo.md       # 数据采集和 YOLO 微调说明
```

## 安装

建议先在普通电脑上调通，再部署到 Jetson。

```bash
cd /Users/zzzyhh123/projects/tennis-ball-jetson-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jetson 上如果已经装了系统版 OpenCV，可以先不要用 pip 覆盖 OpenCV。

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

## 小车控制接口

默认的 `ball_tracker/control.py` 只打印控制量。实际项目里可以替换成：

```text
串口 UART / CAN / ROS2 topic / UDP / GPIO PWM
```

建议控制程序接收的是“预测目标点”，不是当前检测点：

```text
predicted_center = current_center + velocity * control_latency
```

## HSV 标定

先用标定工具调阈值：

```bash
python scripts/calibrate_hsv.py --source 0
```

按 `q` 退出后，把显示出来的 HSV 范围写回 `configs/tracker.yaml`。

## 训练 YOLO

见 [tools/train_yolo.md](tools/train_yolo.md)。
