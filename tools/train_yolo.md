# YOLO Training Notes

## 数据采集

建议先录视频，再抽帧标注：

```bash
ffmpeg -i tennis_video.mp4 -vf fps=5 dataset/images/frame_%05d.jpg
```

训练初版可以从 200-500 张开始。想要更稳，建议 1000-3000 张，覆盖：

- 不同距离：近、中、远
- 不同位置：画面中心、边缘、网兜附近
- 不同状态：静止、滚动、飞行、运动模糊
- 不同环境：地面、墙面、人、小车结构、阴影
- 遮挡情况：被网、桶、车体部分遮挡

## 标注格式

推荐使用 YOLO 格式：

```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

如果只训练一个类别，类别名可以用：

```yaml
names:
  0: tennis ball
```

## 训练示例

```bash
yolo detect train \
  model=yolov8n.pt \
  data=dataset/tennis_ball.yaml \
  imgsz=640 \
  epochs=80 \
  batch=16 \
  device=0
```

输出权重一般在：

```text
runs/detect/train/weights/best.pt
```

把它复制到项目：

```bash
mkdir -p weights
cp runs/detect/train/weights/best.pt weights/best.pt
```

## Jetson 部署建议

优先选择小模型：

```text
yolov8n / yolov11n / yolov5n
```

部署链路：

```text
best.pt -> TensorRT engine -> YOLO 低频校正
HSV ROI -> 高频连续跟踪
```

实际接球系统里，YOLO 不建议阻塞控制循环。让 YOLO 在后台线程跑，HSV 和小车控制保持高频。

