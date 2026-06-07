#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_tracker.config import YoloConfig
from tennis_tracker.detection.yolo_detector import YOLODetector
from tennis_tracker.state import Detection


Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate every frame of a video with YOLO tennis-ball detections."
    )
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output annotated video path.")
    parser.add_argument(
        "--model",
        default="weights/best.pt",
        help="YOLO model path. Defaults to weights/best.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, for example 0, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--class-name",
        default="tennis ball",
        help="Target class name in the YOLO model.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=30,
        help="Number of recent center points to draw as a trail. Use 0 to disable.",
    )
    parser.add_argument(
        "--hide-missing",
        action="store_true",
        help="Do not draw 'no detection' text on frames where the ball is not detected.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. 0 means process the full video.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for output video. Defaults to mp4v.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def parse_device(value: str):
    return int(value) if value.isdigit() else value


def build_detector(args: argparse.Namespace) -> YOLODetector:
    model_path = resolve_path(args.model)
    cfg = YoloConfig(
        enabled=True,
        model_path=str(model_path),
        class_name=args.class_name,
        confidence=args.conf,
        imgsz=args.imgsz,
        device=parse_device(args.device),
        periodic_interval_ms=0,
        request_when_confidence_below=0,
    )
    detector = YOLODetector(cfg)
    detector.load()
    return detector


def open_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {output_path}")
    return writer


def draw_detection(frame, detection: Optional[Detection], trail: Deque[Point], args) -> None:
    if detection is None:
        if not args.hide_missing:
            cv2.putText(
                frame,
                "tennis ball: not detected",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return

    x1, y1, x2, y2 = detection.bbox
    cx, cy = int(round(detection.center[0])), int(round(detection.center[1]))
    radius = max(4, int(round(detection.radius)))
    label = f"tennis ball {detection.confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_w, text_h = text_size
    text_x = max(0, min(x1, frame.shape[1] - text_w - 8))
    text_y = max(text_h + 8, y1 - 8)
    cv2.rectangle(
        frame,
        (text_x, text_y - text_h - 8),
        (text_x + text_w + 8, text_y + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if args.trail > 0:
        trail.append((cx, cy))
        while len(trail) > args.trail:
            trail.popleft()
        for p1, p2 in zip(trail, list(trail)[1:]):
            cv2.line(frame, p1, p2, (255, 0, 255), 2)


def annotate_video(args: argparse.Namespace) -> None:
    source_path = resolve_path(args.source)
    output_path = resolve_path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    detector = build_detector(args)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = open_writer(output_path, fps, width, height, args.codec)
    trail: Deque[Point] = deque(maxlen=max(1, args.trail))
    start = time.time()
    frame_index = 0
    detected_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            detection = detector.detect(frame)
            if detection is not None:
                detected_frames += 1
            draw_detection(frame, detection, trail, args)
            writer.write(frame)

            frame_index += 1
            if frame_index == 1 or frame_index % 100 == 0:
                if total_frames > 0:
                    print(
                        f"processed {frame_index}/{total_frames} frames, "
                        f"detected {detected_frames}",
                        flush=True,
                    )
                else:
                    print(
                        f"processed {frame_index} frames, detected {detected_frames}",
                        flush=True,
                    )
    finally:
        cap.release()
        writer.release()

    elapsed = max(1e-6, time.time() - start)
    print(f"output: {output_path}")
    print(f"frames: {frame_index}")
    print(f"detected_frames: {detected_frames}")
    print(f"fps_processing: {frame_index / elapsed:.2f}")


def main() -> None:
    args = parse_args()
    annotate_video(args)


if __name__ == "__main__":
    main()
