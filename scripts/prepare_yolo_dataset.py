#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "datasets" / "tennis_ball"


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    fill_ratio: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and create a bootstrap YOLO dataset."
    )
    parser.add_argument("--input", default="data/raw", help="Directory with videos.")
    parser.add_argument("--output", default=str(DEFAULT_DATASET), help="Dataset dir.")
    parser.add_argument("--config", default="configs/tracker.yaml", help="Tracker YAML.")
    parser.add_argument("--fps", type=float, default=10.0, help="Extraction FPS.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--class-name", default="tennis ball")
    parser.add_argument("--bbox-scale", type=float, default=1.25)
    parser.add_argument("--keep-empty", action="store_true", help="Keep unlabeled frames.")
    parser.add_argument("--debug-images", type=int, default=80)
    parser.add_argument("--clean", action="store_true", help="Remove output dir first.")
    return parser.parse_args()


def video_files(input_dir: Path) -> list[Path]:
    exts = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def load_hsv_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["hsv"]


def make_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(cfg["lower"], dtype=np.uint8),
        np.array(cfg["upper"], dtype=np.uint8),
    )
    close_iterations = int(cfg.get("close_iterations", 1))
    if close_iterations > 0:
        mask = cv2.dilate(mask, None, iterations=close_iterations)
        mask = cv2.erode(mask, None, iterations=close_iterations)
    if int(cfg["erode_iterations"]) > 0:
        mask = cv2.erode(mask, None, iterations=int(cfg["erode_iterations"]))
    if int(cfg["dilate_iterations"]) > 0:
        mask = cv2.dilate(mask, None, iterations=int(cfg["dilate_iterations"]))
    return mask


def best_candidate(frame: np.ndarray, cfg: dict, bbox_scale: float) -> Optional[Candidate]:
    mask = make_mask(frame, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = frame.shape[:2]
    best: Optional[Candidate] = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        fill_ratio = area / max(1.0, float(w * h))
        aspect = max(w, h) / max(1.0, float(min(w, h)))
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        if circularity < cfg["min_circularity"]:
            continue
        if fill_ratio < cfg["min_mask_fill_ratio"]:
            continue
        if aspect > cfg["max_aspect_ratio"]:
            continue

        cx, cy = x + w / 2.0, y + h / 2.0
        size = max(w, h) * bbox_scale
        x1 = max(0, int(round(cx - size / 2.0)))
        y1 = max(0, int(round(cy - size / 2.0)))
        x2 = min(width - 1, int(round(cx + size / 2.0)))
        y2 = min(height - 1, int(round(cy + size / 2.0)))
        score = area * (0.5 + circularity) * (0.5 + fill_ratio)
        candidate = Candidate(
            bbox=(x1, y1, x2, y2),
            area=area,
            circularity=float(circularity),
            fill_ratio=float(fill_ratio),
            score=float(score),
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def yolo_label(candidate: Candidate, width: int, height: int) -> str:
    x1, y1, x2, y2 = candidate.bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n"


def ensure_layout(output: Path, clean: bool) -> None:
    if clean and output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "debug").mkdir(parents=True, exist_ok=True)


def split_for(index: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    stride = max(2, round(1.0 / val_ratio))
    return "val" if index % stride == 0 else "train"


def iter_frames(video: Path, target_fps: float) -> Iterable[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(source_fps / target_fps))
    frame_index = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % step == 0:
                yield frame_index, frame
    finally:
        cap.release()


def draw_debug(frame: np.ndarray, candidate: Candidate, name: str) -> np.ndarray:
    debug = frame.copy()
    x1, y1, x2, y2 = candidate.bbox
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        debug,
        name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return debug


def write_dataset_yaml(output: Path, class_name: str) -> None:
    yaml_path = output / "tennis_ball.yaml"
    rel_output = output.relative_to(PROJECT_ROOT)
    data = {
        "path": str(rel_output),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def main() -> int:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config_path = (PROJECT_ROOT / args.config).resolve()
    videos = video_files(input_dir)
    if not videos:
        print(f"No videos found in {input_dir}", file=sys.stderr)
        return 1

    hsv_cfg = load_hsv_config(config_path)
    ensure_layout(output, args.clean)
    write_dataset_yaml(output, args.class_name)

    rows = []
    total_frames = 0
    labeled_frames = 0
    debug_written = 0
    split_counts = {"train": 0, "val": 0}
    split_labels = {"train": 0, "val": 0}

    for video in videos:
        for frame_index, frame in iter_frames(video, args.fps):
            total_frames += 1
            candidate = best_candidate(frame, hsv_cfg, args.bbox_scale)
            has_label = candidate is not None
            if not has_label and not args.keep_empty:
                continue

            dataset_index = split_counts["train"] + split_counts["val"]
            split = split_for(dataset_index, args.val_ratio)
            stem = f"{video.stem}_f{frame_index:06d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            with label_path.open("w", encoding="utf-8") as f:
                if candidate is not None:
                    height, width = frame.shape[:2]
                    f.write(yolo_label(candidate, width, height))
                    labeled_frames += 1
                    split_labels[split] += 1
                    if debug_written < args.debug_images:
                        debug = draw_debug(frame, candidate, stem)
                        debug_path = output / "debug" / f"{stem}.jpg"
                        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        debug_written += 1
                else:
                    candidate = Candidate((0, 0, 0, 0), 0.0, 0.0, 0.0, 0.0)

            split_counts[split] += 1
            x1, y1, x2, y2 = candidate.bbox
            rows.append(
                {
                    "image": str(image_path.relative_to(output)),
                    "label": str(label_path.relative_to(output)),
                    "video": video.name,
                    "frame_index": frame_index,
                    "split": split,
                    "has_label": int(has_label),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "area": f"{candidate.area:.3f}",
                    "circularity": f"{candidate.circularity:.5f}",
                    "fill_ratio": f"{candidate.fill_ratio:.5f}",
                }
            )

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"videos: {len(videos)}")
    print(f"sampled_frames: {total_frames}")
    print(f"kept_frames: {split_counts['train'] + split_counts['val']}")
    print(f"labeled_frames: {labeled_frames}")
    print(f"train_images: {split_counts['train']} labels: {split_labels['train']}")
    print(f"val_images: {split_counts['val']} labels: {split_labels['val']}")
    print(f"dataset_yaml: {output / 'tennis_ball.yaml'}")
    print(f"manifest: {manifest}")
    print(f"debug_images: {debug_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
