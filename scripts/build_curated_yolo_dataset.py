#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

from prepare_yolo_dataset import (
    PROJECT_ROOT,
    best_candidate,
    draw_debug,
    ensure_layout,
    iter_frames,
    load_hsv_config,
    split_for,
    write_dataset_yaml,
    yolo_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new YOLO dataset from manually kept debug previews plus "
            "new raw videos."
        )
    )
    parser.add_argument("--old-dataset", default="data/datasets/tennis_ball")
    parser.add_argument("--output", default="data/datasets/tennis_ball_v2")
    parser.add_argument("--config", default="configs/tracker.yaml")
    parser.add_argument("--new-video", action="append", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--class-name", default="tennis ball")
    parser.add_argument("--bbox-scale", type=float, default=1.25)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def find_old_sample(old_dataset: Path, stem: str) -> tuple[Path, Path] | None:
    for split in ("train", "val"):
        image = old_dataset / "images" / split / f"{stem}.jpg"
        label = old_dataset / "labels" / split / f"{stem}.txt"
        if image.exists() and label.exists():
            return image, label
    return None


def copy_old_curated(
    old_dataset: Path,
    output: Path,
    val_ratio: float,
    rows: list[dict[str, str]],
) -> tuple[int, int]:
    kept_debug = sorted((old_dataset / "debug").glob("*.jpg"))
    copied = 0
    missing = 0

    for debug_image in kept_debug:
        stem = debug_image.stem
        sample = find_old_sample(old_dataset, stem)
        if sample is None:
            missing += 1
            continue

        split = split_for(len(rows), val_ratio)
        image_src, label_src = sample
        image_dst = output / "images" / split / image_src.name
        label_dst = output / "labels" / split / label_src.name
        debug_dst = output / "debug" / debug_image.name
        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)
        shutil.copy2(debug_image, debug_dst)
        rows.append(
            {
                "image": str(image_dst.relative_to(output)),
                "label": str(label_dst.relative_to(output)),
                "video": stem.split("_f", 1)[0],
                "frame_index": stem.rsplit("_f", 1)[-1],
                "split": split,
                "source": "old_debug_kept",
                "has_label": "1",
            }
        )
        copied += 1

    return copied, missing


def add_new_videos(
    videos: list[Path],
    output: Path,
    hsv_cfg: dict,
    fps: float,
    val_ratio: float,
    bbox_scale: float,
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    sampled = 0
    kept = 0
    skipped = 0

    for video in videos:
        for frame_index, frame in iter_frames(video, fps):
            sampled += 1
            candidate = best_candidate(frame, hsv_cfg, bbox_scale)
            if candidate is None:
                skipped += 1
                continue

            split = split_for(len(rows), val_ratio)
            stem = f"{video.stem}_f{frame_index:06d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            debug_path = output / "debug" / f"{stem}.jpg"

            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            height, width = frame.shape[:2]
            label_path.write_text(yolo_label(candidate, width, height), encoding="utf-8")
            debug = draw_debug(frame, candidate, stem)
            cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 90])

            rows.append(
                {
                    "image": str(image_path.relative_to(output)),
                    "label": str(label_path.relative_to(output)),
                    "video": video.name,
                    "frame_index": str(frame_index),
                    "split": split,
                    "source": "new_video_auto",
                    "has_label": "1",
                }
            )
            kept += 1

    return sampled, kept, skipped


def write_manifest(output: Path, rows: list[dict[str, str]]) -> None:
    manifest = output / "manifest.csv"
    fieldnames = ["image", "label", "video", "frame_index", "split", "source", "has_label"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    old_dataset = (PROJECT_ROOT / args.old_dataset).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config_path = (PROJECT_ROOT / args.config).resolve()
    videos = [(PROJECT_ROOT / video).resolve() for video in args.new_video]

    missing_videos = [str(video) for video in videos if not video.exists()]
    if missing_videos:
        print(f"Missing videos: {missing_videos}", file=sys.stderr)
        return 1

    ensure_layout(output, args.clean)
    write_dataset_yaml(output, args.class_name)
    rows: list[dict[str, str]] = []

    old_copied, old_missing = copy_old_curated(
        old_dataset=old_dataset,
        output=output,
        val_ratio=args.val_ratio,
        rows=rows,
    )
    hsv_cfg = load_hsv_config(config_path)
    sampled, new_kept, new_skipped = add_new_videos(
        videos=videos,
        output=output,
        hsv_cfg=hsv_cfg,
        fps=args.fps,
        val_ratio=args.val_ratio,
        bbox_scale=args.bbox_scale,
        rows=rows,
    )
    write_manifest(output, rows)

    train_images = len(list((output / "images" / "train").glob("*.jpg")))
    val_images = len(list((output / "images" / "val").glob("*.jpg")))
    debug_images = len(list((output / "debug").glob("*.jpg")))
    print(f"old_debug_kept_copied: {old_copied}")
    print(f"old_debug_missing_samples: {old_missing}")
    print(f"new_video_sampled_frames: {sampled}")
    print(f"new_video_labeled_frames: {new_kept}")
    print(f"new_video_skipped_frames: {new_skipped}")
    print(f"train_images: {train_images}")
    print(f"val_images: {val_images}")
    print(f"debug_images: {debug_images}")
    print(f"dataset_yaml: {output / 'tennis_ball.yaml'}")
    print(f"manifest: {output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
