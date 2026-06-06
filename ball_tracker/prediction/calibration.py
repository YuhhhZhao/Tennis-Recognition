"""相机标定 — 加载/保存/标定 CameraIntrinsics"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .geometry import CameraIntrinsics


def load_calibration(path: str | Path) -> Optional[CameraIntrinsics]:
    """从 .npz 文件加载标定参数. 文件不存在或损坏返回 None."""
    try:
        data = np.load(str(path))
        K = data["K"]
        dist = data["dist_coeffs"]
        return CameraIntrinsics(K=K, dist_coeffs=dist)
    except (FileNotFoundError, KeyError, OSError):
        return None


def save_calibration(path: str | Path, calib: CameraIntrinsics) -> None:
    """保存标定参数为 .npz 文件."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), K=calib.K, dist_coeffs=calib.dist_coeffs)


def calibrate_from_images(
    image_paths: list,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> Optional[CameraIntrinsics]:
    """从棋盘格图片列表标定相机.

    Parameters
    ----------
    image_paths : list of str or np.ndarray
        棋盘格图片路径列表, 或直接传入图像数组列表.
    pattern_size : (cols, rows)
        棋盘格内角点数. 默认 9x6 (10x7 方格).
    square_size_m : float
        棋盘格方格边长 (米).

    Returns
    -------
    CameraIntrinsics or None (标定失败时).
    """
    import cv2

    cols, rows = pattern_size
    pattern_points = np.zeros((cols * rows, 3), np.float32)
    pattern_points[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    )

    obj_points: list = []
    img_points: list = []
    img_shape = None

    for item in image_paths:
        if isinstance(item, (str, Path)):
            img = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
        elif isinstance(item, np.ndarray):
            img = item
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            continue

        if img_shape is None:
            img_shape = img.shape[:2][::-1]  # (w, h)

        found, corners = cv2.findChessboardCorners(img, (cols, rows), None)
        if not found:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(pattern_points)
        img_points.append(corners_refined)

    if len(obj_points) < 10:
        return None

    if img_shape is None:
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    if not ret:
        return None

    return CameraIntrinsics(K=K, dist_coeffs=dist)


def generate_chessboard_png(
    path: str | Path,
    cols: int = 9,
    rows: int = 6,
    square_px: int = 80,
) -> None:
    """生成棋盘格 PNG 图片, 用于打印."""
    import cv2

    width = (cols + 1) * square_px
    height = (rows + 1) * square_px
    board = np.zeros((height, width), dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y1, y2 = r * square_px, (r + 1) * square_px
                x1, x2 = c * square_px, (c + 1) * square_px
                board[y1:y2, x1:x2] = 255
    cv2.imwrite(str(path), board)
