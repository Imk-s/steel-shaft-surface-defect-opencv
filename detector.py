import math

import cv2
import numpy as np

from config import (
    PIT_MAX_ASPECT_RATIO,
    PIT_MIN_AREA,
    PIT_MIN_CIRCULARITY,
    SCRATCH_MAX_WIDTH,
    SCRATCH_MIN_AREA,
    SCRATCH_MIN_ASPECT_RATIO,
    SCRATCH_MIN_LENGTH,
)


def measure_scratch_width(contour, padding=2):
    """用骨架点到轮廓边缘距离的两倍估计划痕宽度。"""
    bx, by, bw, bh = cv2.boundingRect(contour)

    if bw <= 0 or bh <= 0:
        return None

    local_contour = contour.copy()
    local_contour[:, 0, 0] -= bx - padding
    local_contour[:, 0, 1] -= by - padding

    contour_mask = np.zeros(
        (bh + 2 * padding, bw + 2 * padding),
        dtype=np.uint8,
    )
    cv2.drawContours(
        contour_mask,
        [local_contour],
        -1,
        255,
        cv2.FILLED,
    )

    distance = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
    skeleton = np.zeros_like(contour_mask)
    work = contour_mask.copy()
    skeleton_kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3),
    )

    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(
            work,
            cv2.MORPH_OPEN,
            skeleton_kernel,
        )
        skeleton_part = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, skeleton_part)
        work = cv2.erode(work, skeleton_kernel)

    radii = distance[skeleton > 0]
    radii = radii[radii > 0]

    if radii.size == 0:
        return None

    diameters = 2.0 * radii
    return {
        "width": float(np.median(diameters)),
        "mean_width": float(np.mean(diameters)),
        "min_width": float(np.min(diameters)),
        "max_width": float(np.max(diameters)),
        "skeleton_pixels": int(radii.size),
    }


def detect_scratch(contour, width_info=None):
    """检测划痕；宽度使用骨架距离估计，外接矩形仅估计长度。"""
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ = rect

    if rw == 0 or rh == 0:
        return None

    if width_info is None:
        width_info = measure_scratch_width(contour)

    if width_info is None or width_info["width"] <= 0:
        return None

    long_side = max(rw, rh)
    skeleton_width = width_info["width"]
    aspect_ratio = long_side / skeleton_width

    if not (
        area >= SCRATCH_MIN_AREA
        and long_side >= SCRATCH_MIN_LENGTH
        and skeleton_width <= SCRATCH_MAX_WIDTH
        and aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO
    ):
        return None

    return {
        "type": "Scratch",
        "size_px": skeleton_width,
        "width_px": skeleton_width,
        "width_min_px": width_info["min_width"],
        "width_max_px": width_info["max_width"],
        "skeleton_pixels": width_info["skeleton_pixels"],
        "center": (cx, cy),
        "box": np.int32(cv2.boxPoints(rect)),
        "area": area,
        "aspect_ratio": aspect_ratio,
    }


def detect_pit(contour):
    """检测凹点，命中时返回凹点参数。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]

    if rw == 0 or rh == 0:
        return None

    aspect_ratio = max(rw, rh) / min(rw, rh)
    circularity = 4 * math.pi * area / (perimeter * perimeter)

    if not (
        area >= PIT_MIN_AREA
        and circularity >= PIT_MIN_CIRCULARITY
        and aspect_ratio <= PIT_MAX_ASPECT_RATIO
    ):
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return {
        "type": "Pit",
        "size_px": 2 * radius,
        "center": (cx, cy),
        "radius": radius,
        "area": area,
        "circularity": circularity,
    }
