import math

import cv2
import numpy as np

from config import (
    PIT_MAX_ASPECT_RATIO,
    PIT_MIN_CIRCULARITY,
    SCRATCH_MAX_MERGE_RATIO,
    SCRATCH_MIN_ASPECT_RATIO,
    SCRATCH_MIN_BOX_ASPECT_RATIO,
    STAIN_BOTTOM_MARGIN_RATIO,
    STAIN_SIDE_MARGIN_RATIO,
    STAIN_TOP_MARGIN_RATIO,
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


def touches_roi_bottom(contour, roi_height):
    """触及 ROI 下边界的轮廓可能混入钢轴底缘或背景，不能完整测量。"""
    _, y, _, height = cv2.boundingRect(contour)
    return y + height >= roi_height


def touches_detection_boundary(contour, detection_area):
    """被有效区裁断的候选不能当作完整缺陷测量。"""
    x, y, width, height = cv2.boundingRect(contour)
    if cv2.countNonZero(detection_area[y:y + height, x:x + width]) == 0:
        return True
    return (
        detection_area[y, x] == 0
        or detection_area[y + height - 1, x + width - 1] == 0
        or x == 0 or y == 0
        or x + width == detection_area.shape[1]
        or y + height == detection_area.shape[0]
        or detection_area[y, x - 1] == 0
        or detection_area[y - 1, x] == 0
        or detection_area[y + height, x] == 0
        or detection_area[y, x + width] == 0
    )


def detect_scratch(contour, scale, width_info=None):
    """检测划痕；骨架测宽，外接矩形只校验整体是否细长。"""
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
    short_side = min(rw, rh)
    skeleton_width = width_info["width"]
    aspect_ratio = long_side / skeleton_width
    box_aspect_ratio = long_side / short_side
    merge_ratio = area / (long_side * skeleton_width)

    if not (
        area >= scale.scratch_min_area_px
        and long_side >= scale.scratch_min_length_px
        and skeleton_width <= scale.scratch_max_width_px
        and width_info["max_width"] <= scale.scratch_max_width_px
        and aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO
        and box_aspect_ratio >= SCRATCH_MIN_BOX_ASPECT_RATIO
        and merge_ratio <= SCRATCH_MAX_MERGE_RATIO
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


def detect_pit(contour, scale):
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
        area >= scale.pit_min_area_px
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


def stain_rejection_reason(contour, image_shape, scale):
    """面积过小或触及候选区域边界时，无法可靠确定片状缺陷范围。"""
    height, width = image_shape[:2]
    area = cv2.contourArea(contour)
    minimum = scale.stain_min_area_px
    if area < minimum:
        return f"area {area:.1f} < {minimum:.1f}"

    x, y, box_width, box_height = cv2.boundingRect(contour)
    top = int(height * STAIN_TOP_MARGIN_RATIO)
    bottom = int(height * (1 - STAIN_BOTTOM_MARGIN_RATIO))
    side = int(width * STAIN_SIDE_MARGIN_RATIO)
    if y <= top or y + box_height >= bottom or x <= side or x + box_width >= width - side:
        return "触及污渍检测区边界，轮廓可能被截断或混入钢轴边缘/背景"
    return None


def detect_stain(contour, image_shape, scale):
    """片状锈斑/污渍不套用划痕细长或凹点圆度条件。"""
    if stain_rejection_reason(contour, image_shape, scale) is not None:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    return {
        "type": "Stain",
        "center": (center_x, center_y),
        "contour": contour,
        "area": cv2.contourArea(contour),
        "size_px": 2 * radius,
    }
