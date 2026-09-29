import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEBUG_OUTPUT_ROOT,
    DEBUG_PIT_COLOR,
    DEBUG_REJECTED_COLOR,
    DEBUG_SCRATCH_COLOR,
    PIT_MAX_ASPECT_RATIO,
    PIT_MIN_AREA,
    PIT_MIN_CIRCULARITY,
    SCRATCH_MAX_WIDTH,
    SCRATCH_MIN_AREA,
    SCRATCH_MIN_ASPECT_RATIO,
    SCRATCH_MIN_LENGTH,
)


@dataclass
class DebugContext:
    run_dir: Path
    mask_contours: np.ndarray
    clean_contours: np.ndarray
    rejected_contours: np.ndarray


def save_debug_image(path, image):
    """通过内存编码保存图片，兼容 Windows 中文路径。"""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"调试图片编码失败: {path}")

    encoded.tofile(str(path))


def start_debug_run(mask, crop, raw_contours, *, output_root=None, save_images=True):
    """创建本轮 Debug 目录，并标记原始 mask 的所有 M 轮廓。"""
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")[:-3]
    root = DEBUG_OUTPUT_ROOT if output_root is None else Path(output_root)
    run_dir = root / run_name
    if save_images:
        # 同毫秒重复运行时使用递增后缀，保持已有目录和文件完整。
        suffix = 0
        while True:
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                run_dir = root / f"{run_name}_{suffix}"

    mask_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color = (0, 165, 255)

    for contour_id, contour in enumerate(raw_contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        bx, by, bw, bh = cv2.boundingRect(contour)
        label = f"M{contour_id}"
        label_position = (bx, max(14, by - 3))

        cv2.drawContours(mask_contours, [contour], -1, mask_color, 1)
        cv2.putText(
            mask_contours,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            mask_contours,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            mask_color,
            1,
        )
        print(
            f"[MASK {label}] area={area:.1f} "
            f"perimeter={perimeter:.1f} "
            f"bbox=({bx},{by},{bw},{bh})"
        )

    return DebugContext(
        run_dir=run_dir,
        mask_contours=mask_contours,
        clean_contours=crop.copy(),
        rejected_contours=cv2.convertScaleAbs(crop, alpha=0.35, beta=0),
    )


def record_clean_contour(
    context,
    contour_id,
    contour,
    detection,
    scratch_width_info,
):
    """绘制 C 轮廓并打印分类属性；拒绝项额外打印具体原因。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    bx, by, bw, bh = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    long_side = max(rw, rh)
    short_side = min(rw, rh)
    box_aspect_ratio = (
        long_side / short_side if short_side > 0 else float("inf")
    )
    skeleton_width = (
        scratch_width_info["width"]
        if scratch_width_info is not None
        else 0.0
    )
    skeleton_width_min = (
        scratch_width_info["min_width"]
        if scratch_width_info is not None
        else 0.0
    )
    skeleton_width_max = (
        scratch_width_info["max_width"]
        if scratch_width_info is not None
        else 0.0
    )
    skeleton_aspect_ratio = (
        long_side / skeleton_width
        if skeleton_width > 0
        else float("inf")
    )
    circularity = (
        4 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )

    if detection is not None and detection["type"] == "Scratch":
        debug_type = "Scratch"
        debug_color = DEBUG_SCRATCH_COLOR
    elif detection is not None and detection["type"] == "Pit":
        debug_type = "Pit"
        debug_color = DEBUG_PIT_COLOR
    else:
        debug_type = "Rejected"
        debug_color = DEBUG_REJECTED_COLOR

    cv2.drawContours(
        context.clean_contours,
        [contour],
        -1,
        debug_color,
        1,
    )
    label = f"C{contour_id}"
    label_position = (bx, max(10, by - 2))
    cv2.putText(
        context.clean_contours,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        context.clean_contours,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        debug_color,
        1,
    )

    print(
        f"[CLEAN {label}] type={debug_type} "
        f"area={area:.1f} perimeter={perimeter:.1f} "
        f"bbox=({bx},{by},{bw},{bh}) long={long_side:.1f} "
        f"box_short={short_side:.1f} "
        f"box_aspect={box_aspect_ratio:.3f} "
        f"skeleton_width={skeleton_width:.2f} "
        f"skeleton_range={skeleton_width_min:.2f}-{skeleton_width_max:.2f} "
        f"skeleton_aspect={skeleton_aspect_ratio:.3f} "
        f"circularity={circularity:.3f}"
    )

    if detection is not None:
        return

    _draw_rejected_contour(context, contour_id, contour, bx, by, bw, bh)
    scratch_reasons = _scratch_rejection_reasons(
        area,
        rw,
        rh,
        long_side,
        skeleton_width,
        skeleton_aspect_ratio,
        scratch_width_info,
    )
    pit_reasons = _pit_rejection_reasons(
        area,
        perimeter,
        rw,
        rh,
        box_aspect_ratio,
        circularity,
    )
    print(f"[CLEAN {label}] REJECT Scratch: " + "; ".join(scratch_reasons))
    print(f"[CLEAN {label}] REJECT Pit: " + "; ".join(pit_reasons))


def _draw_rejected_contour(context, contour_id, contour, bx, by, bw, bh):
    color = DEBUG_REJECTED_COLOR
    cv2.drawContours(context.rejected_contours, [contour], -1, color, 2)
    cv2.rectangle(
        context.rejected_contours,
        (bx, by),
        (bx + bw, by + bh),
        color,
        1,
    )
    label = f"C{contour_id}"
    label_position = (bx, max(14, by - 4))
    cv2.putText(
        context.rejected_contours,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        context.rejected_contours,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
    )


def _scratch_rejection_reasons(
    area,
    rw,
    rh,
    long_side,
    skeleton_width,
    skeleton_aspect_ratio,
    width_info,
):
    reasons = []

    if area < SCRATCH_MIN_AREA:
        reasons.append(f"area {area:.1f} < {SCRATCH_MIN_AREA}")
    if rw == 0 or rh == 0:
        reasons.append("旋转包围框存在零边长")
    elif long_side < SCRATCH_MIN_LENGTH:
        reasons.append(f"long_side {long_side:.1f} < {SCRATCH_MIN_LENGTH}")

    if width_info is None:
        reasons.append("无法从骨架计算宽度")
    else:
        if skeleton_width > SCRATCH_MAX_WIDTH:
            reasons.append(
                f"skeleton_width {skeleton_width:.2f} > {SCRATCH_MAX_WIDTH}"
            )
        if skeleton_aspect_ratio < SCRATCH_MIN_ASPECT_RATIO:
            reasons.append(
                "skeleton_aspect "
                f"{skeleton_aspect_ratio:.3f} < {SCRATCH_MIN_ASPECT_RATIO}"
            )

    return reasons


def _pit_rejection_reasons(
    area,
    perimeter,
    rw,
    rh,
    aspect_ratio,
    circularity,
):
    reasons = []

    if area < PIT_MIN_AREA:
        reasons.append(f"area {area:.1f} < {PIT_MIN_AREA}")
    if perimeter == 0:
        reasons.append("perimeter = 0")
    if rw == 0 or rh == 0:
        reasons.append("旋转包围框存在零边长")
    else:
        if circularity < PIT_MIN_CIRCULARITY:
            reasons.append(
                f"circularity {circularity:.3f} < {PIT_MIN_CIRCULARITY}"
            )
        if aspect_ratio > PIT_MAX_ASPECT_RATIO:
            reasons.append(
                f"aspect_ratio {aspect_ratio:.3f} > {PIT_MAX_ASPECT_RATIO}"
            )

    return reasons


def finish_debug_run(context, raw_contour_count, clean_contour_count, *, save_images=True):
    if save_images:
        save_debug_image(
            context.run_dir / "debug_mask_contours.jpg",
            context.mask_contours,
        )
        save_debug_image(
            context.run_dir / "debug_clean_contours.jpg",
            context.clean_contours,
        )
        save_debug_image(
            context.run_dir / "debug_rejected_only.jpg",
            context.rejected_contours,
        )
    print(
        "[DEBUG] 阈值轮廓:",
        raw_contour_count,
        "清理后轮廓:",
        clean_contour_count,
    )
    print("[DEBUG] 颜色: 红色=Scratch 绿色=Pit 黄色=Rejected")
    if save_images:
        print("[DEBUG] 调试图片目录:", context.run_dir)
    else:
        print("[DEBUG] 本轮仅返回内存调试图片，未写盘")
