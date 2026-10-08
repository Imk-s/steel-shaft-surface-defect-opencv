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
    DEBUG_STAIN_COLOR,
    PIT_MAX_ASPECT_RATIO,
    PIT_MIN_CIRCULARITY,
    MIN_DEFECT_SIZE_MM,
    SCRATCH_MAX_MERGE_RATIO,
    SCRATCH_MIN_ASPECT_RATIO,
    SCRATCH_MIN_BOX_ASPECT_RATIO,
)


@dataclass
class DebugContext:
    run_dir: Path
    mask_contours: np.ndarray
    clean_contours: np.ndarray
    rejected_contours: np.ndarray
    stain_contours: np.ndarray | None = None


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
        stain_contours=crop.copy(),
    )


def record_clean_contour(
    context,
    contour_id,
    contour,
    detection,
    scratch_width_info,
    rejection_reason=None,
    measurement=None,
    scale=None,
):
    """绘制 C 轮廓并打印分类属性；拒绝项额外打印具体原因。"""
    if scale is None:
        raise ValueError("record_clean_contour 需要当前图片的 PixelScale")
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
    merge_ratio = (
        area / (long_side * skeleton_width)
        if long_side > 0 and skeleton_width > 0
        else float("inf")
    )
    circularity = (
        4 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )

    measurement_rejection = None
    if detection is not None and measurement is not None:
        if not measurement.measurable:
            measurement_rejection = f"不可测: {measurement.reason}"
        elif not measurement.valid:
            size_mm = (
                measurement.width_mm if detection["type"] == "Scratch"
                else measurement.diameter_mm
            )
            measurement_rejection = (
                f"尺寸 {size_mm:.6f} mm < {MIN_DEFECT_SIZE_MM} mm"
            )

    if detection is not None and measurement_rejection is None and detection["type"] == "Scratch":
        debug_type = "Scratch"
        debug_color = DEBUG_SCRATCH_COLOR
    elif detection is not None and measurement_rejection is None and detection["type"] == "Pit":
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

    candidate_text = (
        f"candidate={detection['type']} "
        if detection is not None and measurement_rejection
        else ""
    )
    print(
        f"[CLEAN {label}] type={debug_type} {candidate_text}"
        f"area={area:.1f} perimeter={perimeter:.1f} "
        f"bbox=({bx},{by},{bw},{bh}) long={long_side:.1f} "
        f"box_short={short_side:.1f} "
        f"box_aspect={box_aspect_ratio:.3f} "
        f"skeleton_width={skeleton_width:.2f} "
        f"skeleton_range={skeleton_width_min:.2f}-{skeleton_width_max:.2f} "
        f"skeleton_aspect={skeleton_aspect_ratio:.3f} "
        f"merge_ratio={merge_ratio:.3f} "
        f"circularity={circularity:.3f}"
    )

    if detection is not None and measurement_rejection is None:
        return

    _draw_rejected_contour(context, contour_id, contour, bx, by, bw, bh)
    if measurement_rejection is not None:
        print(f"[CLEAN {label}] REJECT {detection['type']}: {measurement_rejection}")
        return
    if rejection_reason is not None:
        print(f"[CLEAN {label}] REJECT ROI: {rejection_reason}")
        return
    scratch_reasons = _scratch_rejection_reasons(
        area,
        rw,
        rh,
        long_side,
        skeleton_width,
        skeleton_aspect_ratio,
        box_aspect_ratio,
        scratch_width_info,
        scale,
    )
    pit_reasons = _pit_rejection_reasons(
        area,
        perimeter,
        rw,
        rh,
        box_aspect_ratio,
        circularity,
        scale,
    )
    print(f"[CLEAN {label}] REJECT Scratch: " + "; ".join(scratch_reasons))
    print(f"[CLEAN {label}] REJECT Pit: " + "; ".join(pit_reasons))


def record_stain_contour(context, contour_id, contour, detection, measurement, rejection_reason=None):
    """污渍候选独立编号，显示最终类别及过滤原因。"""
    if context.stain_contours is None:
        return
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    label = f"S{contour_id}"
    if detection is None:
        reason = rejection_reason or "未形成有效污渍轮廓"
    elif not measurement.measurable:
        reason = f"不可测: {measurement.reason}"
    elif not measurement.valid:
        reason = f"跨度 {measurement.diameter_mm:.6f} mm < {MIN_DEFECT_SIZE_MM} mm"
    else:
        reason = None
    accepted = reason is None
    color = DEBUG_STAIN_COLOR if accepted else DEBUG_REJECTED_COLOR
    cv2.drawContours(context.stain_contours, [contour], -1, color, 2)
    cv2.putText(
        context.stain_contours, label, (x, max(10, y - 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
    )
    status = "Stain" if accepted else "Rejected"
    print(
        f"[STAIN {label}] type={status} area={area:.1f} "
        f"bbox=({x},{y},{width},{height})"
    )
    if not accepted:
        _draw_rejected_contour(context, contour_id, contour, x, y, width, height, prefix="S")
        print(f"[STAIN {label}] REJECT: {reason}")


def _draw_rejected_contour(context, contour_id, contour, bx, by, bw, bh, prefix="C"):
    color = DEBUG_REJECTED_COLOR
    cv2.drawContours(context.rejected_contours, [contour], -1, color, 2)
    cv2.rectangle(
        context.rejected_contours,
        (bx, by),
        (bx + bw, by + bh),
        color,
        1,
    )
    label = f"{prefix}{contour_id}"
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
    box_aspect_ratio,
    width_info,
    scale,
):
    reasons = []

    if area < scale.scratch_min_area_px:
        reasons.append(f"area {area:.1f} < {scale.scratch_min_area_px:.1f}")
    if rw == 0 or rh == 0:
        reasons.append("旋转包围框存在零边长")
    elif long_side < scale.scratch_min_length_px:
        reasons.append(f"long_side {long_side:.1f} < {scale.scratch_min_length_px:.1f}")

    if width_info is None:
        reasons.append("无法从骨架计算宽度")
    else:
        if skeleton_width > scale.scratch_max_width_px:
            reasons.append(
                f"skeleton_width {skeleton_width:.2f} > {scale.scratch_max_width_px:.2f}"
            )
        if width_info["max_width"] > scale.scratch_max_width_px:
            reasons.append(
                f"skeleton_max_width {width_info['max_width']:.2f} > {scale.scratch_max_width_px:.2f}"
            )
        if skeleton_aspect_ratio < SCRATCH_MIN_ASPECT_RATIO:
            reasons.append(
                "skeleton_aspect "
                f"{skeleton_aspect_ratio:.3f} < {SCRATCH_MIN_ASPECT_RATIO}"
            )
        if box_aspect_ratio < SCRATCH_MIN_BOX_ASPECT_RATIO:
            reasons.append(
                f"box_aspect {box_aspect_ratio:.3f} < {SCRATCH_MIN_BOX_ASPECT_RATIO}"
            )
        if long_side > 0 and skeleton_width > 0:
            merge_ratio = area / (long_side * skeleton_width)
            if merge_ratio > SCRATCH_MAX_MERGE_RATIO:
                reasons.append(
                    f"merge_ratio {merge_ratio:.3f} > {SCRATCH_MAX_MERGE_RATIO}"
                )

    return reasons


def _pit_rejection_reasons(
    area,
    perimeter,
    rw,
    rh,
    aspect_ratio,
    circularity,
    scale,
):
    reasons = []

    if area < scale.pit_min_area_px:
        reasons.append(f"area {area:.1f} < {scale.pit_min_area_px:.1f}")
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


def finish_debug_run(context, raw_contour_count, clean_contour_count, stain_contour_count=0, *, save_images=True):
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
        if context.stain_contours is not None:
            save_debug_image(
                context.run_dir / "debug_stain_contours.jpg",
                context.stain_contours,
            )
    print(
        "[DEBUG] 阈值轮廓:",
        raw_contour_count,
        "清理后轮廓:",
        clean_contour_count,
        "污渍候选轮廓:",
        stain_contour_count,
    )
    print("[DEBUG] Clean: 红色=Scratch 绿色=Pit 黄色=Rejected；Stain: 青色=Stain 黄色=Rejected")
    if save_images:
        print("[DEBUG] 调试图片目录:", context.run_dir)
    else:
        print("[DEBUG] 本轮仅返回内存调试图片，未写盘")
