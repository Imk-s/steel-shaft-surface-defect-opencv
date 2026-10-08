"""无交互检测接口：编排已有函数，不包含 Qt 或新的检测算法。"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, replace
from datetime import datetime
from io import StringIO
import math
from operator import index
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

from config import DEBUG_ALL_CONTOURS, DEBUG_OUTPUT_ROOT, MASK_PATH, MIN_DEFECT_SIZE_MM, RESULT_PATH, pixel_scale
from debug_utils import finish_debug_run, record_clean_contour, record_stain_contour, start_debug_run
from detector import (
    detect_pit, detect_scratch, detect_stain, measure_scratch_width,
    stain_rejection_reason, touches_detection_boundary,
)
from main import draw_detection, write_image
from measurement import (
    DEFAULT_SHAFT_MASK_PATH,
    calibrate_shaft_mask,
    create_shaft_mask_from_roi,
    measure_pit_contour,
    measure_scratch_contour,
    print_calibration,
)
from preprocess import create_detection_area, preprocess_image


@dataclass(frozen=True)
class PipelineResult:
    """最近成功输出的完整快照；空实例也可用作 GUI 初始状态。"""

    result_image: np.ndarray | None = None
    intermediate_images: dict[str, np.ndarray] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    logs: str = ""
    run_directory: Path | None = None


_CAPTURE_LOCK = threading.RLock()


class _ThreadTee:
    """只捕获运行线程的 print，同时保留终端输出。"""

    def __init__(self, terminal, capture, owner):
        self.terminal = terminal
        self.capture = capture
        self.owner = owner

    def write(self, text):
        if threading.get_ident() == self.owner:
            self.capture.write(text)
        if self.terminal is not None:
            self.terminal.write(text)
        return len(text)

    def flush(self):
        if self.terminal is not None:
            self.terminal.flush()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def isatty(self):
        return bool(self.terminal is not None and self.terminal.isatty())


def _validate_inputs(image, roi, diameter_mm):
    if (
        not isinstance(image, np.ndarray) or image.dtype != np.uint8
        or image.ndim != 3 or image.shape[2] != 3 or image.size == 0
    ):
        raise ValueError("检测图片必须是非空 uint8 BGR 原图")
    try:
        if len(roi) != 4:
            raise ValueError
        roi = tuple(index(value) for value in roi)
        diameter_mm = float(diameter_mm)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("ROI 必须是四个原图整数坐标，真实直径必须是正数") from error
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("钢轴真实直径必须是大于 0 的有限数值")
    # 范围检查继续由原有 create_shaft_mask_from_roi 完成。
    return roi, diameter_mm


def run_pipeline(image, roi, diameter_mm, *, save_outputs=True, output_root=None, debug_enabled=None):
    """接收原始图片和原图 ROI，返回图片、调试图和捕获日志。

    GUI 默认保存到 DEBUG_OUTPUT_ROOT 下独立运行目录。save_outputs=False
    完全不写盘，适合测试或直接调用。不会调用 selectROI/imshow/waitKey。
    """
    capture = StringIO()
    owner = threading.get_ident()
    with _CAPTURE_LOCK:
        with (
            redirect_stdout(_ThreadTee(sys.stdout, capture, owner)),
            redirect_stderr(_ThreadTee(sys.stderr, capture, owner)),
        ):
            result = _execute_pipeline(
                image, roi, diameter_mm,
                save_outputs=save_outputs,
                output_root=output_root,
                debug_enabled=DEBUG_ALL_CONTOURS if debug_enabled is None else debug_enabled,
            )
            logs = capture.getvalue()
            if result.run_directory is not None:
                (result.run_directory / "pipeline.log").write_text(logs, encoding="utf-8")
            return replace(result, logs=logs)


def _execute_pipeline(image, roi, diameter_mm, *, save_outputs, output_root, debug_enabled):
    roi, diameter_mm = _validate_inputs(image, roi, diameter_mm)
    roi_x, roi_y, roi_w, roi_h = roi
    shaft_mask = create_shaft_mask_from_roi(image, roi)
    print(f"[PIPELINE] 原图={image.shape[1]}x{image.shape[0]} ROI={roi} D_mm={diameter_mm}")
    print(f"[PIPELINE] 当前尺寸门槛={MIN_DEFECT_SIZE_MM} mm")
    calibration = calibrate_shaft_mask(shaft_mask, diameter_mm)
    print_calibration(calibration)
    scale = pixel_scale(calibration.diameter_px / diameter_mm)
    detection_area = create_detection_area(shaft_mask.shape, calibration)
    area_x, area_y, area_w, area_h = cv2.boundingRect(detection_area)
    print(
        f"[SCALE] px/mm={scale.px_per_mm:.4f} "
        f"min_width_px={scale.min_width_px:.2f} "
        f"component_area_px={scale.min_component_area_px:.1f} "
        f"denoise_kernel={scale.denoise_kernel_px} "
        f"connect_gap={scale.connect_gap_px}"
    )
    print(f"[DETECTION AREA] x={area_x} y={area_y} w={area_w} h={area_h} (ROI 内坐标)")

    result_image = image.copy()
    crop = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    processed = preprocess_image(crop, scale, detection_area)
    mask, clean = processed["mask"], processed["clean"]
    stain_mask = processed["stain_mask"]
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stain_contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    root = DEBUG_OUTPUT_ROOT if output_root is None else Path(output_root)
    context = None
    run_directory = None
    if debug_enabled:
        context = start_debug_run(mask, crop, raw_contours, output_root=root, save_images=save_outputs)
        if save_outputs:
            run_directory = context.run_dir
    elif save_outputs:
        run_directory = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_directory.mkdir(parents=True, exist_ok=False)

    counts = {
        "border_rejected": 0, "shape_rejected": 0, "unmeasurable": 0,
        "below_size": 0, "Scratch": 0, "Pit": 0,
        "stain_rejected": 0, "stain_unmeasurable": 0,
        "stain_below_size": 0, "Stain": 0,
    }
    for contour_id, contour in enumerate(contours):
        width_info = measure_scratch_width(contour)
        border_rejected = touches_detection_boundary(contour, detection_area)
        detection = None if border_rejected else detect_scratch(contour, scale, width_info=width_info)
        if detection is None and not border_rejected:
            detection = detect_pit(contour, scale)
        physical_measurement = None
        if detection is not None:
            physical_measurement = (
                measure_scratch_contour(contour, calibration)
                if detection["type"] == "Scratch"
                else measure_pit_contour(contour, calibration)
            )
        if context is not None:
            record_clean_contour(
                context, contour_id, contour, detection, width_info,
                rejection_reason="触及 ROI 有效检测区边界，轮廓可能被截断" if border_rejected else None,
                measurement=physical_measurement,
                scale=scale,
            )
        if border_rejected:
            counts["border_rejected"] += 1
            continue
        if detection is None:
            counts["shape_rejected"] += 1
            continue

        if detection["type"] == "Scratch":
            size_name, size_mm = "width", physical_measurement.width_mm
        else:
            size_name, size_mm = "diameter", physical_measurement.diameter_mm
        if not physical_measurement.measurable:
            counts["unmeasurable"] += 1
            print(f"[MEASURE C{contour_id}] {detection['type']} 不可测: {physical_measurement.reason}")
            continue
        print(
            f"[MEASURE C{contour_id}] {detection['type']} "
            f"{size_name}={size_mm:.6f} mm "
            + (f"width_px={physical_measurement.width_px:.3f} " if detection["type"] == "Scratch" else "")
            + f"valid={physical_measurement.valid}"
        )
        if not physical_measurement.valid:
            counts["below_size"] += 1
            continue
        draw_detection(result_image, detection, physical_measurement, roi_x, roi_y)
        counts[detection["type"]] += 1

    for contour_id, contour in enumerate(stain_contours):
        border_rejected = touches_detection_boundary(contour, detection_area)
        detection = None if border_rejected else detect_stain(contour, stain_mask.shape, scale)
        physical_measurement = (
            measure_pit_contour(contour, calibration)
            if detection is not None else None
        )
        if context is not None:
            record_stain_contour(
                context, contour_id, contour, detection, physical_measurement,
                rejection_reason=(
                    "触及 ROI 有效检测区边界，轮廓可能被截断"
                    if border_rejected else stain_rejection_reason(contour, stain_mask.shape, scale)
                ) if detection is None else None,
            )
        if detection is None:
            counts["stain_rejected"] += 1
            continue
        if not physical_measurement.measurable:
            counts["stain_unmeasurable"] += 1
            print(f"[MEASURE S{contour_id}] Stain 不可测: {physical_measurement.reason}")
            continue
        print(
            f"[MEASURE S{contour_id}] Stain span={physical_measurement.diameter_mm:.6f} mm "
            f"valid={physical_measurement.valid}"
        )
        if not physical_measurement.valid:
            counts["stain_below_size"] += 1
            continue
        draw_detection(
            result_image, detection, physical_measurement, roi_x, roi_y,
            label=f"S{contour_id}",
        )
        counts["Stain"] += 1

    intermediate_images = {
        "Gray": processed["gray"],
        "Small Scale": processed["small_scale"],
        "Background": processed["large_scale"],
        "Enhanced": processed["enhanced"],
        "Binary Mask": mask,
        "Clean Mask": clean,
        "Stain Mask": stain_mask,
        "Detection Area": detection_area,
        "Shaft Mask": shaft_mask,
    }
    debug_images = {}
    if context is not None:
        finish_debug_run(
            context, len(raw_contours), len(contours), len(stain_contours),
            save_images=save_outputs,
        )
        debug_images = {
            "Mask Contours": context.mask_contours,
            "Clean Contours": context.clean_contours,
            "Rejected Contours": context.rejected_contours,
            "Stain Contours": context.stain_contours,
        }
    if run_directory is not None:
        write_image(run_directory / RESULT_PATH.name, result_image)
        write_image(run_directory / MASK_PATH.name, clean)
        write_image(run_directory / DEFAULT_SHAFT_MASK_PATH.name, shaft_mask)
        for number, (name, array) in enumerate(intermediate_images.items(), start=1):
            write_image(run_directory / f"{number:02d}_{name.lower().replace(' ', '_')}.png", array)
        print("[PIPELINE] 图片和日志保存目录:", run_directory)
    print(
        f"[SUMMARY] Raw={len(raw_contours)} Clean={len(contours)} "
        f"有效区边界拒绝={counts['border_rejected']} 形状拒绝={counts['shape_rejected']} "
        f"不可测={counts['unmeasurable']} "
        f"尺寸不足={counts['below_size']} Scratch={counts['Scratch']} Pit={counts['Pit']} "
        f"StainRaw={len(stain_contours)} 污渍形状拒绝={counts['stain_rejected']} "
        f"污渍不可测={counts['stain_unmeasurable']} "
        f"污渍尺寸不足={counts['stain_below_size']} Stain={counts['Stain']}"
    )
    print("[PIPELINE] 检测完成")
    return PipelineResult(
        result_image=result_image,
        intermediate_images=intermediate_images,
        debug_images=debug_images,
        run_directory=run_directory,
    )
