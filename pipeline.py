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

from config import DEBUG_ALL_CONTOURS, DEBUG_OUTPUT_ROOT, MASK_PATH, MIN_DEFECT_SIZE_MM, RESULT_PATH
from debug_utils import finish_debug_run, record_clean_contour, start_debug_run
from detector import detect_pit, detect_scratch, measure_scratch_width
from main import draw_detection, write_image
from measurement import (
    DEFAULT_SHAFT_MASK_PATH,
    calibrate_shaft_mask,
    create_shaft_mask_from_roi,
    measure_pit_contour,
    measure_scratch_contour,
    print_calibration,
)
from preprocess import preprocess_image


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

    result_image = image.copy()
    crop = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    processed = preprocess_image(crop)
    mask, clean = processed["mask"], processed["clean"]
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    counts = {"shape_rejected": 0, "unmeasurable": 0, "below_size": 0, "Scratch": 0, "Pit": 0}
    for contour_id, contour in enumerate(contours):
        width_info = measure_scratch_width(contour)
        detection = detect_scratch(contour, width_info=width_info)
        if detection is None:
            detection = detect_pit(contour)
        if context is not None:
            record_clean_contour(context, contour_id, contour, detection, width_info)
        if detection is None:
            counts["shape_rejected"] += 1
            continue

        if detection["type"] == "Scratch":
            physical_measurement = measure_scratch_contour(contour, calibration)
            size_name, size_mm = "width", physical_measurement.width_mm
        else:
            physical_measurement = measure_pit_contour(contour, calibration)
            size_name, size_mm = "diameter", physical_measurement.diameter_mm
        if not physical_measurement.measurable:
            counts["unmeasurable"] += 1
            print(f"[MEASURE C{contour_id}] {detection['type']} 不可测: {physical_measurement.reason}")
            continue
        print(
            f"[MEASURE C{contour_id}] {detection['type']} "
            f"{size_name}={size_mm:.6f} mm valid={physical_measurement.valid}"
        )
        if not physical_measurement.valid:
            counts["below_size"] += 1
            continue
        draw_detection(result_image, detection, physical_measurement, roi_x, roi_y)
        counts[detection["type"]] += 1

    intermediate_images = {
        "Gray": processed["gray"],
        "Small Scale": processed["small_scale"],
        "Background": processed["large_scale"],
        "Enhanced": processed["enhanced"],
        "Binary Mask": mask,
        "Clean Mask": clean,
        "Shaft Mask": shaft_mask,
    }
    debug_images = {}
    if context is not None:
        finish_debug_run(context, len(raw_contours), len(contours), save_images=save_outputs)
        debug_images = {
            "Mask Contours": context.mask_contours,
            "Clean Contours": context.clean_contours,
            "Rejected Contours": context.rejected_contours,
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
        f"形状拒绝={counts['shape_rejected']} 不可测={counts['unmeasurable']} "
        f"尺寸不足={counts['below_size']} Scratch={counts['Scratch']} Pit={counts['Pit']}"
    )
    print("[PIPELINE] 检测完成")
    return PipelineResult(
        result_image=result_image,
        intermediate_images=intermediate_images,
        debug_images=debug_images,
        run_directory=run_directory,
    )
