import cv2
import numpy as np
#.\.venv\Scripts\python.exe -m gui
from config import (
    DEBUG_ALL_CONTOURS,
    IMAGE_PATH,
    MASK_PATH,
    PIT_COLOR,
    RESULT_PATH,
    SCRATCH_COLOR,
    STAIN_COLOR,
    TEXT_COLOR,
    pixel_scale,
)
from debug_utils import (
    finish_debug_run,
    record_clean_contour,
    record_stain_contour,
    start_debug_run,
)
from detector import (
    detect_pit, detect_scratch, detect_stain, measure_scratch_width,
    stain_rejection_reason, touches_detection_boundary,
)
from measurement import (
    DEFAULT_SHAFT_MASK_PATH,
    calibrate_shaft_mask,
    create_shaft_mask_from_roi,
    measure_pit_contour,
    measure_scratch_contour,
    print_calibration,
    prompt_diameter_mm,
)
from preprocess import create_detection_area, preprocess_image


def read_image(path):
    """读取图片，兼容 Windows 中文路径。"""
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def write_image(path, image):
    """保存图片，兼容 Windows 中文路径。"""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"图片编码失败: {path}")

    encoded.tofile(str(path))


def draw_detection(result, detection, measurement, roi_x, roi_y, label=None):
    defect_type = detection["type"]
    center_x, center_y = detection["center"]
    draw_x = int(center_x + roi_x)
    draw_y = int(center_y + roi_y)

    if defect_type == "Scratch":
        box = detection["box"].copy()
        box[:, 0] += roi_x
        box[:, 1] += roi_y
        cv2.drawContours(result, [box], 0, SCRATCH_COLOR, 2)
        print(
            "Scratch:",
            "area =", detection["area"],
            "width_px =", measurement.width_px,
            "skeleton_median_px =", detection["width_px"],
            "width_mm =", measurement.width_mm,
            "ratio =", detection["aspect_ratio"],
        )
        size_mm = measurement.width_mm
    elif defect_type == "Pit":
        cv2.circle(
            result,
            (draw_x, draw_y),
            max(1, int(detection["radius"])),
            PIT_COLOR,
            2,
        )
        print(
            "Pit:",
            "area =", detection["area"],
            "diameter_mm =", measurement.diameter_mm,
            "circularity =", detection["circularity"],
        )
        size_mm = measurement.diameter_mm
    elif defect_type == "Stain":
        outline = detection["contour"].copy()
        outline[:, 0, 0] += roi_x
        outline[:, 0, 1] += roi_y
        cv2.drawContours(result, [outline], -1, STAIN_COLOR, 2)
        size_mm = measurement.diameter_mm
        print(
            "Stain:", "area =", detection["area"],
            "span_mm =", size_mm,
        )
    else:
        raise ValueError(f"未知缺陷类型: {defect_type}")

    text = (
        f"{label}: {size_mm:.2f} mm"
        if defect_type == "Stain" and label is not None
        else f"{defect_type}: {size_mm:.2f} mm"
    )

    cv2.putText(
        result,
        text,
        (draw_x, draw_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 if defect_type == "Stain" else 0.6,
        STAIN_COLOR if defect_type == "Stain" else TEXT_COLOR,
        1 if defect_type == "Stain" else 2,
    )


def main():
    image = read_image(IMAGE_PATH)

    if image is None:
        print("图片读取失败:", IMAGE_PATH)
        return

    result = image.copy()
    print("请框选钢轴：左右边缘贴紧钢轴两侧，上下覆盖主体即可")
    roi_x, roi_y, roi_w, roi_h = cv2.selectROI(
        "Select ROI",
        image,
        False,
        False,
    )

    if roi_w == 0 or roi_h == 0:
        print("没有选择 ROI")
        return

    shaft_mask = create_shaft_mask_from_roi(
        image,
        (roi_x, roi_y, roi_w, roi_h),
    )
    write_image(DEFAULT_SHAFT_MASK_PATH, shaft_mask)
    diameter_mm = prompt_diameter_mm()
    calibration = calibrate_shaft_mask(shaft_mask, diameter_mm)
    print_calibration(calibration)
    scale = pixel_scale(calibration.diameter_px / diameter_mm)
    detection_area = create_detection_area(shaft_mask.shape, calibration)
    print(f"[SCALE] px/mm={scale.px_per_mm:.4f} min_width_px={scale.min_width_px:.2f}")

    crop = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    processed = preprocess_image(crop, scale, detection_area)
    mask = processed["mask"]
    clean = processed["clean"]
    stain_mask = processed["stain_mask"]

    raw_contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    stain_contours, _ = cv2.findContours(
        stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    debug_context = None
    if DEBUG_ALL_CONTOURS:
        debug_context = start_debug_run(mask, crop, raw_contours)

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

        if debug_context is not None:
            record_clean_contour(
                debug_context,
                contour_id,
                contour,
                detection,
                width_info,
                rejection_reason="触及 ROI 有效检测区边界，轮廓可能被截断" if border_rejected else None,
                measurement=physical_measurement,
                scale=scale,
            )

        if detection is None:
            continue

        if detection["type"] == "Scratch":
            size_name = "width"
            size_mm = physical_measurement.width_mm
        else:
            size_name = "diameter"
            size_mm = physical_measurement.diameter_mm

        if not physical_measurement.measurable:
            print(
                f"[MEASURE C{contour_id}] {detection['type']} "
                f"不可测: {physical_measurement.reason}"
            )
            continue

        print(
            f"[MEASURE C{contour_id}] {detection['type']} "
            f"{size_name}={size_mm:.6f} mm "
            f"valid={physical_measurement.valid}"
        )

        if not physical_measurement.valid:
            continue

        draw_detection(
            result,
            detection,
            physical_measurement,
            roi_x,
            roi_y,
        )

    for contour_id, contour in enumerate(stain_contours):
        border_rejected = touches_detection_boundary(contour, detection_area)
        detection = None if border_rejected else detect_stain(contour, stain_mask.shape, scale)
        physical_measurement = (
            measure_pit_contour(contour, calibration)
            if detection is not None else None
        )
        if debug_context is not None:
            record_stain_contour(
                debug_context, contour_id, contour, detection, physical_measurement,
                rejection_reason=(
                    "触及 ROI 有效检测区边界，轮廓可能被截断"
                    if border_rejected else stain_rejection_reason(contour, stain_mask.shape, scale)
                ) if detection is None else None,
            )
        if detection is None:
            continue
        if not physical_measurement.measurable:
            print(f"[MEASURE S{contour_id}] Stain 不可测: {physical_measurement.reason}")
            continue
        print(
            f"[MEASURE S{contour_id}] Stain span={physical_measurement.diameter_mm:.6f} mm "
            f"valid={physical_measurement.valid}"
        )
        if physical_measurement.valid:
            draw_detection(
                result, detection, physical_measurement, roi_x, roi_y,
                label=f"S{contour_id}",
            )

    if debug_context is not None:
        finish_debug_run(
            debug_context,
            len(raw_contours),
            len(contours),
            len(stain_contours),
        )

    cv2.imshow("Gray", processed["gray"])
    cv2.imshow("Enhanced", processed["enhanced"])
    cv2.imshow("Mask", mask)
    cv2.imshow("Clean Mask", clean)
    cv2.imshow("Result", result)

    write_image(RESULT_PATH, result)
    write_image(MASK_PATH, clean)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
