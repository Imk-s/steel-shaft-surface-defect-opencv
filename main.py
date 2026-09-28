import cv2
import numpy as np

from config import (
    DEBUG_ALL_CONTOURS,
    IMAGE_PATH,
    MASK_PATH,
    PIT_COLOR,
    RESULT_PATH,
    SCRATCH_COLOR,
    TEXT_COLOR,
)
from debug_utils import (
    finish_debug_run,
    record_clean_contour,
    start_debug_run,
)
from detector import detect_pit, detect_scratch, measure_scratch_width
from measurement import (
    DEFAULT_SHAFT_MASK_PATH,
    calibrate_shaft_mask,
    create_shaft_mask_from_roi,
    measure_pit_contour,
    measure_scratch_contour,
    print_calibration,
    prompt_diameter_mm,
)
from preprocess import preprocess_image


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


def draw_detection(result, detection, measurement, roi_x, roi_y):
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
            "width_px =", detection["width_px"],
            "width_mm =", measurement.width_mm,
            "ratio =", detection["aspect_ratio"],
        )
        size_mm = measurement.width_mm
    else:
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

    text = f"{defect_type}: {size_mm:.2f} mm"

    cv2.putText(
        result,
        text,
        (draw_x, draw_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
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

    crop = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    processed = preprocess_image(crop)
    mask = processed["mask"]
    clean = processed["clean"]

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

    debug_context = None
    if DEBUG_ALL_CONTOURS:
        debug_context = start_debug_run(mask, crop, raw_contours)

    for contour_id, contour in enumerate(contours):
        width_info = measure_scratch_width(contour)
        detection = detect_scratch(contour, width_info=width_info)

        if detection is None:
            detection = detect_pit(contour)

        if debug_context is not None:
            record_clean_contour(
                debug_context,
                contour_id,
                contour,
                detection,
                width_info,
            )

        if detection is None:
            continue

        if detection["type"] == "Scratch":
            physical_measurement = measure_scratch_contour(
                contour,
                calibration,
            )
            size_name = "width"
            size_mm = physical_measurement.width_mm
        else:
            physical_measurement = measure_pit_contour(
                contour,
                calibration,
            )
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

    if debug_context is not None:
        finish_debug_run(
            debug_context,
            len(raw_contours),
            len(contours),
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
