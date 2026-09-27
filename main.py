import cv2
import numpy as np

from config import (
    DEBUG_ALL_CONTOURS,
    IMAGE_PATH,
    MASK_PATH,
    PIT_COLOR,
    PIXEL_TO_MM_SCALE,
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


def draw_detection(result, detection, roi_x, roi_y):
    defect_type = detection["type"]
    size_px = detection["size_px"]
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
            "width =", detection["width_px"],
            "ratio =", detection["aspect_ratio"],
        )
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
            "circularity =", detection["circularity"],
        )

    if PIXEL_TO_MM_SCALE is not None:
        text = f"{defect_type}: {size_px * PIXEL_TO_MM_SCALE:.2f} mm"
    else:
        text = f"{defect_type}: {size_px:.1f} px"

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
    roi_x, roi_y, roi_w, roi_h = cv2.selectROI(
        "Select ROI",
        image,
        False,
        False,
    )

    if roi_w == 0 or roi_h == 0:
        print("没有选择 ROI")
        return

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

        if detection is not None:
            draw_detection(result, detection, roi_x, roi_y)

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
