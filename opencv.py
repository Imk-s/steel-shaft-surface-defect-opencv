import cv2
import numpy as np
import math
from datetime import datetime
from pathlib import Path


# 调试开关：输出所有候选轮廓，不经过缺陷分类条件筛选
DEBUG_ALL_CONTOURS = True
DEBUG_OUTPUT_ROOT = Path(r"D:\Code\Py\钢轴项目\debug\_picture")
SCRATCH_MIN_AREA = 20
SCRATCH_MIN_LENGTH = 10
SCRATCH_MAX_WIDTH = 40
SCRATCH_MIN_ASPECT_RATIO = 2.5


def save_debug_image(path, image):
    """通过内存编码保存图片，兼容 Windows 中文路径。"""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"调试图片编码失败: {path}")

    encoded.tofile(str(path))


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
        dtype=np.uint8
    )
    cv2.drawContours(
        contour_mask,
        [local_contour],
        -1,
        255,
        cv2.FILLED
    )

    distance = cv2.distanceTransform(
        contour_mask,
        cv2.DIST_L2,
        5
    )

    skeleton = np.zeros_like(contour_mask)
    work = contour_mask.copy()
    skeleton_kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3)
    )

    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(
            work,
            cv2.MORPH_OPEN,
            skeleton_kernel
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
        "skeleton_pixels": int(radii.size)
    }


def detect_scratch(
    contour,
    min_area=SCRATCH_MIN_AREA,
    min_length=SCRATCH_MIN_LENGTH,
    max_width=SCRATCH_MAX_WIDTH,
    min_aspect_ratio=SCRATCH_MIN_ASPECT_RATIO,
    width_info=None
):
    """检测单个轮廓是否为划痕，命中时返回划痕参数。"""
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ = rect

    if rw == 0 or rh == 0:
        return None

    long_side = max(rw, rh)

    if width_info is None:
        width_info = measure_scratch_width(contour)

    if width_info is None or width_info["width"] <= 0:
        return None

    skeleton_width = width_info["width"]
    aspect_ratio = long_side / skeleton_width

    if not (
        area >= min_area and
        long_side >= min_length and
        skeleton_width <= max_width and
        aspect_ratio >= min_aspect_ratio
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
        "aspect_ratio": aspect_ratio
    }


def detect_pit(
    contour,
    min_area=32,
    min_circularity=0.55,
    max_aspect_ratio=5.0
):
    """检测单个轮廓是否为凹点，命中时返回凹点参数。"""
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
        area >= min_area and
        circularity >= min_circularity and
        aspect_ratio <= max_aspect_ratio
    ):
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    return {
        "type": "Pit",
        "size_px": 2 * radius,
        "center": (cx, cy),
        "radius": radius,
        "area": area,
        "circularity": circularity
    }

# =========================
# 1. 读取图片
# =========================
image_path = r"D:\Code\Py\钢轴项目\testpicture\max.jpg"
img = cv2.imdecode(
    np.fromfile(image_path, dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if img is None:
    print("图片读取失败")
    exit()

result = img.copy()


# =========================
# 2. 手动选择 ROI
# =========================
roi = cv2.selectROI("Select ROI", img, False, False)

x, y, w, h = roi

if w == 0 or h == 0:
    print("没有选择 ROI")
    exit()

crop = img[y:y+h, x:x+w]


# =========================
# 3. 灰度
# =========================
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


# =========================
# 4. 双尺度增强
# 小尺度用于抑制像素噪声，同时保留缺陷细节
# 大尺度用于估计缓慢变化的背景光照
# 绝对差同时保留亮缺陷和暗缺陷的响应
# =========================
small_sigma = 1.0
large_sigma = 15.0

small_scale = cv2.GaussianBlur(
    gray,
    (0, 0),
    small_sigma
)

large_scale = cv2.GaussianBlur(
    gray,
    (0, 0),
    large_sigma
)

enhanced = cv2.absdiff(
    small_scale,
    large_scale
)


# =========================
# 5. 阈值分割
# Otsu 自动帮我们选阈值
# =========================
# _, mask = cv2.threshold(
#     enhanced,
#     20,
#     255,
#     cv2.THRESH_BINARY
# )
mask = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    -4
)

# =========================
# 6. 掩膜清理
# =========================

# 去掉一些孤立的小噪点
open_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (2, 2)
)

clean = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    open_kernel
)

# 连接轻微断开的区域
close_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 11)
)

clean = cv2.morphologyEx(
    clean,
    cv2.MORPH_CLOSE,
    close_kernel
)


# =========================
# 7. 找轮廓
# =========================
raw_contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


if DEBUG_ALL_CONTOURS:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")[:-3]
    debug_run_dir = DEBUG_OUTPUT_ROOT / run_name
    debug_run_dir.mkdir(parents=True, exist_ok=False)

    # M 编号：直接对应原始 mask 上的轮廓
    debug_mask_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for mask_contour_id, raw_cnt in enumerate(raw_contours):
        mask_area = cv2.contourArea(raw_cnt)
        mask_perimeter = cv2.arcLength(raw_cnt, True)
        mbx, mby, mbw, mbh = cv2.boundingRect(raw_cnt)
        mask_label = f"M{mask_contour_id}"
        mask_label_position = (mbx, max(14, mby - 3))

        cv2.drawContours(
            debug_mask_contours,
            [raw_cnt],
            -1,
            (0, 165, 255),
            1
        )
        cv2.putText(
            debug_mask_contours,
            mask_label,
            mask_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            3
        )
        cv2.putText(
            debug_mask_contours,
            mask_label,
            mask_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1
        )

        print(
            f"[MASK {mask_label}] area={mask_area:.1f} "
            f"perimeter={mask_perimeter:.1f} "
            f"bbox=({mbx},{mby},{mbw},{mbh})"
        )

    # C 编号：对应清理后实际送入分类器的轮廓
    debug_clean_contours = crop.copy()

    # 单独显示拒绝项，压暗背景以突出黄色轮廓和编号
    debug_rejected_contours = cv2.convertScaleAbs(
        crop,
        alpha=0.35,
        beta=0
    )


# =========================
# 8. 简单分类
# =========================

# 如果以后完成标定，比如：
# 10 mm = 200 px
# 那么 scale = 10 / 200 = 0.05
#
# 暂时没有标定就写 None
scale = None

for contour_id, cnt in enumerate(contours):
    scratch_width_info = measure_scratch_width(cnt)
    detection = detect_scratch(cnt, width_info=scratch_width_info)

    # 划痕优先；不满足划痕条件时再检查凹点
    if detection is None:
        detection = detect_pit(cnt)

    if DEBUG_ALL_CONTOURS:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        box_aspect_ratio = (
            long_side / short_side
            if short_side > 0 else float("inf")
        )
        skeleton_width = (
            scratch_width_info["width"]
            if scratch_width_info is not None else 0.0
        )
        skeleton_width_min = (
            scratch_width_info["min_width"]
            if scratch_width_info is not None else 0.0
        )
        skeleton_width_max = (
            scratch_width_info["max_width"]
            if scratch_width_info is not None else 0.0
        )
        skeleton_aspect_ratio = (
            long_side / skeleton_width
            if skeleton_width > 0 else float("inf")
        )
        circularity = (
            4 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )
        if detection is not None and detection["type"] == "Scratch":
            debug_type = "Scratch"
            debug_color = (0, 0, 255)
        elif detection is not None and detection["type"] == "Pit":
            debug_type = "Pit"
            debug_color = (0, 255, 0)
        else:
            debug_type = "Rejected"
            debug_color = (0, 255, 255)

        cv2.drawContours(
            debug_clean_contours,
            [cnt],
            -1,
            debug_color,
            1
        )
        clean_label = f"C{contour_id}"
        clean_label_position = (bx, max(10, by - 2))
        cv2.putText(
            debug_clean_contours,
            clean_label,
            clean_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 0),
            2
        )
        cv2.putText(
            debug_clean_contours,
            clean_label,
            clean_label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            debug_color,
            1
        )

        print(
            f"[CLEAN C{contour_id}] type={debug_type} "
            f"area={area:.1f} perimeter={perimeter:.1f} "
            f"bbox=({bx},{by},{bw},{bh}) long={long_side:.1f} "
            f"box_short={short_side:.1f} "
            f"box_aspect={box_aspect_ratio:.3f} "
            f"skeleton_width={skeleton_width:.2f} "
            f"skeleton_range="
            f"{skeleton_width_min:.2f}-{skeleton_width_max:.2f} "
            f"skeleton_aspect={skeleton_aspect_ratio:.3f} "
            f"circularity={circularity:.3f}"
        )

        if detection is None:
            scratch_reasons = []
            pit_reasons = []

            cv2.drawContours(
                debug_rejected_contours,
                [cnt],
                -1,
                (0, 255, 255),
                2
            )
            cv2.rectangle(
                debug_rejected_contours,
                (bx, by),
                (bx + bw, by + bh),
                (0, 255, 255),
                1
            )
            rejected_label_position = (bx, max(14, by - 4))
            cv2.putText(
                debug_rejected_contours,
                f"C{contour_id}",
                rejected_label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                2
            )
            cv2.putText(
                debug_rejected_contours,
                f"C{contour_id}",
                rejected_label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1
            )

            if area < SCRATCH_MIN_AREA:
                scratch_reasons.append(
                    f"area {area:.1f} < {SCRATCH_MIN_AREA}"
                )
            if rw == 0 or rh == 0:
                scratch_reasons.append("旋转包围框存在零边长")
            else:
                if long_side < SCRATCH_MIN_LENGTH:
                    scratch_reasons.append(
                        f"long_side {long_side:.1f} "
                        f"< {SCRATCH_MIN_LENGTH}"
                    )

            if scratch_width_info is None:
                scratch_reasons.append("无法从骨架计算宽度")
            else:
                if skeleton_width > SCRATCH_MAX_WIDTH:
                    scratch_reasons.append(
                        f"skeleton_width {skeleton_width:.2f} "
                        f"> {SCRATCH_MAX_WIDTH}"
                    )
                if skeleton_aspect_ratio < SCRATCH_MIN_ASPECT_RATIO:
                    scratch_reasons.append(
                        f"skeleton_aspect {skeleton_aspect_ratio:.3f} "
                        f"< {SCRATCH_MIN_ASPECT_RATIO}"
                    )

            if area < 32:
                pit_reasons.append(f"area {area:.1f} < 32")
            if perimeter == 0:
                pit_reasons.append("perimeter = 0")
            if rw == 0 or rh == 0:
                pit_reasons.append("旋转包围框存在零边长")
            else:
                if circularity < 0.55:
                    pit_reasons.append(
                        f"circularity {circularity:.3f} < 0.55"
                    )
                if box_aspect_ratio > 5.0:
                    pit_reasons.append(
                        f"aspect_ratio {box_aspect_ratio:.3f} > 5.0"
                    )

            print(
                f"[CLEAN C{contour_id}] REJECT Scratch: "
                + "; ".join(scratch_reasons)
            )
            print(
                f"[CLEAN C{contour_id}] REJECT Pit: "
                + "; ".join(pit_reasons)
            )

    if detection is None:
        continue

    defect_type = detection["type"]
    size_px = detection["size_px"]
    center_x, center_y = detection["center"]
    draw_x = int(center_x + x)
    draw_y = int(center_y + y)

    if defect_type == "Scratch":
        box = detection["box"].copy()
        box[:, 0] += x
        box[:, 1] += y

        cv2.drawContours(
            result,
            [box],
            0,
            (0, 0, 255),
            2
        )

        print(
            "Scratch:",
            "area =", detection["area"],
            "width =", detection["width_px"],
            "ratio =", detection["aspect_ratio"]
        )

    else:
        cv2.circle(
            result,
            (draw_x, draw_y),
            max(1, int(detection["radius"])),
            (255, 0, 0),
            2
        )

        print(
            "Pit:",
            "area =", detection["area"],
            "circularity =", detection["circularity"]
        )


    # ---------- 尺寸文本 ----------
    if scale is not None:

        size_mm = size_px * scale

        text = (
            f"{defect_type}: "
            f"{size_mm:.2f} mm"
        )

    else:

        text = (
            f"{defect_type}: "
            f"{size_px:.1f} px"
        )


    cv2.putText(
        result,
        text,
        (draw_x, draw_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


if DEBUG_ALL_CONTOURS:
    save_debug_image(
        debug_run_dir / "debug_mask_contours.jpg",
        debug_mask_contours
    )
    save_debug_image(
        debug_run_dir / "debug_clean_contours.jpg",
        debug_clean_contours
    )
    save_debug_image(
        debug_run_dir / "debug_rejected_only.jpg",
        debug_rejected_contours
    )

    print(
        "[DEBUG] 阈值轮廓:", len(raw_contours),
        "清理后轮廓:", len(contours)
    )
    print("[DEBUG] 颜色: 红色=Scratch 绿色=Pit 黄色=Rejected")
    print("[DEBUG] 调试图片目录:", debug_run_dir)


# =========================
# 9. 显示中间结果
# =========================
cv2.imshow("Gray", gray)
cv2.imshow("Enhanced", enhanced)
cv2.imshow("Mask", mask)
cv2.imshow("Clean Mask", clean)
cv2.imshow("Result", result)

cv2.imwrite("result.jpg", result)
cv2.imwrite("mask.jpg", clean)

cv2.waitKey(0)
cv2.destroyAllWindows()
