import cv2
import numpy as np
import math


def detect_scratch(
    contour,
    min_area=20,
    min_length=10,
    max_width=18,
    min_aspect_ratio=3
):
    """检测单个轮廓是否为划痕，命中时返回划痕参数。"""
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ = rect

    if rw == 0 or rh == 0:
        return None

    long_side = max(rw, rh)
    short_side = min(rw, rh)
    aspect_ratio = long_side / short_side

    if not (
        area >= min_area and
        long_side >= min_length and
        # 弯曲划痕连接附近亮点后，包围框短边会略大于真实线宽
        short_side <= max_width and
        aspect_ratio >= min_aspect_ratio
    ):
        return None

    return {
        "type": "Scratch",
        "size_px": short_side,
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
img = cv2.imread(r"D:\Code\Py\testpicture\min.jpg")

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
# 3. 灰度 + 轻微降噪
# =========================
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)


# =========================
# 4. 暗缺陷增强
# 用大范围模糊估计正常光照背景
# background - gray 保留局部偏暗缺陷
# =========================
background = cv2.GaussianBlur(gray, (0, 0), 15)

enhanced = cv2.subtract(background, gray)

# enhanced = cv2.normalize(
#     enhanced,
#     None,
#     0,
#     255,
#     cv2.NORM_MINMAX
# )


# =========================
# 5. 阈值分割
# Otsu 自动帮我们选阈值
# =========================
_, mask = cv2.threshold(
    enhanced,
    20,
    255,
    cv2.THRESH_BINARY 
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
contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
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

for cnt in contours:
    detection = detect_scratch(cnt)

    # 划痕优先；不满足划痕条件时再检查凹点
    if detection is None:
        detection = detect_pit(cnt)

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
