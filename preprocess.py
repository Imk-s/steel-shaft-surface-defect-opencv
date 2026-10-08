import cv2
import numpy as np

from config import (
    ADAPTIVE_C,
    CONNECT_MIN_ASPECT_RATIO,
    DETECTION_RADIUS_RATIO,
    DETECTION_VERTICAL_MARGIN_RATIO,
    STAIN_BOTTOM_MARGIN_RATIO,
    STAIN_DARK_MAX_ASPECT_RATIO,
    STAIN_DARK_MAX_COLOR_OVERLAP,
    STAIN_DARK_RESIDUAL,
    STAIN_DARK_TEXTURE_THRESHOLD,
    STAIN_SIDE_MARGIN_RATIO,
    STAIN_STRONG_CHROMA_DELTA,
    STAIN_TEXTURE_THRESHOLD,
    STAIN_TEXTURE_TOP_RATIO,
    STAIN_TOP_MARGIN_RATIO,
    STAIN_WEAK_CHROMA_DELTA,
    MEASURABLE_RADIUS_RATIO,
)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dual_scale_enhance(gray, scale):
    """小尺度抑制噪声，大尺度估计背景，同时增强亮暗缺陷。"""
    small_scale = cv2.GaussianBlur(gray, (0, 0), scale.small_sigma_px)
    large_scale = cv2.GaussianBlur(gray, (0, 0), scale.large_sigma_px)
    enhanced = cv2.absdiff(small_scale, large_scale)
    return small_scale, large_scale, enhanced


def create_adaptive_mask(enhanced, scale):
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        scale.adaptive_block_px,
        ADAPTIVE_C,
    )


def clean_mask(mask, scale):
    """轻度去噪、删除小连通域，只连接足够细长的水平/竖直候选。"""
    kernel_size = scale.denoise_kernel_px
    foreground = (mask > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        foreground, cv2.CV_16U, np.ones((kernel_size, kernel_size), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    denoised = np.where(
        (foreground > 0) & (neighbors >= min(3, kernel_size * kernel_size)), 255, 0,
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(denoised, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= scale.min_component_area_px
    keep[0] = False
    filtered = keep[labels].astype(np.uint8) * 255
    if scale.connect_gap_px <= 1 or count <= 1:
        return filtered

    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    vertical = keep & (heights >= scale.connect_min_length_px) & (
        heights >= CONNECT_MIN_ASPECT_RATIO * widths
    )
    horizontal = keep & (widths >= scale.connect_min_length_px) & (
        widths >= CONNECT_MIN_ASPECT_RATIO * heights
    )
    vertical[0] = horizontal[0] = False
    vertical_mask = vertical[labels].astype(np.uint8) * 255
    horizontal_mask = horizontal[labels].astype(np.uint8) * 255
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale.connect_gap_px))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale.connect_gap_px, 1))
    connected_vertical = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, vertical_kernel)
    connected_horizontal = cv2.morphologyEx(horizontal_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    return cv2.bitwise_or(filtered, cv2.bitwise_or(connected_vertical, connected_horizontal))


def create_detection_area(shape, calibration):
    height, width = shape[:2]
    _, y0, _, shaft_height = calibration.bounding_box
    margin = int(np.ceil(shaft_height * DETECTION_VERTICAL_MARGIN_RATIO))
    left = max(0, int(np.ceil(calibration.center_x - DETECTION_RADIUS_RATIO * calibration.radius_px)))
    right = min(width, int(np.floor(calibration.center_x + DETECTION_RADIUS_RATIO * calibration.radius_px)) + 1)
    top = max(0, y0 + margin)
    bottom = min(height, y0 + shaft_height - margin)
    if left >= right or top >= bottom:
        raise ValueError("ROI 有效检测区为空，请扩大 ROI 或检查钢轴标定")
    area = np.zeros((height, width), dtype=np.uint8)
    area[top:bottom, left:right] = 255
    return area


def create_stain_mask(image, enhanced, small_scale, scale):
    """提取片状颜色/纹理异常；不改变划痕、凹点使用的二值 Mask。"""
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    sigma = scale.stain_background_sigma_px
    background = cv2.GaussianBlur(lab, (0, 0), sigma)
    chroma_delta = np.max(
        cv2.absdiff(lab[:, :, 1:3], background[:, :, 1:3]), axis=2
    )

    window = scale.stain_texture_window_px
    response = enhanced.astype(np.float32)
    local_mean = cv2.blur(response, (window, window))
    local_variance = cv2.blur(response * response, (window, window)) - local_mean * local_mean
    texture = cv2.sqrt(np.maximum(local_variance, 0))
    strong_color = chroma_delta > STAIN_STRONG_CHROMA_DELTA
    textured_color = (
        (chroma_delta > STAIN_WEAK_CHROMA_DELTA)
        & (texture > STAIN_TEXTURE_THRESHOLD)
    )
    textured_color[:int(height * STAIN_TEXTURE_TOP_RATIO)] = False
    stain_mask = np.where(strong_color | textured_color, 255, 0).astype(np.uint8)

    stain_mask[:int(height * STAIN_TOP_MARGIN_RATIO)] = 0
    stain_mask[int(height * (1 - STAIN_BOTTOM_MARGIN_RATIO)):] = 0
    side = int(width * STAIN_SIDE_MARGIN_RATIO)
    stain_mask[:, :side] = 0
    stain_mask[:, width - side:] = 0

    kernel_size = scale.stain_close_kernel_px
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (scale.stain_open_kernel_px, scale.stain_open_kernel_px)
    )
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, close_kernel)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, open_kernel)
    # 形态学运算后再次裁边，防止轮廓穿进排除的边缘区域。
    stain_mask[:int(height * STAIN_TOP_MARGIN_RATIO)] = 0
    stain_mask[int(height * (1 - STAIN_BOTTOM_MARGIN_RATIO)):] = 0
    stain_mask[:, :side] = 0
    stain_mask[:, width - side:] = 0

    # 无明显色差的深色污渍：只保留较大的非线状纹理区域，避免把每道细划痕
    # 或照明阴影当作片状污渍。
    gray_background = cv2.GaussianBlur(to_grayscale(image), (0, 0), sigma)
    dark_residual = small_scale.astype(np.int16) - gray_background.astype(np.int16)
    dark_mask = np.where(
        (dark_residual < -STAIN_DARK_RESIDUAL)
        & (texture > STAIN_DARK_TEXTURE_THRESHOLD), 255, 0,
    ).astype(np.uint8)
    dark_mask[:int(height * STAIN_TEXTURE_TOP_RATIO)] = 0
    dark_mask[int(height * (1 - STAIN_BOTTOM_MARGIN_RATIO)):] = 0
    dark_mask[:, :side] = 0
    dark_mask[:, width - side:] = 0
    dark_kernel_size = scale.stain_dark_close_kernel_px
    dark_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dark_kernel_size, dark_kernel_size)
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, dark_kernel)
    dark_mask[:int(height * STAIN_TEXTURE_TOP_RATIO)] = 0
    dark_mask[int(height * (1 - STAIN_BOTTOM_MARGIN_RATIO)):] = 0
    dark_mask[:, :side] = 0
    dark_mask[:, width - side:] = 0
    dark_contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    dark_min_area = scale.stain_dark_min_area_px
    for contour in dark_contours:
        if cv2.contourArea(contour) < dark_min_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if y <= int(height * STAIN_TEXTURE_TOP_RATIO):
            continue
        if max(box_width, box_height) / min(box_width, box_height) > STAIN_DARK_MAX_ASPECT_RATIO:
            continue
        center_x, _ = cv2.minEnclosingCircle(contour)[0]
        if abs(center_x - width / 2) > MEASURABLE_RADIUS_RATIO * width / 2:
            continue
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y
        local_region = np.zeros((box_height, box_width), dtype=np.uint8)
        cv2.drawContours(local_region, [local_contour], -1, 255, cv2.FILLED)
        overlap = cv2.countNonZero(cv2.bitwise_and(
            local_region, stain_mask[y:y + box_height, x:x + box_width]
        )) / cv2.countNonZero(local_region)
        if overlap < STAIN_DARK_MAX_COLOR_OVERLAP:
            cv2.drawContours(stain_mask, [contour], -1, 255, cv2.FILLED)
    return stain_mask


def preprocess_image(image, scale, detection_area=None):
    gray = to_grayscale(image)
    small_scale, large_scale, enhanced = dual_scale_enhance(gray, scale)
    mask = create_adaptive_mask(enhanced, scale)
    if detection_area is not None:
        mask = cv2.bitwise_and(mask, detection_area)
    clean = clean_mask(mask, scale)
    stain_mask = create_stain_mask(image, enhanced, small_scale, scale)
    if detection_area is not None:
        clean = cv2.bitwise_and(clean, detection_area)
        stain_mask = cv2.bitwise_and(stain_mask, detection_area)

    return {
        "gray": gray,
        "small_scale": small_scale,
        "large_scale": large_scale,
        "enhanced": enhanced,
        "mask": mask,
        "clean": clean,
        "stain_mask": stain_mask,
    }
