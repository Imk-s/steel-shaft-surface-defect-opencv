import cv2

from config import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    CLOSE_KERNEL,
    LARGE_SIGMA,
    OPEN_KERNEL,
    SMALL_SIGMA,
)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dual_scale_enhance(gray):
    """小尺度抑制噪声，大尺度估计背景，同时增强亮暗缺陷。"""
    small_scale = cv2.GaussianBlur(gray, (0, 0), SMALL_SIGMA)
    large_scale = cv2.GaussianBlur(gray, (0, 0), LARGE_SIGMA)
    enhanced = cv2.absdiff(small_scale, large_scale)
    return small_scale, large_scale, enhanced


def create_adaptive_mask(enhanced):
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def clean_mask(mask):
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    return cv2.morphologyEx(clean, cv2.MORPH_CLOSE, CLOSE_KERNEL)


def preprocess_image(image):
    gray = to_grayscale(image)
    small_scale, large_scale, enhanced = dual_scale_enhance(gray)
    mask = create_adaptive_mask(enhanced)
    clean = clean_mask(mask)

    return {
        "gray": gray,
        "small_scale": small_scale,
        "large_scale": large_scale,
        "enhanced": enhanced,
        "mask": mask,
        "clean": clean,
    }
