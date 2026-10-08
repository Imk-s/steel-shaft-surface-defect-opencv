from dataclasses import dataclass
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 输入输出
IMAGE_PATH = PROJECT_ROOT / "testpicture" / "max.jpg"
RESULT_PATH = PROJECT_ROOT / "result.jpg"
MASK_PATH = PROJECT_ROOT / "mask.jpg"

# Debug
DEBUG_ALL_CONTOURS = True
DEBUG_OUTPUT_ROOT = PROJECT_ROOT / "debug" / "_picture"

# 双尺度增强
SMALL_SIGMA_MM = 0.02
LARGE_SIGMA_MM = 0.30

# 局部自适应阈值
ADAPTIVE_BLOCK_MM = 0.60
ADAPTIVE_C = -4

# 形态学核
DENOISE_KERNEL_MM = 0.04
MIN_COMPONENT_AREA_MM2 = 0.003
CONNECT_GAP_MM = 0.06
CONNECT_MIN_LENGTH_MM = 0.10
CONNECT_MIN_ASPECT_RATIO = 2.0

# 划痕判定
SCRATCH_MIN_AREA_MM2 = 0.008
SCRATCH_MIN_LENGTH_MM = 0.20
SCRATCH_MAX_WIDTH_MM = 0.80
SCRATCH_MIN_ASPECT_RATIO = 2.5
# 骨架中位宽度仍用于测宽；以下仅防止纹理粘连成片后冒充细长划痕。
SCRATCH_MIN_BOX_ASPECT_RATIO = 1.5
SCRATCH_MAX_MERGE_RATIO = 3.0

# 凹点判定
PIT_MIN_AREA_MM2 = 0.012
PIT_MIN_CIRCULARITY = 0.55
PIT_MAX_ASPECT_RATIO = 5.0

# 片状锈斑/污渍：相对于局部背景的 Lab 色差，辅以纹理变化。
# 比例参数按 ROI 尺寸缩放，避免只适合某一种照片分辨率。
STAIN_BACKGROUND_SIGMA_MM = 1.20
STAIN_TEXTURE_WINDOW_MM = 0.60
STAIN_STRONG_CHROMA_DELTA = 8
STAIN_WEAK_CHROMA_DELTA = 4
STAIN_TEXTURE_THRESHOLD = 10.0
STAIN_TEXTURE_TOP_RATIO = 0.18
STAIN_CLOSE_KERNEL_MM = 0.09
STAIN_OPEN_KERNEL_MM = 0.06
STAIN_DARK_CLOSE_KERNEL_MM = 0.135
STAIN_TOP_MARGIN_RATIO = 0.12
STAIN_BOTTOM_MARGIN_RATIO = 0.05
STAIN_SIDE_MARGIN_RATIO = 0.03
STAIN_MIN_AREA_MM2 = 0.108
STAIN_DARK_RESIDUAL = 25
STAIN_DARK_TEXTURE_THRESHOLD = 10.0
STAIN_DARK_MIN_AREA_MM2 = 0.36
STAIN_DARK_MAX_ASPECT_RATIO = 3.0
STAIN_DARK_MAX_COLOR_OVERLAP = 0.30

# 圆柱投影中央有效测量区域：|x - x_c| <= ratio * R_px
MEASURABLE_RADIUS_RATIO = 0.60
DETECTION_RADIUS_RATIO = MEASURABLE_RADIUS_RATIO
DETECTION_VERTICAL_MARGIN_RATIO = 0.04
MIN_DEFECT_SIZE_MM = 0.2
SCRATCH_PCA_RADIUS_MM = 0.20


@dataclass(frozen=True)
class PixelScale:
    px_per_mm: float
    min_width_px: float
    small_sigma_px: float
    large_sigma_px: float
    adaptive_block_px: int
    denoise_kernel_px: int
    min_component_area_px: float
    connect_gap_px: int
    connect_min_length_px: float
    scratch_min_area_px: float
    scratch_min_length_px: float
    scratch_max_width_px: float
    pit_min_area_px: float
    stain_background_sigma_px: float
    stain_texture_window_px: int
    stain_close_kernel_px: int
    stain_open_kernel_px: int
    stain_dark_close_kernel_px: int
    stain_min_area_px: float
    stain_dark_min_area_px: float
    scratch_pca_radius_px: float


def _odd_pixels(value, minimum=3):
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def pixel_scale(px_per_mm):
    if not math.isfinite(px_per_mm) or px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive and finite")
    return PixelScale(
        px_per_mm=px_per_mm,
        min_width_px=MIN_DEFECT_SIZE_MM * px_per_mm,
        small_sigma_px=SMALL_SIGMA_MM * px_per_mm,
        large_sigma_px=LARGE_SIGMA_MM * px_per_mm,
        adaptive_block_px=_odd_pixels(ADAPTIVE_BLOCK_MM * px_per_mm),
        denoise_kernel_px=_odd_pixels(DENOISE_KERNEL_MM * px_per_mm, minimum=1),
        min_component_area_px=max(1.0, MIN_COMPONENT_AREA_MM2 * px_per_mm**2),
        connect_gap_px=_odd_pixels(CONNECT_GAP_MM * px_per_mm, minimum=1),
        connect_min_length_px=CONNECT_MIN_LENGTH_MM * px_per_mm,
        scratch_min_area_px=max(1.0, SCRATCH_MIN_AREA_MM2 * px_per_mm**2),
        scratch_min_length_px=SCRATCH_MIN_LENGTH_MM * px_per_mm,
        scratch_max_width_px=SCRATCH_MAX_WIDTH_MM * px_per_mm,
        pit_min_area_px=max(1.0, PIT_MIN_AREA_MM2 * px_per_mm**2),
        stain_background_sigma_px=STAIN_BACKGROUND_SIGMA_MM * px_per_mm,
        stain_texture_window_px=_odd_pixels(STAIN_TEXTURE_WINDOW_MM * px_per_mm),
        stain_close_kernel_px=_odd_pixels(STAIN_CLOSE_KERNEL_MM * px_per_mm, minimum=1),
        stain_open_kernel_px=_odd_pixels(STAIN_OPEN_KERNEL_MM * px_per_mm, minimum=1),
        stain_dark_close_kernel_px=_odd_pixels(STAIN_DARK_CLOSE_KERNEL_MM * px_per_mm, minimum=1),
        stain_min_area_px=max(1.0, STAIN_MIN_AREA_MM2 * px_per_mm**2),
        stain_dark_min_area_px=max(1.0, STAIN_DARK_MIN_AREA_MM2 * px_per_mm**2),
        scratch_pca_radius_px=SCRATCH_PCA_RADIUS_MM * px_per_mm,
    )

# 像素到毫米的标定比例，未标定时保持 None
PIXEL_TO_MM_SCALE = None

# 绘制颜色（BGR）
SCRATCH_COLOR = (0, 0, 255)
PIT_COLOR = (255, 0, 0)
STAIN_COLOR = (0, 165, 255)
TEXT_COLOR = (0, 255, 0)
DEBUG_SCRATCH_COLOR = (0, 0, 255)
DEBUG_PIT_COLOR = (0, 255, 0)
DEBUG_STAIN_COLOR = (255, 255, 0)
DEBUG_REJECTED_COLOR = (0, 255, 255)
