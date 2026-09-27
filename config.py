import cv2
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 输入输出
IMAGE_PATH = PROJECT_ROOT / "testpicture" / "MVIMG_20260923_183352.jpg"
RESULT_PATH = PROJECT_ROOT / "result.jpg"
MASK_PATH = PROJECT_ROOT / "mask.jpg"

# Debug
DEBUG_ALL_CONTOURS = True
DEBUG_OUTPUT_ROOT = PROJECT_ROOT / "debug" / "_picture"

# 双尺度增强
SMALL_SIGMA = 1.0
LARGE_SIGMA = 15.0

# 局部自适应阈值
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = -4

# 形态学核
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))

# 划痕判定
SCRATCH_MIN_AREA = 20
SCRATCH_MIN_LENGTH = 10
SCRATCH_MAX_WIDTH = 40
SCRATCH_MIN_ASPECT_RATIO = 2.5

# 凹点判定
PIT_MIN_AREA = 32
PIT_MIN_CIRCULARITY = 0.55
PIT_MAX_ASPECT_RATIO = 5.0

# 像素到毫米的标定比例，未标定时保持 None
PIXEL_TO_MM_SCALE = None

# 绘制颜色（BGR）
SCRATCH_COLOR = (0, 0, 255)
PIT_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 255, 0)
DEBUG_SCRATCH_COLOR = (0, 0, 255)
DEBUG_PIT_COLOR = (0, 255, 0)
DEBUG_REJECTED_COLOR = (0, 255, 255)
