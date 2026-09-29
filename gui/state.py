"""当前输入与最近成功输出分开保存，图片始终使用原始分辨率。"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from pipeline import PipelineResult as OutputState


@dataclass
class InputState:
    current_image: np.ndarray | None = None
    current_image_path: Path | None = None
    current_roi: tuple[int, int, int, int] | None = None
    diameter_mm: float | None = None

    def validate(self):
        if self.current_image is None:
            raise ValueError("请先导入钢轴图片。")
        if self.current_roi is None:
            raise ValueError("请先选择钢轴 ROI。")
        height, width = self.current_image.shape[:2]
        x, y, roi_width, roi_height = self.current_roi
        if (
            x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0
            or x + roi_width > width or y + roi_height > height
        ):
            raise ValueError("ROI 必须位于原图内部且宽高大于 0，请重新选择。")
        if self.diameter_mm is None or not math.isfinite(self.diameter_mm) or self.diameter_mm <= 0:
            raise ValueError("请输入合法的钢轴真实直径，单位 mm，且必须大于 0。")


def parse_diameter(text: str) -> float:
    try:
        diameter = float(text.strip())
    except ValueError as error:
        raise ValueError("请输入数字形式的钢轴真实直径，例如 30 或 30.0。") from error
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("钢轴真实直径必须是大于 0 的有限数值。")
    return diameter


