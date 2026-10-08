"""防止边界及聚集纹理被误认为完整、细长的划痕。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from config import MIN_DEFECT_SIZE_MM, pixel_scale
from debug_utils import DebugContext, record_clean_contour
from detector import detect_scratch, touches_detection_boundary, touches_roi_bottom
from preprocess import create_detection_area


def rectangle(x, y, width, height):
    return np.array(
        [[[x, y]], [[x + width, y]], [[x + width, y + height]], [[x, y + height]]],
        dtype=np.int32,
    )


def width_info(width=4.0, maximum=8.0):
    return {"width": width, "mean_width": width, "min_width": 2.0,
            "max_width": maximum, "skeleton_pixels": 10}


class ContourGuardTests(unittest.TestCase):
    def test_size_threshold_is_positive(self):
        self.assertGreater(MIN_DEFECT_SIZE_MM, 0)

    def test_roi_bottom_touch_and_interior(self):
        self.assertTrue(touches_roi_bottom(rectangle(10, 90, 50, 9), 100))
        self.assertFalse(touches_roi_bottom(rectangle(10, 80, 50, 9), 100))

    def test_thin_scratch_kept_but_merged_region_rejected(self):
        scale = pixel_scale(50)
        self.assertIsNotNone(detect_scratch(rectangle(10, 10, 4, 70), scale, width_info()))
        self.assertIsNone(detect_scratch(rectangle(10, 10, 30, 100), scale, width_info(6, 30)))
        self.assertIsNone(detect_scratch(rectangle(10, 10, 4, 70), scale, width_info(4, scale.scratch_max_width_px + 1)))
        self.assertIsNone(detect_scratch(rectangle(10, 10, 24, 24), scale, width_info(12, 20)))

    def test_debug_reports_the_actual_rejection(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        context = DebugContext(Path("unused"), crop.copy(), crop.copy(), crop.copy())
        log = StringIO()
        with redirect_stdout(log):
            record_clean_contour(
                context, 0, rectangle(10, 10, 30, 80), None, width_info(6, 30),
                scale=pixel_scale(50),
            )
            record_clean_contour(
                context, 1, rectangle(10, 90, 30, 9), None, width_info(),
                rejection_reason="触及 ROI 下边界",
                scale=pixel_scale(50),
            )
        self.assertIn("[CLEAN C0] REJECT Scratch: merge_ratio", log.getvalue())
        self.assertIn("[CLEAN C1] REJECT ROI: 触及 ROI 下边界", log.getvalue())


    def test_debug_marks_subthreshold_candidate_as_rejected(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        context = DebugContext(Path("unused"), crop.copy(), crop.copy(), crop.copy())
        below_threshold = MIN_DEFECT_SIZE_MM / 2
        measurement = SimpleNamespace(measurable=True, valid=False, width_mm=below_threshold)
        log = StringIO()
        with redirect_stdout(log):
            record_clean_contour(
                context, 2, rectangle(10, 10, 4, 70),
                {"type": "Scratch"}, width_info(), measurement=measurement,
                scale=pixel_scale(50),
            )
        self.assertIn("[CLEAN C2] type=Rejected candidate=Scratch", log.getvalue())
        self.assertIn("[CLEAN C2] REJECT Scratch: ", log.getvalue())
        self.assertIn(
            f"{below_threshold:.6f} mm < {MIN_DEFECT_SIZE_MM} mm",
            log.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
