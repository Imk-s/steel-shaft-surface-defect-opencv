"""尺度、有效区、二值清理及划痕宽度的无 GUI 回归测试。"""

import unittest

import cv2
import numpy as np

from config import MIN_DEFECT_SIZE_MM, pixel_scale
from detector import touches_detection_boundary
from measurement import calibrate_shaft_mask, measure_scratch_contour
from preprocess import clean_mask, create_detection_area


def rectangle_contour(x, y, width, height):
    return np.array(
        [[[x, y]], [[x + width, y]], [[x + width, y + height]], [[x, y + height]]],
        dtype=np.int32,
    )


class ScaledDetectionTests(unittest.TestCase):
    def test_spatial_thresholds_follow_px_per_mm(self):
        small = pixel_scale(50)
        large = pixel_scale(100)
        self.assertEqual(MIN_DEFECT_SIZE_MM, 0.5)
        self.assertAlmostEqual(large.min_width_px, 2 * small.min_width_px)
        self.assertAlmostEqual(large.scratch_min_area_px, 4 * small.scratch_min_area_px)
        self.assertEqual(small.denoise_kernel_px, 3)
        self.assertEqual(large.denoise_kernel_px, 5)
        self.assertGreater(large.connect_gap_px, small.connect_gap_px)

    def test_detection_area_excludes_all_four_sides(self):
        shaft = np.full((100, 200), 255, dtype=np.uint8)
        calibration = calibrate_shaft_mask(shaft, 30)
        area = create_detection_area(shaft.shape, calibration)
        self.assertEqual(area[3, 100], 0)
        self.assertEqual(area[4, 100], 255)
        self.assertEqual(area[96, 100], 0)
        self.assertEqual(area[50, 39], 0)
        self.assertEqual(area[50, 40], 255)
        self.assertEqual(area[50, 159], 255)
        self.assertEqual(area[50, 160], 0)
        self.assertFalse(touches_detection_boundary(rectangle_contour(45, 10, 5, 10), area))
        self.assertTrue(touches_detection_boundary(rectangle_contour(40, 10, 5, 10), area))
        self.assertTrue(touches_detection_boundary(rectangle_contour(45, 4, 5, 10), area))

    def test_clean_removes_small_components_then_bridges_only_lines(self):
        mask = np.zeros((70, 80), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (11, 20), 255, -1)
        cv2.rectangle(mask, (10, 23), (11, 33), 255, -1)
        cv2.rectangle(mask, (40, 20), (44, 24), 255, -1)
        cv2.rectangle(mask, (40, 27), (44, 31), 255, -1)
        cv2.rectangle(mask, (60, 50), (61, 51), 255, -1)
        clean = clean_mask(mask, pixel_scale(50))
        self.assertEqual(clean[22, 10], 255)
        self.assertEqual(clean[26, 42], 0)
        self.assertEqual(clean[50, 60], 0)

    def test_measured_width_is_twice_maximum_skeleton_radius(self):
        shaft = np.full((140, 200), 255, dtype=np.uint8)
        calibration = calibrate_shaft_mask(shaft, 30)
        contour = rectangle_contour(98, 30, 7, 70)
        result = measure_scratch_contour(contour, calibration)
        self.assertTrue(result.measurable)
        self.assertAlmostEqual(result.width_px, 2 * result.max_radius_px, places=6)
        self.assertAlmostEqual(
            result.width_mm, result.width_px * calibration.mm_per_px, delta=0.05
        )


if __name__ == "__main__":
    unittest.main()
