"""片状颜色异常与原有划痕/凹点分开检测。"""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import debug_utils
from config import pixel_scale
from detector import detect_stain, stain_rejection_reason
import main as cli
from pipeline import run_pipeline
from preprocess import preprocess_image


class StainTests(unittest.TestCase):
    def test_brown_patch_detected_but_top_reflection_masked(self):
        image = np.full((260, 400, 3), 120, dtype=np.uint8)
        cv2.circle(image, (200, 140), 30, (40, 90, 145), cv2.FILLED)
        cv2.rectangle(image, (170, 0), (290, 17), (10, 10, 210), cv2.FILLED)

        scale = pixel_scale(400 / 30)
        mask = preprocess_image(image, scale)["stain_mask"]
        self.assertEqual(mask[10, 230], 0)
        self.assertGreater(np.count_nonzero(mask[100:180, 160:240]), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(any(detect_stain(contour, mask.shape, scale) for contour in contours))

        with redirect_stdout(StringIO()):
            result = run_pipeline(image, (0, 0, 400, 260), 30, save_outputs=False)
        self.assertIn("Stain=1", result.logs)
        self.assertIn("[STAIN S", result.logs)
        self.assertIn("Stain Mask", result.intermediate_images)
        self.assertIn("Stain Contours", result.debug_images)
        self.assertGreater(np.count_nonzero(result.result_image != image), 0)

    def test_stain_rejection_explains_small_and_clipped_regions(self):
        small = np.array([[[100, 100]], [[103, 100]], [[103, 103]], [[100, 103]]], dtype=np.int32)
        clipped = np.array([[[100, 31]], [[180, 31]], [[180, 65]], [[100, 65]]], dtype=np.int32)
        scale = pixel_scale(400 / 30)
        self.assertIn("area", stain_rejection_reason(small, (260, 400), scale))
        self.assertIn("边界", stain_rejection_reason(clipped, (260, 400), scale))

    def test_achromatic_dark_patch_uses_texture_branch(self):
        image = np.full((260, 400, 3), 120, dtype=np.uint8)
        cv2.circle(image, (200, 140), 30, (45, 45, 45), cv2.FILLED)
        scale = pixel_scale(400 / 30)
        mask = preprocess_image(image, scale)["stain_mask"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(any(detect_stain(contour, mask.shape, scale) for contour in contours))
        with redirect_stdout(StringIO()):
            result = run_pipeline(image, (0, 0, 400, 260), 30, save_outputs=False)
        self.assertIn("Stain=1", result.logs)

    def test_cli_and_pipeline_draw_the_same_stain_result(self):
        image = np.full((260, 400, 3), 120, dtype=np.uint8)
        cv2.circle(image, (200, 140), 30, (40, 90, 145), cv2.FILLED)
        saved = {}
        with (
            patch.object(cli, "read_image", return_value=image),
            patch.object(cli.cv2, "selectROI", return_value=(0, 0, 400, 260)),
            patch.object(cli, "prompt_diameter_mm", return_value=30),
            patch.object(cli, "write_image", side_effect=lambda path, array: saved.update({path.name: array.copy()})),
            patch.object(cli, "start_debug_run", side_effect=lambda *args: debug_utils.start_debug_run(*args, save_images=False)),
            patch.object(cli, "finish_debug_run", side_effect=lambda *args: debug_utils.finish_debug_run(*args, save_images=False)),
            patch.object(cli.cv2, "imshow"),
            patch.object(cli.cv2, "waitKey", return_value=0),
            patch.object(cli.cv2, "destroyAllWindows"),
            redirect_stdout(StringIO()),
        ):
            cli.main()
        with redirect_stdout(StringIO()):
            result = run_pipeline(image, (0, 0, 400, 260), 30, save_outputs=False)
        np.testing.assert_array_equal(result.result_image, saved[cli.RESULT_PATH.name])


if __name__ == "__main__":
    unittest.main()
