"""统一 pipeline 与原 CLI 的像素一致性、日志和保存回归。"""

from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import debug_utils
import main as cli
import pipeline


def sample_image():
    image = np.full((320, 480, 3), 145, dtype=np.uint8)
    cv2.rectangle(image, (220, 100), (226, 210), (35, 35, 35), -1)
    cv2.circle(image, (300, 140), 10, (25, 25, 25), -1)
    return image


class PipelineTests(unittest.TestCase):
    def test_matches_original_cli_pixels_and_preserves_original(self):
        image = sample_image()
        original = image.copy()
        roi = (40, 30, 400, 260)
        saved = {}
        with (
            patch.object(cli, "read_image", return_value=image),
            patch.object(cli.cv2, "selectROI", return_value=roi),
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
        terminal = StringIO()
        with (
            redirect_stdout(terminal),
            patch.object(cv2, "imshow", side_effect=AssertionError("GUI 不应调用 HighGUI")),
            patch.object(cv2, "selectROI", side_effect=AssertionError("GUI 不应调用 HighGUI")),
            patch.object(cv2, "waitKey", side_effect=AssertionError("GUI 不应调用 HighGUI")),
        ):
            result = pipeline.run_pipeline(image, roi, 30, save_outputs=False)
        np.testing.assert_array_equal(result.result_image, saved[cli.RESULT_PATH.name])
        np.testing.assert_array_equal(result.intermediate_images["Clean Mask"], saved[cli.MASK_PATH.name])
        np.testing.assert_array_equal(result.intermediate_images["Shaft Mask"], saved[cli.DEFAULT_SHAFT_MASK_PATH.name])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result.logs, terminal.getvalue())
        self.assertIn("[CLEAN C", result.logs)
        self.assertIn("[MEASURE C", result.logs)
        self.assertIn("[SUMMARY]", result.logs)
        self.assertEqual(len(result.intermediate_images), 7)
        self.assertEqual(len(result.debug_images), 3)
        self.assertIsNone(result.run_directory)

    def test_invalid_inputs_rejected_before_any_output(self):
        image = sample_image()
        for roi, diameter in (((0, 0, 0, 10), 30), ((-1, 0, 10, 10), 30), ((0, 0, 900, 10), 30), ((0.5, 0, 10, 10), 30), ((0, 0, 20, 20), 0), ((0, 0, 20, 20), float("nan"))):
            with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                pipeline.run_pipeline(image, roi, diameter, save_outputs=False)

    def test_saved_artifacts_and_disabled_debug(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory) / "中文运行记录"
            with redirect_stdout(StringIO()):
                first = pipeline.run_pipeline(sample_image(), (40, 30, 400, 260), 30, output_root=root)
                second = pipeline.run_pipeline(sample_image(), (40, 30, 400, 260), 30, output_root=root, debug_enabled=False)
            self.assertNotEqual(first.run_directory, second.run_directory)
            self.assertEqual(second.debug_images, {})
            self.assertEqual((first.run_directory / "pipeline.log").read_text(encoding="utf-8"), first.logs)
            for name in ("result.jpg", "mask.jpg", "shaft_mask.jpg", "debug_clean_contours.jpg", "04_enhanced.png"):
                self.assertTrue((first.run_directory / name).is_file())
            self.assertEqual(cli.read_image(first.run_directory / "result.jpg").shape, sample_image().shape)

    def test_other_thread_stdout_not_added_to_run_logs(self):
        original = pipeline.preprocess_image
        def preprocessing(image):
            other = threading.Thread(target=lambda: print("UNRELATED_OTHER_THREAD"))
            other.start()
            other.join()
            return original(image)
        with redirect_stdout(StringIO()) as terminal, patch.object(pipeline, "preprocess_image", side_effect=preprocessing):
            result = pipeline.run_pipeline(sample_image(), (40, 30, 400, 260), 30, save_outputs=False)
        self.assertIn("UNRELATED_OTHER_THREAD", terminal.getvalue())
        self.assertNotIn("UNRELATED_OTHER_THREAD", result.logs)

    def test_repeated_millisecond_debug_directory_keeps_prior_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            fixed = datetime(2026, 9, 29, 20, 0, 0)
            mask = np.zeros((10, 10), dtype=np.uint8)
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
            with patch.object(debug_utils, "datetime") as clock:
                clock.now.return_value = fixed
                first = debug_utils.start_debug_run(mask, crop, [], output_root=Path(directory))
                second = debug_utils.start_debug_run(mask, crop, [], output_root=Path(directory))
            self.assertNotEqual(first.run_dir, second.run_dir)


if __name__ == "__main__":
    unittest.main()
