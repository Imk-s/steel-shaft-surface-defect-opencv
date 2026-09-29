"""真实 QThread、GUI 响应、成功/失败状态与完整检测集成测试。"""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QElapsedTimer, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.__main__ import configure_application_font
from gui.main_window import MainWindow
from pipeline import PipelineResult
from test_pipeline import sample_image


class AsyncGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application_font(cls.app)

    def setUp(self):
        self.window = MainWindow(pipeline_options={"save_outputs": False})
        self.window.show()
        self.app.processEvents()
        self.prepare(sample_image())

    def prepare(self, image):
        self.window.input_state.current_image = image
        self.window.input_state.current_image_path = Path("sample.png")
        self.window.input_view.set_image(image)
        self.window.input_view.set_roi((40, 30, 400, 260))
        self.window._on_roi_selected(self.window.input_view.roi)
        self.window.diameter_input.setText("30")

    def wait_until(self, condition, timeout=10000):
        timer = QElapsedTimer()
        timer.start()
        while not condition() and timer.elapsed() < timeout:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(condition(), "后台任务没有在限定时间内完成")

    def tearDown(self):
        if self.window.is_running:
            self.wait_until(lambda: not self.window.is_running)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_thread_responsive_snapshot_and_atomic_success(self):
        started, release = threading.Event(), threading.Event()
        owner = threading.get_ident()
        worker_threads = []
        callbacks = []
        output = PipelineResult(result_image=sample_image(), logs="SUCCESS A")
        old = self.window.output_state
        original = self.window.apply_output_state
        def commit(value):
            callbacks.append(threading.get_ident())
            return original(value)
        def work(image, roi, diameter_mm, **options):
            worker_threads.append(threading.get_ident())
            np.testing.assert_array_equal(image, sample_image())
            self.assertIsNot(image, self.window.input_state.current_image)
            self.assertEqual(roi, (40, 30, 400, 260))
            self.assertEqual(diameter_mm, 30)
            started.set()
            if not release.wait(3):
                raise RuntimeError("测试任务等待超时")
            return output
        beats = []
        heartbeat = QTimer()
        heartbeat.setInterval(5)
        heartbeat.timeout.connect(lambda: beats.append(1))
        with patch("gui.worker.run_pipeline", side_effect=work), patch.object(self.window, "apply_output_state", side_effect=commit):
            heartbeat.start()
            self.window.generate_button.click()
            self.wait_until(started.is_set)
            self.assertFalse(self.window.generate_button.isEnabled())
            self.assertFalse(self.window.import_button.isEnabled())
            self.assertIs(self.window.output_state, old)
            self.window._on_generate()
            self.assertEqual(len(worker_threads), 1)
            self.window.intermediate_button.click()
            self.assertIs(self.window.stack.currentWidget(), self.window.intermediate_page)
            QTest.qWait(60)
            self.assertGreaterEqual(len(beats), 3)
            release.set()
            self.wait_until(lambda: not self.window.is_running)
            heartbeat.stop()
        self.assertNotEqual(worker_threads, [owner])
        self.assertEqual(callbacks, [owner])
        self.assertIs(self.window.output_state, output)
        self.assertTrue(self.window.generate_button.isEnabled())
        self.assertEqual(self.window.logs_view.toPlainText(), "SUCCESS A")

    def test_failed_worker_preserves_successful_output(self):
        old = PipelineResult(result_image=sample_image(), logs="PREVIOUS SUCCESS")
        self.window.apply_output_state(old)
        with patch("gui.worker.run_pipeline", side_effect=RuntimeError("模拟检测失败")), patch.object(self.window, "show_generation_error") as error:
            self.window.generate_button.click()
            self.wait_until(lambda: not self.window.is_running)
            error.assert_called_once()
            self.assertIn("模拟检测失败", error.call_args.args[0])
        self.assertIs(self.window.output_state, old)
        self.assertIs(self.window.result_view.original_image, old.result_image)
        self.assertTrue(self.window.generate_button.isEnabled())
        self.assertEqual(self.window.logs_view.toPlainText(), "PREVIOUS SUCCESS")

    def test_real_pipeline_runs_in_worker(self):
        image = self.window.input_state.current_image
        original = image.copy()
        with redirect_stdout(StringIO()), patch.object(self.window, "show_generation_error") as error:
            self.window.generate_button.click()
            self.wait_until(lambda: not self.window.is_running)
            error.assert_not_called()
        self.assertIn("[MEASURE C", self.window.output_state.logs)
        self.assertEqual(self.window.output_state.result_image.shape, image.shape)
        self.assertEqual(len(self.window.output_state.intermediate_images), 7)
        self.assertEqual(len(self.window.output_state.debug_images), 3)
        self.assertEqual(self.window.logs_view.toPlainText(), self.window.output_state.logs)
        self.assertEqual(len(self.window.intermediate_page.image_views), 7)
        self.assertEqual(len(self.window.debug_page.image_views), 3)
        np.testing.assert_array_equal(image, original)

    def test_close_during_run_defers_without_destroying_thread(self):
        release = threading.Event()
        def work(**kwargs):
            release.wait(3)
            return PipelineResult(result_image=sample_image())
        with patch("gui.worker.run_pipeline", side_effect=work):
            self.window.generate_button.click()
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.assertTrue(self.window.is_running)
            release.set()
            self.wait_until(lambda: not self.window.is_running)
        self.assertFalse(self.window.isVisible())

    def test_full_A_to_4K_B_workflow_success_and_failure(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            a_path = Path(directory) / "钢轴A.png"
            b_path = Path(directory) / "钢轴B.png"
            a = sample_image()
            self.assertTrue(QImage(a.data, 480, 320, a.strides[0], QImage.Format.Format_BGR888).save(str(a_path)))
            b = QImage(4000, 3000, QImage.Format.Format_RGB32)
            b.fill(0xFF0000FF)
            self.assertTrue(b.save(str(b_path)))
            self.window.load_image(a_path)
            self.window.input_view.set_roi((40, 30, 400, 260))
            self.window._on_roi_selected(self.window.input_view.roi)
            with redirect_stdout(StringIO()), patch.object(self.window, "show_generation_error") as error:
                self.window.generate_button.click()
                self.wait_until(lambda: not self.window.is_running)
                error.assert_not_called()
            output_a = self.window.output_state
            old_intermediate = self.window.intermediate_page.scroll_area.widget()
            old_debug = self.window.debug_page.scroll_area.widget()
            self.window.load_image(b_path)
            self.assertEqual(self.window.input_state.current_image.shape, (3000, 4000, 3))
            self.assertIsNone(self.window.input_state.current_roi)
            self.assertIs(self.window.output_state, output_a)
            self.assertIs(self.window.intermediate_page.scroll_area.widget(), old_intermediate)
            self.assertIs(self.window.debug_page.scroll_area.widget(), old_debug)
            self.assertEqual(self.window.logs_view.toPlainText(), output_a.logs)
            self.window.input_view.set_roi((40, 30, 400, 260))
            self.window._on_roi_selected(self.window.input_view.roi)
            with patch("gui.worker.run_pipeline", side_effect=RuntimeError("B 失败")), patch.object(self.window, "show_generation_error"):
                self.window.generate_button.click()
                self.wait_until(lambda: not self.window.is_running)
            self.assertIs(self.window.output_state, output_a)
            self.assertIs(self.window.intermediate_page.scroll_area.widget(), old_intermediate)
            self.assertIs(self.window.debug_page.scroll_area.widget(), old_debug)
            self.assertEqual(self.window.logs_view.toPlainText(), output_a.logs)
            with redirect_stdout(StringIO()), patch.object(self.window, "show_generation_error") as error:
                self.window.generate_button.click()
                self.wait_until(lambda: not self.window.is_running)
                error.assert_not_called()
            output_b = self.window.output_state
            self.assertIsNot(output_b, output_a)
            self.assertEqual(output_b.result_image.shape, (3000, 4000, 3))
            self.assertIs(self.window.result_view.original_image, output_b.result_image)
            self.assertIsNot(self.window.intermediate_page.scroll_area.widget(), old_intermediate)
            self.assertIsNot(self.window.debug_page.scroll_area.widget(), old_debug)
            self.assertEqual(self.window.logs_view.toPlainText(), output_b.logs)
            self.assertIn("原图=4000x3000", output_b.logs)

    def test_bad_debug_in_worker_result_preserves_entire_output(self):
        old = PipelineResult(
            result_image=sample_image(),
            intermediate_images={"Old intermediate": np.zeros((20, 20), dtype=np.uint8)},
            debug_images={"Old debug": np.zeros((20, 20, 3), dtype=np.uint8)},
            logs="LAST SUCCESS",
        )
        self.window.apply_output_state(old)
        old_intermediate = self.window.intermediate_page.scroll_area.widget()
        old_debug = self.window.debug_page.scroll_area.widget()
        bad = PipelineResult(result_image=sample_image(), debug_images={"Bad": None}, logs="BAD NEW LOG")
        with patch("gui.worker.run_pipeline", return_value=bad), patch.object(self.window, "show_generation_error") as error:
            self.window.generate_button.click()
            self.wait_until(lambda: not self.window.is_running)
            error.assert_called_once()
        self.assertIs(self.window.output_state, old)
        self.assertIs(self.window.intermediate_page.scroll_area.widget(), old_intermediate)
        self.assertIs(self.window.debug_page.scroll_area.widget(), old_debug)
        self.assertIs(self.window.result_view.original_image, old.result_image)
        self.assertEqual(self.window.logs_view.toPlainText(), "LAST SUCCESS")


if __name__ == "__main__":
    unittest.main()
