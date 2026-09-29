"""离屏 GUI 测试：不会打开本机窗口或调用 OpenCV GUI。"""

import math
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from gui.__main__ import configure_application_font
from gui.image_view import ImageView, array_to_qimage
from gui.image_gallery import ImageGallery
from gui.main_window import MainWindow
from gui.state import OutputState


class GuiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application_font(cls.app)

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()


class LayoutTests(GuiTestCase):
    def test_main_page_and_navigation(self):
        self.assertEqual(self.window.stack.count(), 4)
        self.assertIs(self.window.stack.currentWidget(), self.window.main_page)
        for button, page in (
            (self.window.intermediate_button, self.window.intermediate_page),
            (self.window.debug_button, self.window.debug_page),
            (self.window.logs_button, self.window.logs_page),
        ):
            button.click()
            self.assertIs(self.window.stack.currentWidget(), page)
            page.findChild(QPushButton, "backButton").click()
            self.assertIs(self.window.stack.currentWidget(), self.window.main_page)

    def test_layout_resizes_both_previews(self):
        for width, height in ((800, 600), (1500, 960)):
            self.window.resize(width, height)
            self.app.processEvents()
            left = self.window.input_view.geometry()
            right = self.window.result_view.geometry()
            self.assertGreater(left.width(), 0)
            self.assertGreater(right.height(), 0)
            self.assertLessEqual(abs(left.width() - right.width()), 1)


class ImageTests(GuiTestCase):
    def test_original_pixels_and_aspect_survive_all_sizes(self):
        for width, height in ((1920, 1080), (4000, 3000), (900, 1600)):
            image = np.full((height, width, 3), (17, 73, 181), dtype=np.uint8)
            view = ImageView()
            view.set_image(image)
            view.show()
            for view_width, view_height in ((480, 320), (720, 800), (300, 600)):
                view.resize(view_width, view_height)
                self.app.processEvents()
                rect = view.image_rect
                self.assertAlmostEqual(rect.width() / rect.height(), width / height)
                self.assertAlmostEqual(rect.center().x(), QRectF(view.contentsRect()).center().x())
                self.assertAlmostEqual(rect.center().y(), QRectF(view.contentsRect()).center().y())
                self.assertLessEqual(rect.width(), view.contentsRect().width() + 1e-6)
                self.assertLessEqual(rect.height(), view.contentsRect().height() + 1e-6)
                self.assertIs(view.original_image, image)
                self.assertEqual(image.shape, (height, width, 3))
                self.assertTrue(np.all(image == (17, 73, 181)))
            view.close()
            view.deleteLater()

    def test_color_conversion_owns_memory_and_handles_strides(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:] = (0, 0, 255)
        qt_image = array_to_qimage(image[:, ::2])
        image[:] = 0
        self.assertEqual(qt_image.pixelColor(0, 0), QColor("red"))
        gray = array_to_qimage(np.full((2, 2), 123, dtype=np.uint8))
        self.assertEqual(gray.pixelColor(1, 1).red(), 123)
        with self.assertRaises(ValueError):
            array_to_qimage(np.zeros((2, 2), dtype=np.float32))

    def test_chinese_path_import_and_failed_import_preserves_input(self):
        # 测试图片在项目临时目录内创建，退出时自动清理。
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "钢轴样例.png"
            for width, height in ((1920, 1080), (4000, 3000), (900, 1600)):
                image = QImage(width, height, QImage.Format.Format_RGB32)
                image.fill(Qt.GlobalColor.red)
                self.assertTrue(image.save(str(path)))
                self.window.load_image(path)
                original = self.window.input_state.current_image
                self.assertEqual(original.shape, (height, width, 3))
                self.assertEqual(self.window.input_state.current_image_path, path.resolve())
            with self.assertRaises(FileNotFoundError):
                self.window.load_image(Path(directory) / "不存在.png")
            self.assertIs(self.window.input_state.current_image, original)


class RoiTests(GuiTestCase):
    def test_mapping_and_resize_for_each_resolution(self):
        view = self.window.input_view
        for width, height in ((1920, 1080), (4000, 3000), (900, 1600)):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            view.set_image(image)
            roi = (width // 5, height // 4, width // 2, height // 2)
            view.set_roi(roi)
            for window_size in ((800, 600), (1500, 960), (1000, 800)):
                self.window.resize(*window_size)
                self.app.processEvents()
                for x, y in ((0, 0), (width, height), (width * 0.35, height * 0.7)):
                    mapped = view.view_to_image(view.image_to_view(x, y), clamp=True)
                    self.assertAlmostEqual(mapped.x(), x)
                    self.assertAlmostEqual(mapped.y(), y)
                self.assertEqual(view.roi, roi)
                self.assertAlmostEqual(view.roi_rect.width(), roi[2] * view.display_scale)
                self.assertAlmostEqual(view.roi_rect.left(), view.offset_x + roi[0] * view.display_scale)
                self.assertEqual(image.shape, (height, width, 3))

    def test_real_mouse_drag_reversed_and_clipped(self):
        view = self.window.input_view
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        view.set_image(image)
        for reverse in (False, True):
            first = view.image_to_view(240, 180).toPoint()
            last = view.image_to_view(1440, 900).toPoint()
            if reverse:
                first, last = last, first
            start = view.view_to_image(QPointF(first), clamp=True)
            end = view.view_to_image(QPointF(last), clamp=True)
            x0 = math.floor(min(start.x(), end.x()))
            y0 = math.floor(min(start.y(), end.y()))
            x1 = math.ceil(max(start.x(), end.x()))
            y1 = math.ceil(max(start.y(), end.y()))
            view.set_selection_enabled(True)
            QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=first)
            QTest.mouseMove(view, last)
            QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=last)
            self.assertEqual(view.roi, (x0, y0, x1 - x0, y1 - y0))
            self.assertEqual(self.window.input_state.current_roi, view.roi)
            self.assertFalse(view.selection_enabled)

        view.set_selection_enabled(True)
        first = view.image_to_view(300, 300).toPoint()
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(view.width() + 100, view.height() + 100))
        x, y, width, height = view.roi
        self.assertEqual(x + width, 1920)
        self.assertEqual(y + height, 1080)
        self.assertTrue(np.all(image == 0))

    def test_letterbox_click_cancel_and_new_image(self):
        view = self.window.input_view
        view.set_image(np.zeros((1600, 900, 3), dtype=np.uint8))
        view.set_roi((100, 200, 400, 800))
        previous = view.roi
        view.set_selection_enabled(True)
        self.assertIsNone(view.view_to_image(QPointF(0, 0)))
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(30, 30))
        self.assertEqual(view.roi, previous)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        self.assertEqual(view.roi, previous)
        self.assertFalse(view.selection_enabled)
        view.set_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertIsNone(view.roi)


class StateTests(GuiTestCase):
    def make_output(self, value, image_count=1):
        return OutputState(
            result_image=np.full((360, 640, 3), value, dtype=np.uint8),
            intermediate_images={
                f"Intermediate {i}": np.full((20, 30), value, dtype=np.uint8)
                for i in range(image_count)
            },
            debug_images={
                f"Debug {i}": np.full((40, 30, 3), value, dtype=np.uint8)
                for i in range(image_count)
            },
            logs=f"logs {value}",
        )

    def test_import_B_and_input_changes_preserve_A_then_commit_B(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            paths = [Path(directory) / "图片A.png", Path(directory) / "图片B.png"]
            for path, color in zip(paths, (Qt.GlobalColor.red, Qt.GlobalColor.blue)):
                image = QImage(1920, 1080, QImage.Format.Format_RGB32)
                image.fill(color)
                self.assertTrue(image.save(str(path)))
            self.window.load_image(paths[0])
            self.window.diameter_input.setText("30")
            self.window.input_view.set_roi((100, 200, 800, 600))
            self.window._on_roi_selected(self.window.input_view.roi)
            output_a = self.make_output(73)
            self.window.apply_output_state(output_a)
            self.window.load_image(paths[1])
            self.assertIsNone(self.window.input_state.current_roi)
            self.assertIsNone(self.window.input_view.roi)
            self.assertEqual(self.window.input_state.current_image_path, paths[1].resolve())
            self.assertTrue(np.all(self.window.input_state.current_image == (255, 0, 0)))
            self.assertIs(self.window.output_state, output_a)
            self.assertIs(self.window.result_view.original_image, output_a.result_image)
            self.assertEqual(self.window.output_state.logs, "logs 73")
            self.assertEqual(self.window.logs_view.toPlainText(), "logs 73")
            self.window.diameter_input.setText("45")
            self.window.input_view.set_roi((500, 100, 900, 800))
            self.window._on_roi_selected(self.window.input_view.roi)
            self.assertEqual(self.window.input_state.diameter_mm, 45)
            self.assertIs(self.window.output_state, output_a)
            self.window.resize(900, 640)
            self.app.processEvents()
            self.assertIs(self.window.result_view.original_image, output_a.result_image)
            output_b = self.make_output(181, image_count=5)
            self.window.apply_output_state(output_b)
            self.assertIs(self.window.output_state, output_b)
            self.assertIs(self.window.result_view.original_image, output_b.result_image)
            self.assertEqual(len(self.window.output_state.intermediate_images), 5)
            self.assertEqual(len(self.window.output_state.debug_images), 5)
            self.assertEqual(self.window.output_state.logs, "logs 181")
            self.assertEqual(self.window.logs_view.toPlainText(), "logs 181")

    def test_failure_and_invalid_output_preserve_last_success(self):
        output_a = self.make_output(73)
        self.window.apply_output_state(output_a)
        with patch("gui.main_window.QMessageBox.critical") as message:
            self.window.show_generation_error("测试失败")
            message.assert_called_once()
        for invalid in (
            OutputState(),
            OutputState(result_image=np.zeros((2, 2), dtype=float)),
            OutputState(result_image=output_a.result_image, debug_images=None),
            OutputState(result_image=output_a.result_image, intermediate_images={"坏图": np.zeros((2, 2), dtype=float)}),
            OutputState(result_image=output_a.result_image, debug_images={"坏 Debug": None}),
        ):
            with self.assertRaises(ValueError):
                self.window.apply_output_state(invalid)
            self.assertIs(self.window.output_state, output_a)
            self.assertIs(self.window.result_view.original_image, output_a.result_image)
            self.assertEqual(self.window.logs_view.toPlainText(), output_a.logs)

    def test_empty_pages_and_dynamic_output_counts(self):
        for page in (self.window.intermediate_page, self.window.debug_page, self.window.logs_page):
            self.assertIn("暂无", page.findChild(QLabel, "emptyState").text())
        for count in (0, 1, 7):
            self.window.apply_output_state(self.make_output(50, image_count=count))
            self.assertEqual(len(self.window.output_state.debug_images), count)
            self.assertEqual(len(self.window.output_state.intermediate_images), count)

    def test_generate_validates_before_starting_worker(self):
        with patch("gui.main_window.QMessageBox.warning") as warning:
            self.window.generate_button.click()
            self.assertIn("导入", warning.call_args.args[2])
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.window.input_state.current_image = image
            self.window.input_view.set_image(image)
            self.window.generate_button.click()
            self.assertIn("ROI", warning.call_args.args[2])
            self.window.input_view.set_roi((100, 100, 800, 600))
            self.window._on_roi_selected(self.window.input_view.roi)
            for text in ("", "0", "-1", "abc", "nan", "inf"):
                self.window.diameter_input.setText(text)
                self.window.generate_button.click()
                self.assertIn("直径", warning.call_args.args[2])
                self.assertIsNone(self.window.input_state.diameter_mm)
        previous = self.window.output_state
        self.window.diameter_input.setText("30.5")
        with patch.object(self.window, "_start_generation") as start:
            self.window.generate_button.click()
            start.assert_called_once()
        self.assertIs(self.window.output_state, previous)


class GalleryTests(GuiTestCase):
    def test_debug_reuses_gallery_for_arbitrary_names_and_counts(self):
        self.assertIsInstance(self.window.debug_page, ImageGallery)
        self.assertIsInstance(self.window.intermediate_page, ImageGallery)
        self.window.debug_button.click()
        for count in (0, 1, 6):
            images = {f"不是固定名称 {i}": np.zeros((300, 150, 3), dtype=np.uint8) for i in range(count)}
            self.window.debug_page.set_images(images)
            self.app.processEvents()
            self.assertEqual(list(self.window.debug_page.image_views), list(images))

    def test_dynamic_two_columns_scroll_and_aspect(self):
        gallery = self.window.intermediate_page
        self.assertIsInstance(gallery, ImageGallery)
        self.window.intermediate_button.click()
        for count in (0, 1, 7, 2):
            images = {f"任意图片 {i}": np.zeros((200 + 20 * i, 400, 3), dtype=np.uint8) for i in range(count)}
            gallery.set_images(images)
            self.app.processEvents()
            self.assertEqual(list(gallery.image_views), list(images))
            grid = gallery.scroll_area.widget().layout()
            for i, (name, image) in enumerate(images.items()):
                self.assertIsNotNone(grid.itemAtPosition(i // 2, i % 2))
                view = gallery.image_views[name]
                self.assertAlmostEqual(view.image_rect.width() / view.image_rect.height(), image.shape[1] / image.shape[0])
            if count == 7:
                self.assertGreater(gallery.scroll_area.verticalScrollBar().maximum(), 0)

    def test_bad_intermediate_does_not_replace_previous_gallery(self):
        previous = {"Old": np.zeros((100, 200), dtype=np.uint8)}
        gallery = self.window.intermediate_page
        gallery.set_images(previous)
        old = gallery.scroll_area.widget()
        with self.assertRaises(ValueError):
            gallery.set_images({"New": np.zeros((10, 10), dtype=float)})
        self.assertIs(gallery.scroll_area.widget(), old)
        self.assertEqual(list(gallery.image_views), ["Old"])


class LogTests(GuiTestCase):
    def test_read_only_monospace_complete_logs_and_replacement(self):
        self.assertTrue(self.window.logs_view.isReadOnly())
        self.assertTrue(self.window.logs_view.font().fixedPitch())
        logs = "[MEASURE C7] Scratch width=0.312000 mm valid=True\n" + "完整中文日志\n" * 2000
        first = OutputState(result_image=np.zeros((80, 120, 3), dtype=np.uint8), logs=logs)
        self.window.apply_output_state(first)
        self.window.logs_button.click()
        self.assertEqual(self.window.logs_view.toPlainText(), logs)
        self.assertFalse(self.window.logs_empty.isVisible())
        second = OutputState(result_image=np.zeros((80, 120, 3), dtype=np.uint8), logs="第二轮日志")
        self.window.apply_output_state(second)
        self.assertEqual(self.window.logs_view.toPlainText(), "第二轮日志")
        with patch("gui.main_window.QMessageBox.critical"):
            self.window.show_generation_error("失败")
        self.assertEqual(self.window.logs_view.toPlainText(), "第二轮日志")


class CliRegressionTests(unittest.TestCase):
    def test_original_cli_flow_without_windows_or_file_writes(self):
        import main as cli

        image = np.full((160, 200, 3), 127, dtype=np.uint8)
        terminal = StringIO()
        with (
            patch.object(cli, "read_image", return_value=image),
            patch.object(cli.cv2, "selectROI", return_value=(30, 20, 120, 100)),
            patch.object(cli, "prompt_diameter_mm", return_value=30),
            patch.object(cli, "write_image") as write,
            patch.object(cli, "DEBUG_ALL_CONTOURS", False),
            patch.object(cli.cv2, "imshow") as imshow,
            patch.object(cli.cv2, "waitKey", return_value=0),
            patch.object(cli.cv2, "destroyAllWindows"),
            redirect_stdout(terminal),
        ):
            cli.main()
        self.assertIn("D_mm = 30.000000", terminal.getvalue())
        self.assertEqual(write.call_count, 3)
        self.assertEqual(imshow.call_count, 5)


if __name__ == "__main__":
    unittest.main()
