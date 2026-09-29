"""GUI 页面、状态提交和后台检测任务管理。"""

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config import IMAGE_PATH
from main import read_image

from .image_view import ImageView
from .image_gallery import ImageGallery
from .state import InputState, OutputState, parse_diameter
from .worker import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self, *, pipeline_options=None):
        super().__init__()
        self.input_state = InputState()
        self.output_state = OutputState()
        self._pipeline_options = dict(pipeline_options or {})
        self._thread = None
        self._worker = None
        self._close_requested = False
        self._last_run_status = ""
        self.setWindowTitle("钢轴表面缺陷检测")
        self.resize(1180, 780)
        self.setMinimumSize(760, 560)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.main_page = self._build_main_page()
        self.stack.addWidget(self.main_page)
        self.intermediate_page = ImageGallery(
            "中间结果", "暂无中间结果，请先运行检测"
        )
        self.intermediate_page.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.main_page)
        )
        self.debug_page = ImageGallery(
            "Debug 图", "暂无 Debug 图，请先运行检测"
        )
        self.debug_page.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.main_page)
        )
        self.logs_page = self._build_logs_page(
            "后台日志", "暂无运行日志，请先运行检测"
        )
        for page in (self.intermediate_page, self.debug_page, self.logs_page):
            self.stack.addWidget(page)

        self.intermediate_button.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.intermediate_page)
        )
        self.debug_button.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.debug_page)
        )
        self.logs_button.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.logs_page)
        )
        self.generate_button.clicked.connect(self._on_generate)
        self.import_button.clicked.connect(self._import_image)
        self.roi_button.clicked.connect(self._toggle_roi_selection)
        self.input_view.roi_selected.connect(self._on_roi_selected)
        self.input_view.roi_selection_changed.connect(self._on_selection_changed)
        self.diameter_input.textChanged.connect(self._on_diameter_changed)
        self.statusBar().showMessage("导入图片、选择钢轴 ROI、输入真实直径后开始生成")
        self.setStyleSheet("""
            QMainWindow { background: #f3f5f8; }
            QGroupBox {
                background: white; border: 1px solid #dce1e8;
                border-radius: 8px; margin-top: 12px;
                padding: 18px 12px 12px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; }
            QPushButton { padding: 8px 14px; }
            QLineEdit { padding: 7px; }
            QPushButton#generateButton {
                color: white; background: #2563eb; border: none;
                border-radius: 5px; padding: 10px;
            }
            QPushButton#generateButton:hover { background: #1d4ed8; }
            QPushButton#generateButton:disabled { background: #94a3b8; }
        """)

    def _build_main_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 14)
        heading = QLabel("钢轴表面缺陷检测")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(heading)
        columns = QHBoxLayout()
        columns.setSpacing(18)
        layout.addLayout(columns, 1)

        input_box = QGroupBox("当前输入")
        input_layout = QVBoxLayout(input_box)
        self.import_button = QPushButton("导入图片")
        input_layout.addWidget(self.import_button)
        self.input_view = ImageView("请导入钢轴图片")
        input_layout.addWidget(self.input_view, 1)
        self.image_info = QLabel("尚未导入图片")
        self.image_info.setWordWrap(True)
        input_layout.addWidget(self.image_info)
        self.roi_button = QPushButton("选择 ROI")
        self.roi_button.setEnabled(False)
        input_layout.addWidget(self.roi_button)
        self.roi_info = QLabel("尚未选择 ROI")
        self.roi_info.setWordWrap(True)
        input_layout.addWidget(self.roi_info)
        diameter_row = QHBoxLayout()
        diameter_row.addWidget(QLabel("钢轴真实直径"))
        self.diameter_input = QLineEdit()
        self.diameter_input.setPlaceholderText("例如 30")
        diameter_row.addWidget(self.diameter_input, 1)
        diameter_row.addWidget(QLabel("mm"))
        input_layout.addLayout(diameter_row)
        self.generate_button = QPushButton("开始生成")
        self.generate_button.setObjectName("generateButton")
        input_layout.addWidget(self.generate_button)
        columns.addWidget(input_box, 1)

        output_box = QGroupBox("图片结果")
        output_layout = QVBoxLayout(output_box)
        output_layout.addWidget(QLabel("最近一次生成结果"))
        self.result_view = ImageView("暂无生成结果")
        output_layout.addWidget(self.result_view, 1)
        note = QLabel("导入图片、修改直径或 ROI 后，仍保留最近一次成功结果。")
        note.setWordWrap(True)
        output_layout.addWidget(note)
        self.intermediate_button = QPushButton("查看中间结果")
        self.debug_button = QPushButton("查看 Debug 图")
        self.logs_button = QPushButton("查看后台日志")
        for button in (self.intermediate_button, self.debug_button, self.logs_button):
            output_layout.addWidget(button)
        columns.addWidget(output_box, 1)
        return page

    def _build_logs_page(self, title, message):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        header = QHBoxLayout()
        back = QPushButton("← 返回")
        back.setObjectName("backButton")
        back.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        header.addWidget(back)
        header.addWidget(QLabel(title))
        header.addStretch()
        layout.addLayout(header)
        empty = QLabel(message)
        empty.setObjectName("emptyState")
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty, 1)
        self.logs_empty = empty
        self.logs_view = QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.logs_view.setFont(font)
        self.logs_view.hide()
        layout.addWidget(self.logs_view, 1)
        return page

    def _import_image(self):
        directory = (
            self.input_state.current_image_path.parent
            if self.input_state.current_image_path is not None
            else IMAGE_PATH.parent
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入钢轴图片",
            str(directory),
            "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)",
        )
        if filename:
            try:
                self.load_image(Path(filename))
            except Exception as error:
                QMessageBox.warning(self, "图片导入失败", str(error))

    def load_image(self, path: Path):
        """读图成功后更新当前输入；复用命令行的中文路径读图函数。"""
        path = Path(path).resolve()
        image = read_image(path)
        if image is None:
            raise ValueError(f"无法读取图片：{path.name}")
        self.input_view.set_image(image)
        self.input_state.current_image = image
        self.input_state.current_image_path = path
        self.input_state.current_roi = None
        height, width = image.shape[:2]
        self.image_info.setText(f"{path.name}  ·  {width} × {height} px")
        self.image_info.setToolTip(str(path))
        self.roi_info.setText("尚未选择 ROI")
        self.roi_button.setEnabled(True)
        self.statusBar().showMessage("图片已导入；下一步选择 ROI")

    def _toggle_roi_selection(self):
        if self.input_state.current_image is None:
            QMessageBox.warning(self, "请选择图片", "请先导入钢轴图片。")
            return
        self.input_view.set_selection_enabled(not self.input_view.selection_enabled)

    def _on_selection_changed(self, enabled):
        self.roi_button.setText("取消选择 ROI（Esc）" if enabled else "选择 ROI")
        if enabled:
            self.statusBar().showMessage("在左侧图片上拖拽；左右边缘贴紧钢轴两侧，Esc 取消")
        else:
            self.statusBar().showMessage("ROI 选择结束")

    def _on_roi_selected(self, roi):
        self.input_state.current_roi = roi
        x, y, width, height = roi
        self.roi_info.setText(f"原图 ROI：x={x}，y={y}，w={width}，h={height}")
        self.statusBar().showMessage("ROI 已保存为原图坐标")

    def _on_diameter_changed(self, text):
        try:
            self.input_state.diameter_mm = parse_diameter(text)
        except ValueError:
            self.input_state.diameter_mm = None

    def apply_output_state(self, output: OutputState):
        """成功回调的唯一完整输出提交入口。

        卡片和预览先准备，成功后连续提交。失败时旧图、旧图库、旧快照
        均保留，主线程中提交期间不处理事件。
        """
        if not isinstance(output, OutputState) or output.result_image is None:
            raise ValueError("成功输出必须包含 Result 图片")
        if (
            not isinstance(output.intermediate_images, dict)
            or not isinstance(output.debug_images, dict)
            or not isinstance(output.logs, str)
        ):
            raise ValueError("成功输出的中间图和 Debug 图必须是字典，日志必须是字符串")
        prepared = self.intermediate_page.prepare_images(output.intermediate_images)
        prepared_debug = None
        try:
            prepared_debug = self.debug_page.prepare_images(output.debug_images)
            self.result_view.set_image(output.result_image)
        except Exception:
            prepared.content.deleteLater()
            if prepared_debug is not None:
                prepared_debug.content.deleteLater()
            raise
        self.intermediate_page.commit_images(prepared)
        self.debug_page.commit_images(prepared_debug)
        self.output_state = output
        self._refresh_output_summaries()

    def _refresh_output_summaries(self):
        self.logs_view.setPlainText(self.output_state.logs)
        self.logs_empty.setVisible(not self.output_state.logs)
        self.logs_view.setVisible(bool(self.output_state.logs))

    def show_generation_error(self, message: str, details: str | None = None):
        """失败回调只显示错误，保留最近成功输出。"""
        self.statusBar().showMessage("生成失败；保留最近一次成功结果")
        if details is None:
            QMessageBox.critical(self, "生成失败", message)
        else:
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Critical)
            dialog.setWindowTitle("生成失败")
            dialog.setText(message)
            dialog.setInformativeText("已保留最近一次成功结果。")
            dialog.setDetailedText(details)
            dialog.exec()

    @property
    def is_running(self):
        return self._thread is not None

    def _on_generate(self):
        if self.is_running:
            return
        try:
            self.input_state.validate()
            if self.input_view.selection_enabled:
                raise ValueError("请先完成 ROI 拖拽，或按 Esc 结束 ROI 选择。")
        except ValueError as error:
            QMessageBox.warning(self, "请检查输入", str(error))
            return
        self._start_generation()

    def _start_generation(self):
        # 点击时取原分辨率输入快照，防止后台任务引用之后变更的输入。
        worker = PipelineWorker(
            self.input_state.current_image.copy(),
            tuple(self.input_state.current_roi),
            self.input_state.diameter_mm,
            pipeline_options=self._pipeline_options,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_worker_succeeded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._last_run_status = "处理中……"
        self._set_busy(True)
        self.statusBar().showMessage(self._last_run_status)
        thread.start()

    def _set_busy(self, busy):
        self.generate_button.setEnabled(not busy)
        self.generate_button.setText("处理中……" if busy else "开始生成")
        self.import_button.setEnabled(not busy)
        self.diameter_input.setEnabled(not busy)
        self.input_view.setEnabled(not busy)
        self.roi_button.setEnabled(not busy and self.input_state.current_image is not None)

    @Slot(object)
    def _on_worker_succeeded(self, output):
        try:
            self.apply_output_state(output)
        except Exception as error:
            self._on_worker_failed(f"结果显示失败：{error}", "")
            return
        self._last_run_status = "检测完成；已更新最近一次成功结果"
        if output.run_directory is not None:
            self._last_run_status += f" · 保存目录：{output.run_directory.name}"
        self.statusBar().showMessage(self._last_run_status)

    @Slot(str, str)
    def _on_worker_failed(self, message, details):
        self._last_run_status = "生成失败；保留最近一次成功结果"
        self.show_generation_error(message, details or None)

    @Slot()
    def _on_thread_finished(self):
        self._thread = self._worker = None
        self._set_busy(False)
        self.statusBar().showMessage(self._last_run_status)
        if self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self.is_running:
            self._close_requested = True
            self.statusBar().showMessage("检测完成后将关闭窗口……")
            event.ignore()
            return
        super().closeEvent(event)
