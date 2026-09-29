"""原始 ndarray 与 Qt 预览分离的可复用图片控件。"""

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


def array_to_qimage(image: np.ndarray) -> QImage:
    """复制到 Qt 自有存储；彩色图沿用 OpenCV 的 BGR/BGRA 通道顺序。"""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("预览图片必须是非空 uint8 ndarray")
    if image.ndim == 2:
        data = np.ascontiguousarray(image)
        image_format = QImage.Format.Format_Grayscale8
    elif image.ndim == 3 and image.shape[2] == 3:
        data = np.ascontiguousarray(image)
        image_format = QImage.Format.Format_BGR888
    elif image.ndim == 3 and image.shape[2] == 4:
        data = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])
        image_format = QImage.Format.Format_RGBA8888
    else:
        raise ValueError("预览图片必须是灰度、BGR 或 BGRA 图片")
    height, width = data.shape[:2]
    return QImage(data.data, width, height, data.strides[0], image_format).copy()


class ImageView(QWidget):
    roi_selected = Signal(object)
    roi_selection_changed = Signal(bool)

    def __init__(self, empty_text="暂无图片", parent=None):
        super().__init__(parent)
        self.empty_text = empty_text
        self.original_image: np.ndarray | None = None
        self._source_pixmap = QPixmap()
        self._display_pixmap = QPixmap()
        self.display_scale = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._roi: tuple[int, int, int, int] | None = None
        self.selection_enabled = False
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContentsMargins(8, 8, 8, 8)
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(460, 340)

    @property
    def image_rect(self):
        if self.original_image is None:
            return QRectF()
        height, width = self.original_image.shape[:2]
        return QRectF(
            self.offset_x, self.offset_y,
            width * self.display_scale, height * self.display_scale,
        )

    def set_image(self, image: np.ndarray | None):
        pixmap = QPixmap() if image is None else QPixmap.fromImage(array_to_qimage(image))
        self.set_selection_enabled(False)
        self._roi = None
        self.original_image = image
        self._source_pixmap = pixmap
        self._fit_image()

    @property
    def roi(self):
        return self._roi

    @property
    def roi_rect(self):
        """ROI 覆盖层的显示坐标；原始 ROI 不随控件缩放改变。"""
        if self._roi is None or self.original_image is None:
            return QRectF()
        x, y, width, height = self._roi
        return QRectF(
            self.image_to_view(x, y),
            QSizeF(width * self.display_scale, height * self.display_scale),
        )

    def set_roi(self, roi: tuple[int, int, int, int] | None):
        if roi is not None:
            if self.original_image is None:
                raise ValueError("请先导入图片")
            x, y, width, height = roi
            image_height, image_width = self.original_image.shape[:2]
            if any(not isinstance(value, int) for value in roi):
                raise ValueError("ROI 必须使用原图整数坐标")
            if (
                x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > image_width or y + height > image_height
            ):
                raise ValueError("ROI 必须位于原图内部且宽高大于 0")
        self._roi = roi
        self.update()

    def image_to_view(self, x: float, y: float):
        return QPointF(
            x * self.display_scale + self.offset_x,
            y * self.display_scale + self.offset_y,
        )

    def view_to_image(self, point: QPointF, clamp=False):
        if self.original_image is None or self.display_scale <= 0:
            return None
        if not clamp and not self.image_rect.contains(point):
            return None
        height, width = self.original_image.shape[:2]
        x = (point.x() - self.offset_x) / self.display_scale
        y = (point.y() - self.offset_y) / self.display_scale
        return QPointF(min(max(x, 0.0), width), min(max(y, 0.0), height))

    def set_selection_enabled(self, enabled: bool):
        enabled = bool(enabled and self.original_image is not None)
        changed = enabled != self.selection_enabled
        self.selection_enabled = enabled
        self._drag_start = self._drag_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if enabled:
            self.setFocus()
        if changed:
            self.roi_selection_changed.emit(enabled)
        self.update()

    def _fit_image(self):
        area = self.contentsRect()
        if self._source_pixmap.isNull() or area.width() <= 0 or area.height() <= 0:
            self._display_pixmap = QPixmap()
            self.display_scale = 0.0
            self.offset_x = self.offset_y = 0.0
        else:
            height, width = self.original_image.shape[:2]
            self.display_scale = min(area.width() / width, area.height() / height)
            self.offset_x = area.x() + (area.width() - width * self.display_scale) / 2.0
            self.offset_y = area.y() + (area.height() - height * self.display_scale) / 2.0
            self._display_pixmap = self._source_pixmap.scaled(
                area.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#edf0f5"))
        painter.setPen(QColor("#dce1e8"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._display_pixmap.isNull():
            painter.setPen(QColor("#667085"))
            painter.drawText(
                self.contentsRect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.empty_text,
            )
        else:
            # 浮点目标框保持精确宽高比，避免 QPixmap 整数尺寸的舍入偏差。
            painter.drawPixmap(
                self.image_rect,
                self._display_pixmap,
                QRectF(self._display_pixmap.rect()),
            )
            self._paint_roi(painter)

    def _paint_roi(self, painter):
        rect = self.roi_rect
        if self._drag_start is not None and self._drag_end is not None:
            rect = QRectF(
                self.image_to_view(self._drag_start.x(), self._drag_start.y()),
                self.image_to_view(self._drag_end.x(), self._drag_end.y()),
            ).normalized()
        if rect.isEmpty():
            return
        painter.save()
        painter.setClipRect(self.image_rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#0891b2"), 2))
        painter.setBrush(QColor(34, 211, 238, 30))
        painter.drawRect(rect)
        painter.restore()

    def mousePressEvent(self, event):
        if self.selection_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self.view_to_image(event.position())
            if point is not None:
                self._drag_start = self._drag_end = point
                event.accept()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selection_enabled and self._drag_start is not None:
            self._drag_end = self.view_to_image(event.position(), clamp=True)
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self.selection_enabled and self._drag_start is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            start = self._drag_start
            end = self.view_to_image(event.position(), clamp=True)
            if (
                abs(end.x() - start.x()) * self.display_scale >= 3
                and abs(end.y() - start.y()) * self.display_scale >= 3
            ):
                height, width = self.original_image.shape[:2]
                x0 = max(0, math.floor(min(start.x(), end.x())))
                y0 = max(0, math.floor(min(start.y(), end.y())))
                x1 = min(width, math.ceil(max(start.x(), end.x())))
                y1 = min(height, math.ceil(max(start.y(), end.y())))
                self.set_roi((x0, y0, x1 - x0, y1 - y0))
                self.set_selection_enabled(False)
                self.roi_selected.emit(self._roi)
            else:
                self._drag_start = self._drag_end = None
                self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.selection_enabled and event.key() == Qt.Key.Key_Escape:
            self.set_selection_enabled(False)
            event.accept()
            return
        super().keyPressEvent(event)
