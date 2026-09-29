"""动态双列图库，中间结果和 Debug 共用。"""

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .image_view import ImageView


@dataclass
class PreparedGallery:
    content: QWidget
    image_views: dict[str, ImageView]


class ImageGallery(QWidget):
    back_requested = Signal()

    def __init__(self, title, empty_text, parent=None):
        super().__init__(parent)
        self.empty_text = empty_text
        self.image_views = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        header = QHBoxLayout()
        back = QPushButton("← 返回")
        back.setObjectName("backButton")
        back.clicked.connect(self.back_requested)
        header.addWidget(back)
        header.addWidget(QLabel(title))
        header.addStretch()
        layout.addLayout(header)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.scroll_area, 1)
        self.set_images({})

    def prepare_images(self, images):
        """构建未展示的整批卡片，失败时不触碰当前页面。"""
        if not isinstance(images, dict):
            raise ValueError("图库图片必须是按名称索引的字典")
        content = QWidget()
        views = {}
        try:
            grid = QGridLayout(content)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(16)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            empty = QLabel(self.empty_text)
            empty.setObjectName("emptyState")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            empty.setVisible(not images)
            grid.addWidget(empty, 0, 0, 1, 2)
            for number, (name, image) in enumerate(images.items()):
                if not isinstance(name, str):
                    raise ValueError("图库图片名称必须是字符串")
                if image is None:
                    raise ValueError(f"图库图片不能为空：{name}")
                card = QGroupBox()
                card_layout = QVBoxLayout(card)
                label = QLabel(name)
                label.setTextFormat(Qt.TextFormat.PlainText)
                label.setWordWrap(True)
                card_layout.addWidget(label)
                view = ImageView()
                view.setMinimumHeight(240)
                view.set_image(image)
                views[name] = view
                card_layout.addWidget(view, 1)
                grid.addWidget(card, number // 2, number % 2)
            grid.setRowStretch((len(images) + 1) // 2, 1)
            return PreparedGallery(content, views)
        except Exception:
            content.deleteLater()
            raise

    def commit_images(self, prepared):
        self.scroll_area.setWidget(prepared.content)
        self.image_views = prepared.image_views

    def set_images(self, images):
        self.commit_images(self.prepare_images(images))
