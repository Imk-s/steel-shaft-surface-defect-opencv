import os
from pathlib import Path
import sys


def configure_application_font(app):
    """选择可读中文字体；Windows 离屏后端无系统字体时显式加载。"""
    from PySide6.QtGui import QFont, QFontDatabase

    families = QFontDatabase.families()
    windows_directory = os.environ.get("WINDIR")
    if not families and windows_directory:
        font_path = Path(windows_directory) / "Fonts" / "msyh.ttc"
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.families()
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC"):
        if family in families:
            app.setFont(QFont(family, 10))
            break


def main():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        raise SystemExit(
            "缺少 PySide6，请先运行：python -m pip install -r requirements-gui.txt"
        ) from error

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("钢轴表面缺陷检测")
    configure_application_font(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
