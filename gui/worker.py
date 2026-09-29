"""后台 Worker：只调用高层 pipeline，通过信号交付完整结果。"""

import traceback

from PySide6.QtCore import QObject, Signal, Slot

from pipeline import run_pipeline


class PipelineWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str, str)
    finished = Signal()

    def __init__(self, image, roi, diameter_mm, *, pipeline_options=None):
        super().__init__()
        self.image = image
        self.roi = roi
        self.diameter_mm = diameter_mm
        self.pipeline_options = dict(pipeline_options or {})

    @Slot()
    def run(self):
        try:
            output = run_pipeline(
                image=self.image,
                roi=self.roi,
                diameter_mm=self.diameter_mm,
                **self.pipeline_options,
            )
            self.succeeded.emit(output)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}", traceback.format_exc())
        finally:
            self.finished.emit()
