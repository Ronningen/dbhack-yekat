"""
    Script for  detector model inference
"""

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

class Detector():
    def __init__(self, path, device='cpu') -> None:
        self.model = YOLO(path)
        self.device = device

    def predict(self, source, stream=False, show=False) -> Results:
        """
            returns YOLO().track(source, stream) results
        """
        return self.model.track(source, stream, show=show, line_width=2, show_conf=False)