"""
    Script for  detector model inference
"""

from ultralytics import YOLO

class Detector():
    def __init__(self, path, device='cpu') -> None:
        self.model = YOLO(path)
        self.device = device

    def predict(self, source, stream=False):
        return self.model.predict(source, stream)