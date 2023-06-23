"""
    Scrypt for classifier model inference
"""

import torch

class Classifier():
    def __init__(self, path, device='cpu') -> None:
        self.model = None
        self.device = device

    def predict(self, image):
        return 'vehicle'