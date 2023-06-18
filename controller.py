"""
    Scrypt to control all models and provide results to the app
"""

import pathlib
import os

import torch
import pandas as pd
from ultralytics.yolo.engine.results import Results

from detector import Detector
from classifier import Classifier
from metamodel import Meta


ROOT = pathlib.Path(__file__).parent
BUFF = 10


class Controller():
    def __init__(self, stream=True) -> None:
        self.stream = stream

        device='cpu'
        if torch.cuda.is_available():
            device='0'

        self.detector = Detector(ROOT.joinpath('bin/detector.onnx'), device)
        self.classifier = Classifier(ROOT.joinpath('bin/classifier.onnx'), device)
        self.meta = Meta(ROOT.joinpath('bin/meta.cbm'))

    def predict(self, source):
        """
            Predicts from source or from each file in a source if directory
        """
        rows=[]
        try: 
            isdir = os.path.isdir(source)
        except: 
            isdir = False
        if isdir:
            for leaf in os.walk(source):
                if len(leaf[2])>0: # TODO: check if contatins images/videos
                    rows.extend(self._predict(leaf[0]))
        else: 
            rows.extend(self._predict(source))
        pd = pd.DataFrame(data=rows,columns=['time','movement'])
        return pd
    
    def _predict(self, source):
        """
            Predict from single source
        """
        action = []
        buffer = []

        for result in self.detector.predict(source, self.stream):
            cls = self.classifier.predict(result) # TODO

            buffer.append( (result.boxes.data, cls) )
            if buffer.size > BUFF:
                buffer.pop(0)

            action.append(self.meta.predict(buffer)) # TODO
        
        return action