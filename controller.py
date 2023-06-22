"""
    Scrypt to control all models and provide results to the app
"""
import lap
import pathlib
import os

import torch
import pandas as pd
from ultralytics.yolo.engine.results import Results

from detector import Detector
from classifier import Classifier
from metamodel import Meta, std_predict


ROOT = pathlib.Path(__file__).parent
BUFF = 10


def _buff2track(buff) -> dict:
    tracks = {}
    for sample in buff:
        for obj in sample.tolist():
            # xyxy conf
            id = obj[4]
            if id:
                tracks[id] = tracks.get(id,[]) + [obj[:4]+[obj[5]]]
    return tracks


class Controller():
    def __init__(self, stream=True) -> None:
        self.stream = stream

        device='cpu'
        if torch.cuda.is_available():
            device='0'

        self.detector = Detector(ROOT.joinpath('bin/v8m_ep3.pt'), device)
        # self.classifier = Classifier(ROOT.joinpath('bin/classifier.onnx'), device)
        # self.meta = Meta(ROOT.joinpath('bin/meta.cbm'))
    
    def predict(self, source, show=False):
        """
            Predict from single source
        """
        results = []
        buffer = []

        for result in self.detector.predict(source, self.stream, show=show):
            # cls = self.classifier.predict(result) # TODO

            buffer.append(result.boxes.data)
            if len(buffer) > BUFF:
                buffer.pop(0)

            tracks = _buff2track(buffer)
            activities = {}
            for id in tracks:
                # action.append(self.meta.predict(tracks[id])) 
                # activities.append(std_predict(tracks[id]))
                activities[id] = std_predict(tracks[id])
            if self.stream:
                yield activities
            else:
                results.append(activities) 

        if not self.stream:
            return results
    
    def predict_all(self, source):
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
                    rows.extend(self.predict(leaf[0]))
        else: 
            rows.extend(self.predict(source))
        return rows


if __name__=='__main__':
    c = Controller(True)
    c.predict('/Users/samedi/Downloads/613iFGJm0HM_1906_04.mp4')