"""
    Scrypt to control all models and provide results to the app
"""
import lap
import pathlib
import os

import torch
import pandas as pd
import numpy as np
from ultralytics.yolo.engine.results import Results

from .detector import Detector
from .classifier import Classifier
from .metamodel import Meta, std_predict


ROOT = pathlib.Path(__file__).parent
BUFF = 16
CLST = 16


# def _buff2track(buff) -> dict:
#     """
#     :param buff: list of data in frames -
#         [
#             (yolo.results.boxes.data, orig_img),
#             ...
#         ]
#     :return: dict with track data -
#         { 
#             id: [(box, orig_img), ...], 
#             ...
#         }
#     """
#     tracks = {}
#     for sample in buff:
#         for obj in sample[0].tolist():
#             # xyxy img
#             id = obj[4]
#             if id:
#                 box = obj[:4]
#                 tracks[id] = tracks.get(id,[]) + [ (box, sample[1]) ]
#     return tracks


def _track2clip(track, video, scale=0.1) -> torch.FloatTensor:
    """
    :param track: 
    :param video: video to clip [numpy img [H W C], ...]
    :returns: [T C H W] clip
    """
    track_np = np.array(list(map(lambda t: t[1], track[-BUFF:])))

    min_x = np.min(track_np[:,0])
    min_y = np.min(track_np[:,1])
    max_x = np.min(track_np[:,2])
    max_y = np.min(track_np[:,3])
    w_crop = max_x - min_x
    h_crop = max_y - min_y
    # пересчитываем координаты со скейлом
    min_x = max(0, int(min_x - w_crop * scale))
    min_y = max(0, int(min_y - h_crop * scale))
    max_x = min(int(max_x + w_crop * scale), video[0].shape[1])
    max_y = min(int(max_y + h_crop * scale), video[0].shape[0])
    w_crop = max_x - min_x
    h_crop = max_y - min_y

    clip = torch.zeros([BUFF, video[0].shape[2], h_crop, w_crop])
    for i, frame in enumerate(video):
        clip[i] = torch.from_numpy(np.transpose(frame[min_y:max_y, min_x:max_x], [2,0,1]).copy())
    return clip


class Controller():
    def __init__(self, stream=True) -> None:
        self.stream = stream

        device='cpu'
        if torch.cuda.is_available():
            device='0'

        self.detector = Detector(ROOT.joinpath('../../bin/best.pt'), device)
        self.classifier = Classifier(ROOT.joinpath('../../bin/checkpoint_13.zip'), device)

        self.last_tracks = {}
    
    def predict(self, source, show=False):
        """
            Predict from video
        """
        self.last_tracks = {}
        tracks = {} # история всех треков за все видео id: [(номер фрейма, бокс, активность), ...]
        tracks_counter = {} # счетчик расчета движения для клипов из трека
        video_buff = [] # TODO: для оптимизации заменить на queue

        for frameidx, result in enumerate(self.detector.predict(source, self.stream, show)):
            activities = {} # dict с моментальными результатами классификации (заполняется не на каждый кадр, но есть заглушки из пустых строк '')
            boxes, img = result.boxes.data, result.orig_img
            
            # обновить видео буфер
            video_buff.append(img)
            if len(video_buff) == BUFF:
                video_buff.pop(0) 

            # разобрирает детекцию на треки и обрабатывает
            for obj in boxes.tolist():
                box, id, cls = obj[:4], obj[4], obj[6]
                if not id: continue

                activity = ''
                c = tracks_counter.get(id, 0) + 1
                if c < CLST:
                    # обновить счетчик
                    tracks_counter[id] = c
                else:
                    # предсказать активность для трека
                    _, activity = self.classifier.predict(_track2clip(tracks[id], video_buff))
                    # сбросить счетчик
                    tracks_counter[id] = 0 
                # положить в трек (номер фрейма, бокс)
                activities[id] = activity
                tracks[id] = tracks.get(id, []) + [(frameidx, box, activity)]

            if self.stream:
                yield frameidx, result, activities

        self.last_tracks = tracks
        if not self.stream:
            return tracks
    
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


if __name__ == '__main__':
    c = Controller(True)
    c.predict('/Users/samedi/Downloads/613iFGJm0HM_1906_04.mp4', show=True)