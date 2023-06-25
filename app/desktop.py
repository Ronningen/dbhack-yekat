"""
    GUI and json forming scrypt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from functools import wraps
import json
from datetime import timedelta

import cv2
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage

# from pyqtgraph import PlotWidget, plot
# import pyqtgraph as pg

from ultralytics.yolo.engine.results import Results
from back.controller import Controller
from back.custom_plot import cuctom_plot


NAMES = {0: 'подъёмный кран', 
         1: 'экскаватор', 
         2: 'грузовой автомобиль', 
         3: 'трактор', 
         4: 'автобетоносмеситель', 
         5: 'асфальтоукладчик', 
         6: 'мини погрузчик', 
         7: 'холодный фрез'}

def frame2time(frame, fps):
    return str(timedelta(seconds=frame/fps))

def box_in(box, width, height, margin = 0.05):
    return not (box[0] - width*margin < 0  or box[2] + width*margin > width \
             or box[1] - height*margin < 0 or box[3] + height*margin > height)

def notes_factory(track, total_frames, width, height, luft=6): #TODO
    notes = []

    if track[0][0] <= luft:
        notes.append("был в кадре на начало видео")
    elif box_in(track[0][1], width, height):
        notes.append("появился из-за препятствия")

    if track[-1][0] >= total_frames - luft:
        notes.append("был в кадре в конце видео")
    elif box_in(track[0][1], width, height):
        notes.append("скрылся за препятствием")

    return notes


def check_file_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fname:
            self.show_popup_window('Сначала загрузите видео!')
        else:
            return func(self, *args, **kwargs)
    return wrapper

class NoFocusButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFont(QFont('Arial', 14))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "мониторинг строительных работ"
        self.fname = ''
        self.model = Controller(stream=True)
        self.json = []
        self.tmp_activities = {}
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QIcon())
        self.setWindowTitle('занятость техники')
        vbox = QVBoxLayout()

        # кнопки
        btnOpen = NoFocusButton("Загрузить видео")
        btnOpen.clicked.connect(self.getFile)
        btnSave = NoFocusButton("Сохранить в .json")
        btnSave.clicked.connect(self.saveFile)
        topButtons = QHBoxLayout()
        topButtons.addWidget(btnOpen)
        topButtons.addWidget(btnSave)
        vbox.addLayout(topButtons)

        # видеовывод
        self.vlabel = QLabel(text='Загрузите видео')
        vbox.addWidget(self.vlabel)

        # статистика
        # mainWindow = QHBoxLayout()

        # self.canvas = pg.PlotWidget()
        # self.canvas.setBackground('w')
        # self.data = self.canvas.plot([], [], pen=pg.mkPen(color=(200, 50, 50), width=2))
        # self.canvas.setYRange(0,1)
        # self.datay = []

        # self.onlabel = QLabel()
        # self.onlabel.setWordWrap(True)
        # self.offlabel = QLabel()
        # self.offlabel.setWordWrap(True)

        # left = QVBoxLayout()
        # left.addWidget(QLabel(text='работают:'))
        # left.addWidget(self.onlabel)
        # right = QVBoxLayout()
        # right.addWidget(QLabel(text='простаивают:'))
        # right.addWidget(self.offlabel)

        # mainWindow.addLayout(left)
        # mainWindow.addLayout(right)
        # vbox.addLayout(mainWindow)

        # завязка цикла предсказания на таймер

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.updata)

        self.setLayout(vbox)
        btnOpen.setFocus()
        self.show()

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        exit()

    def show_popup_window(self, text):
        msg = QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText(text)
        msg.exec()

    def updata(self):
        """
            streams current processing frame and accamulates json
        """
        try:
            frameidx, result, activities = self.yielder.__next__()
            for k in activities:
                if activities[k]:
                    self.tmp_activities[k] = activities[k]
            img = Image.fromarray(cuctom_plot(
                result, conf=False, line_width=4, labels=True, boxes=True, font_size=1, states=self.tmp_activities
                )[:,:,::-1])
            img.thumbnail((1028, 1000), Image.LANCZOS)
            self.vlabel.setPixmap(QPixmap.fromImage(ImageQt(img).copy()))

        except StopIteration:
            tracks = self.model.last_tracks
            clss = self.model.last_tracks_cls

            video = cv2.VideoCapture(self.json[-1]['path'])
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)

            for id in tracks:
                event = {'id': int(id), 
                         'class': NAMES[max(clss[id], key=clss[id].get)],
                         'start': frame2time(tracks[id][0][0], fps), 
                         'end': frame2time(tracks[id][-1][0], fps), 
                         'notes': notes_factory(tracks[id], total_frames, width, height),  
                         'summative work time': frame2time(0, fps), 
                         'percentage working': 0,
                         'work events':[], 
                         'trajectory_xy':[]}
                
                state = ''
                for frameidx, box, activity in tracks[id]:
                    event['trajectory_xy'].append([(box[0]+box[2])/2, (box[1]+box[3])/2])
                    # если изменилась активность - меняю состояние и записываю
                    if activity and state != activity:
                        if activity == 'on': 
                            # если первая активность - работа, то начало - 0 кадр
                            event['work events'].append({'start work': frameidx if state else 0})
                        # если до этого работал - записать что перестал
                        elif state == 'on':
                            event['work events'][-1]['stop work'] = frameidx
                        # после записей обновляю состояние
                        state = activity
                # если к концу видео не остановилась работа - считаем что остановилась на последнем кадре
                if state == 'on':
                    event['work events'][-1]['stop work'] = total_frames
                # подсчет времени работы
                work_sum = 0
                for work_event in event['work events']:
                    work_sum += work_event['stop work'] - work_event['start work']
                    work_event['stop work'] = frame2time(work_event['stop work'], fps)
                    work_event['start work'] = frame2time(work_event['start work'], fps)
                event['summative work time'] = frame2time(work_sum, fps)
                event['percentage working'] = int(work_sum/total_frames*100)
                
                self.json[-1]['events'].append(event)

            self.timer.stop()
            self.show_popup_window('Обработка видео завершена, вы можете сохранить статистику')

    def getFile(self, *args, **kwargs):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.path.expanduser("~/Desktop"), "Video files (*.mp4)")
        try:
            files = self.fname[0]
            for file in files:
                if not any(file.lower().endswith(ext) for ext in ['mp4']):
                    self.show_popup_window('Добавлено не видео! можно добавлять только видео.')
                    return
                else:
                    self.tmp_activities = {}
                    self.json.append({'path': file, 'events': []})
                    self.yielder = self.model.predict(file, show=False)
                    self.timer.start()
                    
        except IndexError as e:
            pass

    @check_file_loaded
    def saveFile(self, *args, **kwargs):
        if len(self.json) == 0:
            self.show_popup_window("Загрузите видео и дождитесь окончания обработки!")
            return
        
        savefname = QFileDialog.getSaveFileName(self, "Save file", os.path.expanduser("~/Desktop"), ".json")[0]
        if not savefname.endswith('.json'):
            savefname += '.json'
        json.dump(self.json, open(savefname, "w", encoding ='utf8'), ensure_ascii=False, indent=4)

if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())