"""
    GUI scrypt
"""

import os
import sys
from functools import wraps
import json
import operator

import cv2
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

import pandas as pd
from ultralytics.yolo.engine.results import Results
from back.controller import Controller
from back.custom_plot import cuctom_plot


def frame2time(frame: int, *args, **kwargs) -> str:
    return str(frame)


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
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QIcon())
        self.setWindowTitle('занятость техники')
        vbox = QVBoxLayout()

        # кнопки
        btnOpen = NoFocusButton("Загрузить")
        btnOpen.clicked.connect(self.getFile)
        btnSave = NoFocusButton("Сохранить .json")
        btnSave.clicked.connect(self.saveFile)
        topButtons = QHBoxLayout()
        topButtons.addWidget(btnOpen)
        topButtons.addWidget(btnSave)
        vbox.addLayout(topButtons)

        # видеовывод
        self.vlabel = QLabel()
        vbox.addWidget(self.vlabel)

        # статистика
        mainWindow = QHBoxLayout()

        self.canvas = pg.PlotWidget()
        self.canvas.setBackground('w')
        self.data = self.canvas.plot([], [], pen=pg.mkPen(color=(200, 50, 50), width=2))
        self.canvas.setYRange(0,1)
        self.datay = []

        self.onlabel = QLabel()
        self.onlabel.setWordWrap(True)
        self.offlabel = QLabel()
        self.offlabel.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(QLabel(text='работают:'))
        left.addWidget(self.onlabel)
        right = QVBoxLayout()
        right.addWidget(QLabel(text='простаивают:'))
        right.addWidget(self.offlabel)

        mainWindow.addLayout(left)
        mainWindow.addLayout(right)
        vbox.addLayout(mainWindow)

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
        try:
            frameidx, result, activities = self.yielder.__next__()
            img = Image.fromarray(cuctom_plot(result, conf=True, line_width=4, labels=True, boxes=True)[:,:,::-1])
            img.thumbnail((1028, 1000), Image.ANTIALIAS)
            self.vlabel.setPixmap(QPixmap.fromImage(ImageQt(img)))

        except StopIteration:
            tracks = self.model.last_tracks
            clss = self.model.last_tracks_cls
            for id in tracks:
                event = {'id': id, 'class': max(clss[id], key=clss[id].get), 
                         'start': frame2time(tracks[id][0][0]), 'end': frame2time(tracks[id][-1][0]), 
                         'notes':[], 'activity events':[], 'trajectory':[]}
                for frameidx, box, activity in tracks[id]:
                    event['trajectory'].append([(box[0]+box[2])/2, (box[1]+box[3])/2])

                self.json[-1]['events'].append(event)

            self.timer.stop()
            self.show_popup_window('Обработка видео сохранена, вы можете сохранить файл')

    def getFile(self, *args, **kwargs):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.path.expanduser("~/Desktop"), "Video files (*.mp4)")
        try:
            files = self.fname[0]
            for file in files:
                if not any(file.lower().endswith(ext) for ext in ['mp4']):
                    self.show_popup_window('Добавлено не видео! можно добавлять только видео.')
                    return
                else:
                    self.json.append({'path':file, 'events':[]})
                    self.yielder = self.model.predict(file, show=False)
                    self.timer.start()
                    
        except IndexError as e:
            pass

    @check_file_loaded
    def saveFile(self, *args, **kwargs): # TODO
        if len(self.json) == 0:
            self.show_popup_window("Загрузите видео и дождитесь окончания обработки!")
            return
        
        savefname = QFileDialog.getSaveFileName(self, "Save file", os.path.expanduser("~/Desktop"), ".json")[0]
        json.dump(self.json, open(savefname, "w", encoding ='utf8'), indent=4)

if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())