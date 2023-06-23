"""
    GUI scrypt
"""

import os
import sys
from functools import wraps

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QMessageBox, QCheckBox
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

import pandas as pd
from controller import Controller


def check_file_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fname:
            self.show_popup_window('Сначала загрузите файлы!')
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
        self.InitWindow()
        self.model = Controller(stream=True)

    def InitWindow(self):
        self.setWindowIcon(QIcon())
        self.setWindowTitle('занятость техники')
        vbox = QVBoxLayout()

        btnOpen = NoFocusButton("Загрузить")
        btnOpen.clicked.connect(self.getFile)
        btnSave = NoFocusButton("Сохранить csv")
        btnSave.clicked.connect(self.saveFile)

        topButtons = QHBoxLayout()
        topButtons.addWidget(btnOpen)
        topButtons.addWidget(btnSave)
        vbox.addLayout(topButtons)

        mainWindow = QVBoxLayout()
        self.canvas = pg.PlotWidget()
        self.canvas.setBackground('w')
        self.data = self.canvas.plot([], [], pen=pg.mkPen(color=(200, 50, 50), width=2))
        self.canvas.setYRange(0,1)
        self.datay = []
        mainWindow.addWidget(self.canvas)
        vbox.addLayout(mainWindow)

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.updata)

        self.setLayout(vbox)
        btnOpen.setFocus()
        self.show()

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        exit()

    def show_popup_window(self, error):
        msg = QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText(error)
        msg.exec_()

    def updata(self):
        result = self.yielder.__next__()
        self.datay.append(sum(result.values())/len(result))
        self.data.setData(range(len(self.datay)), self.datay)

    def getFile(self, *args, **kwargs):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.path.expanduser("~/Desktop"), "Video files (*.mp4)")
        try:
            files = self.fname[0]
            for file in files:
                if not any(file.lower().endswith(ext) for ext in ['mp4']):
                    self.show_popup_window('Добавлено не видео! можно добавлять только видео.')
                    return
                else:
                    self.yielder = self.model.predict(file, show=True)
                    self.timer.start()


        except IndexError as e:
            pass

    @check_file_loaded
    def saveFile(self, *args, **kwargs): # TODO
        pass
        # savefname = QFileDialog.getSaveFileName(self, "Save file", os.path.expanduser("~/Desktop"), ".csv")
        # d = {'кликун':0, 'малый':0, 'щипун':0}
        # n = len(self.fname[0])
        # df = pd.DataFrame({'фото':['']*n,'вид':['']*n})
        # for i in range(n):
        #     pred = self.showingImage(i)
        #     d[pred] += 1
        #     df.at[i,'фото'] = self.fname[0][i]
        #     df.at[i,'вид'] = pred
        # self.show_popup_window(f"подсчет фото - кликун: {d['кликун']}, малый: {d['малый']}, щипун: {d['щипун']}")
        # df.to_csv(savefname[0]+'.csv')
        # self.current = n-1


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())