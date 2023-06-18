"""
    GUI scrypt
"""

import os
import sys
from functools import wraps

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QMessageBox, QCheckBox
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage

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
        self.model = Controller(True)

    def InitWindow(self):
        self.setWindowIcon(QIcon())
        self.setWindowTitle('icon')
        vbox = QVBoxLayout()

        btnOpen = NoFocusButton("Загрузить")
        btnOpen.clicked.connect(self.getFile)
        btnSave = NoFocusButton("Сохранить csv")
        btnSave.clicked.connect(self.saveFile)

        topButtons = QHBoxLayout()
        topButtons.addWidget(btnOpen)
        topButtons.addWidget(btnSave)
        vbox.addLayout(topButtons)

        self.canvas = QLabel("Добро пожаловать!")
        self.canvas.setFont(QFont('Arial', 22))
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mainWindow = QVBoxLayout()
        mainWindow.addWidget(self.canvas)
        vbox.addLayout(mainWindow)

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

    def getFile(self,*args, **kwargs):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.path.expanduser("~/Desktop"), "Image files (*.jpg *.gif *.jpeg)")
        try:
            files = self.fname[0]
            for file in files:
                if not any(file.lower().endswith(ext) for ext in ['.jpg', '.gif', '.jpeg']):
                    self.show_popup_window('Добавлено не фото! можно добавлять только фото.')
                    return
                else:
                    self.model.predict(file) # TODO
        except IndexError as e:
            pass

        self.canvas.setFocus()

    @check_file_loaded
    def saveFile(self, *args, **kwargs): # TODO
        savefname = QFileDialog.getSaveFileName(self, "Save file", os.path.expanduser("~/Desktop"), ".csv")
        d = {'кликун':0, 'малый':0, 'щипун':0}
        n = len(self.fname[0])
        df = pd.DataFrame({'фото':['']*n,'вид':['']*n})
        for i in range(n):
            pred = self.showingImage(i)
            d[pred] += 1
            df.at[i,'фото'] = self.fname[0][i]
            df.at[i,'вид'] = pred
        self.show_popup_window(f"подсчет фото - кликун: {d['кликун']}, малый: {d['малый']}, щипун: {d['щипун']}")
        df.to_csv(savefname[0]+'.csv')
        self.current = n-1


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())