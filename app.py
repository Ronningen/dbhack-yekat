import os
import sys

import numpy as np
import pandas as pd
from functools import wraps
import torch
import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QMessageBox, QCheckBox
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage


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

class ImageGallery(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Приложение для автоматической разметки лебедей"
        self.current = 0
        self.fname = ''
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QIcon())
        self.setWindowTitle('icon')

        vbox = QVBoxLayout()

        topButtons = QHBoxLayout()
        btnOpenImages = NoFocusButton("Загрузить изображения")
        btnUseModel = NoFocusButton("Сохранить в .csv")
        self.flagBox = QCheckBox("Детекция")
        self.flagBox.setFont(QFont('Arial', 14))

        mainWindow = QVBoxLayout()
        self.canvas = QLabel("Добро пожаловать!")
        self.canvas.setFont(QFont('Arial', 22))
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.label = QLabel("")
        # self.label.setFont(QFont('Palatino', 22))
        # self.label.setAlignment(Qt.AlignCenter)

        arrows = QHBoxLayout()
        btnPrevImage = NoFocusButton("Назад")
        btnNextImage = NoFocusButton("Вперед")

        topButtons.addWidget(btnOpenImages)
        topButtons.addWidget(self.flagBox)
        topButtons.addWidget(btnUseModel)
        vbox.addLayout(topButtons)

        # mainWindow.addWidget(self.label)
        mainWindow.addWidget(self.canvas)
        vbox.addLayout(mainWindow)

        arrows.addWidget(btnPrevImage)
        arrows.addWidget(btnNextImage)
        vbox.addLayout(arrows)
        self.setLayout(vbox)

        btnOpenImages.clicked.connect(self.getImage)
        btnUseModel.clicked.connect(self.useModel)

        btnNextImage.clicked.connect(lambda: self.nextImage())
        btnPrevImage.clicked.connect(lambda: self.prevImage())

        btnOpenImages.setFocus()

        self.show()

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        exit()

    def show_popup_window(self, error):
        msg = QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText(error)
        msg.exec_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left: self.prevImage()
        elif event.key() == Qt.Key.Key_Right: self.nextImage()

    def showingImage(self, i):
        imagePath = self.fname[0][i]
        ####
        img = cv2.imread(imagePath)
        input_size = (640,640)
        ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR)

        self.drawer.set_image(img[:])
        img = img[:, :, ::-1]
        name_conf_bb_list = self.detector_model(img)  # BGR to RGB

        if self.classifier_model:
            conf_class = self.classifier_model(img)
            res = torch.argmax(conf_class)
            print(res)

        max_conf = 0
        pred_name = 0
        for cls_name, conf, *bbox in name_conf_bb_list:
            bbox = np.array(bbox).reshape(-1, 2)
            if self.flagBox.isChecked():
                self.drawer.draw_bbox(bbox, cls_name, conf)
            if conf > max_conf:
                max_conf = conf
                pred_name = cls_name


        self.drawer.draw_text(f"{pred_name} {max_conf:.2f}")

        # self.drawer.show()

        img = self.drawer.image
        pixmap = QPixmap.fromImage(QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped())

        # pixmap = pixmap.scaled(600,400,Qt.KeepAspectRatio)
        self.canvas.setPixmap(pixmap)

        return pred_name

    def getImage(self,*args, **kwargs):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file', os.path.expanduser("~/Desktop"), "Image files (*.jpg *.gif *.jpeg)")
        try:
            imagePaths = self.fname[0]
            # Iterate over selected file paths
            for imagePath in imagePaths:
                # Check if the file has an image extension
                if not any(imagePath.lower().endswith(ext) for ext in ['.jpg', '.gif', '.jpeg']):
                    self.show_popup_window('Добавлено не фото! можно добавлять только фото.')
                    return

            # All files are images
            self.showOneImage()
        except IndexError as e:
            # Handle index error if no files were selected
            pass

        self.canvas.setFocus()

    @check_file_loaded
    def nextImage(self,*args, **kwargs):
        """След кадр"""
        try:
            if self.current >= len(self.fname[0]) - 1:
                self.show_popup_window('Это последнее изображение!')
            else:
                self.current += 1
                self.showOneImage()

        except IndexError as e:  # сначала загрузите датасет функция
            print(e)

    @check_file_loaded
    def prevImage(self,*args, **kwargs):
        """Пред кадр"""
        if self.current > 0:
            self.current -= 1
            self.showOneImage()
        else:
            self.show_popup_window('Это первое изображение!')

    def showOneImage(self,*args, **kwargs):
        pred_name = self.showingImage(self.current)
        # self.label.setText(pred_name)

    @check_file_loaded
    def useModel(self,*args, **kwargs):
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
    window = ImageGallery()
    sys.exit(App.exec())