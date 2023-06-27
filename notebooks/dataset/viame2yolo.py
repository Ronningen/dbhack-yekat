# модуль содежит код для генерации датасета в формате yolo из разметки viame

import os.path
import random
import string
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


class ViameToYolo:
    """Читает csv Viame в df и создает файлы с кадрами и разметкой в удобном формате"""
    col_id_to_name = {
        0: 'track_id', 1: 'time_frame', 2: 'frame', 3: 'TL_x', 4: 'TL_y', 5: 'BR_x', 6: 'BR_y',
        7: "detection_confidence", 8: "target length (0 or -1 if invalid)",
        9: "class_name", 10: "scope",
        11: 'attributes', 12: "file"
    }

    def __init__(self, path_input_dir, path_out_dir=None,
                 img_size_out=None, class_names: dict = [],
                 class_ignore: list = [],
                 class_accumulation={},
                 step_frame=1):
        """
        Example: class_accumulation = {"number":['0','1','2',..], "car":['truck', 'bus', ..]}
        """
        self.step = step_frame
        self.img_size_out = img_size_out
        # self.path_csv = path_csv
        # self.uniq_name = os.path.basename(path_csv).split("\\")[-1].split(".")[-2] + "_"  # имя файла без расширения
        # self.path_video = path_video
        self.path_input_dir = path_input_dir
        if path_out_dir is None:
            self.path_out_dir = path_input_dir + "_out_" + datetime.now().strftime("%m%d%H%M%S")
        else:
            self.path_out_dir = path_out_dir
        self.class_ignore = class_ignore
        self.class_names = class_names
        self.class_name2class_id = {name: i for i, name in enumerate(class_names)}
        self.class_id2class_name = {i: name for i, name in enumerate(class_names)}
        self.len_class = len(self.class_names)
        self.class_accumulation = class_accumulation
        print(f"class_id2class_name = {self.class_id2class_name}")

    def _row_to_bbox_yolo_format(self, row, size_w=None, size_h=None):
        """Возвращает относительные координаты центра и ширину, высоту бокса"""
        if size_w is not None:
            self.size_w = size_w
        if size_h is not None:
            self.size_h = size_h
        if self.size_h is None or self.size_w is None:
            raise ValueError("self.size_h is None or self.size_w is None")

        w = (row['BR_x'] - row['TL_x']) / self.size_w
        h = (row['BR_y'] - row['TL_y']) / self.size_h
        xc = row['TL_x'] / self.size_w + w / 2
        yc = row['TL_y'] / self.size_h + h / 2
        class_name = str(row['class_name'])
        class_id = self._get_class_id(class_name)

        return class_id, xc, yc, w, h

    def _row_to_poly_xy(self, row, size_w=None, size_h=None):
        """Возвращает относительные координаты полигона, список [x, y, x, y, x, y, ...]"""
        if size_w is not None:
            self.size_w = size_w
        if size_h is not None:
            self.size_h = size_h
        if self.size_h is None or self.size_w is None:
            raise ValueError("self.size_h is None or self.size_w is None")

        attr = row['attributes']  # (poly) 918 421 939 420 945 478 907 478 918 421
        lst_pnts = [float(z) for z in attr.split()[1:]]
        lst_pnts = [z / self.size_w if i % 2 == 0 else z / self.size_h  # переводим в диапазон 0 1 относительно w h
                    for i, z in enumerate(lst_pnts)]

        class_name = str(row['class_name'])
        class_id = self._get_class_id(class_name)
        return [class_id] + lst_pnts

    def _row_to_poly_4pts(self, row, size_w=None, size_h=None):
        """Возвращает относительные координаты полигона, список [x, y, x, y, x, y, ...]"""
        if size_w is not None:
            self.size_w = size_w
        if size_h is not None:
            self.size_h = size_h
        if self.size_h is None or self.size_w is None:
            raise ValueError("self.size_h is None or self.size_w is None")

        attr = row['attributes']  # (poly) 918 421 939 420 945 478 907 478 918 421
        lst_pnts = [float(z) for z in attr.split()[1:]]
        lst_pnts = [z / self.size_w if i % 2 == 0 else z / self.size_h  # переводим в диапазон 0 1 относительно w h
                    for i, z in enumerate(lst_pnts)]

        class_name = str(row['class_name'])
        class_id = self._get_class_id(class_name)
        return [class_id] + lst_pnts[:8]  # return class_id and only 4 xy points

    def create_ds_polygon_yolo(self, path_video, path_csv, bar=True, proc_image=True):
        if not os.path.exists(path_csv) or not os.path.exists(path_video):
            raise FileNotFoundError(f"not exists {path_csv} or {path_video}")

        # created out dirs
        self._create_output_dirs()
        path_labels = os.path.join(self.path_out_dir, 'labels')
        path_images = os.path.join(self.path_out_dir, 'images')

        df = self._csv2df(path_csv)
        df = df.dropna(subset=['attributes'])
        df = df[~df['class_name'].isin(self.class_ignore)]

        # images
        if proc_image:
            self._save_frames(path_video, set(df['frame']), bar=True)

        # labels
        uniq_name = self._get_base_name(path_csv)
        # class_index = 0
        for _, row in df.dropna(subset=['attributes']).iterrows():
            with open(os.path.join(path_labels, uniq_name + str(row['frame']) + ".txt"), "a") as f:
                print(*self._row_to_poly_xy(row), file=f)
        print("Labels created from " + path_csv)

    def create_ds_bbox_yolo(self, path_video, path_csv, bar=True, proc_image=True):
        """Создает файлы для обучения в формате YOLO. out_dir/images/...jpg out_dir/labels/...txt
        пример txt:
        0 0.11585937499999999 0.47625 0.04890625 0.034166666666666616
        0 0.1703125 0.5284722222222222 0.015625 0.05694444444444444
        """
        if not os.path.exists(path_csv) or not os.path.exists(path_video):
            raise FileNotFoundError(f"not exists {path_csv} or {path_video}")

        # created out dirs
        self._create_output_dirs()
        path_labels = os.path.join(self.path_out_dir, 'labels')
        path_images = os.path.join(self.path_out_dir, 'images')

        df = self._csv2df(path_csv)
        df = df[~df['class_name'].isin(self.class_ignore)]
        df = df[df.frame.apply(lambda x: (x % self.step) == 0)]

        # images
        if proc_image:
            self._save_frames(path_video, set(df['frame']), bar=True)

        # labels
        # class_index = 0
        uniq_name = self._get_base_name(path_csv)
        for _, row in df.iterrows():
            with open(os.path.join(path_labels, uniq_name + str(row['frame']) + ".txt"), "a") as f:
                print(*self._row_to_bbox_yolo_format(row), file=f)
        print("Labels created from " + path_csv)

    def create_ds_polygon_4pts_yolo(self, path_video, path_csv, bar=True, proc_image=True):
        if not os.path.exists(path_csv) or not os.path.exists(path_video):
            raise FileNotFoundError(f"not exists {path_csv} or {path_video}")

        # created out dirs
        self._create_output_dirs()
        path_labels = os.path.join(self.path_out_dir, 'labels')
        path_images = os.path.join(self.path_out_dir, 'images')

        df = self._csv2df(path_csv)
        df = df.dropna(subset=['attributes'])
        df = df[~df['class_name'].isin(self.class_ignore)]

        # images
        if proc_image:
            self._save_frames(path_video, set(df['frame']), bar=True)

        # labels
        uniq_name = self._get_base_name(path_csv)
        if not proc_image:
            # save size
            cam = cv2.VideoCapture(path_video)
            self.size_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.size_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cam.release()
        # class_index = 0
        for _, row in df.dropna(subset=['attributes']).iterrows():
            with open(os.path.join(path_labels, uniq_name + str(row['frame']) + ".txt"), "a") as f:
                print(*self._row_to_poly_4pts(row), file=f)
        print("Labels created from " + path_csv)

    def _create_output_dirs(self):
        path_labels = os.path.join(self.path_out_dir, 'labels')
        path_images = os.path.join(self.path_out_dir, 'images')
        os.makedirs(path_labels, exist_ok=True)
        os.makedirs(path_images, exist_ok=True)

    def _save_frames(self, path_video, frames, bar=True):
        """Сохраняет кадры с номерами из frames на диск. Если указаны self.img_size_out происходит ресайз,
            записывает актуальные self.size_w, self.size_h"""

        cam = cv2.VideoCapture(path_video)
        uniq_name = self._get_base_name(path_video)
        # save size
        self.size_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.size_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"start reading {path_video}; frames={count_frames}, size={self.size_w},{self.size_h}")
        currentframe = 0
        if type(self.img_size_out) not in (tuple, int):
            print("img_size_out имеет не допустимые параметры:", self.img_size_out, "; ресайз не применяется;")
        for i in tqdm(range(0, int(count_frames)), disable=not bar):
            ret, frame = cam.read()
            if ret:
                if currentframe in frames:
                    if self.img_size_out:
                        if type(self.img_size_out) is tuple:
                            frame = cv2.resize(frame, self.img_size_out, interpolation=cv2.INTER_CUBIC)
                        elif type(self.img_size_out) is int:
                            max_shape = max(self.size_w, self.size_h)
                            scale = self.img_size_out / max_shape
                            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    path_img = os.path.join(self.path_out_dir, "images", uniq_name + str(currentframe) + ".jpg")
                    cv2.imwrite(path_img, frame)
                currentframe += 1
            else:
                break
        if count_frames != currentframe:
            print(f"=> Считаны не все кадры, {currentframe} из {count_frames:.0f}.")
        cam.release()
        cv2.destroyAllWindows()

    def _csv2df(self, path_csv):
        with open(path_csv, mode='r', encoding='utf-8') as f:
            _desc = f.readline() + f.readline()  # первый две строки содержат описание колонок
            df = pd.read_csv(f, header=None, names=self.col_id_to_name.values(), engine='python')
            df.sort_values(by='frame', inplace=True)
            df.file = path_csv
        return df

    @staticmethod
    def _get_base_name(path_file):
        return os.path.basename(path_file).split("\\")[-1].split(".")[-2] + "_"  # имя файла без расширения + "_"

    def _get_class_id(self, class_name):
        # пред обработка, берем имя класса до _
        class_name = class_name.split('_', 1)[0]

        # изменяем имя класса если этого требует аккумулятор
        for key, list_class in self.class_accumulation.items():
            if class_name in list_class:
                class_name = key
                break

        if class_name in self.class_name2class_id:
            return self.class_name2class_id[class_name]
        # create new class id
        class_id = self.len_class
        self.len_class += 1
        self.class_name2class_id[class_name] = class_id
        self.class_id2class_name[class_id] = class_name
        return class_id

    def get_all_class(self):
        files = [p for p in os.listdir(self.path_input_dir) if p.endswith('.csv')]
        # print(files)
        df_list = []
        for i in range(len(files)):
            path_csv = os.path.join(self.path_input_dir, files[i])
            df_list.append(self._csv2df(path_csv))
        # df = df[~df['class_name'].isin(self.class_ignore)]
        df = pd.concat(df_list)
        del df_list
        class_name = pd.unique(df.sort_values(by='class_name')['class_name'])
        # print("Все классы:", class_name)
        print("Только имена классов до _ :", set(cls.split("_", 1)[0] for cls in class_name))

    def start_processing(self, out_format='bbox_yolo', bar=True, proc_image=True):
        """"
        out_format - формат разметки
        proc_image - True, сохранять кадры из видео.
        bar - True, прогресбар
        out_format может быть: 'bbox_yolo', 'poly_yolo', 'poly4pts_yolo'
        """
        handler = {
            'bbox_yolo': self.create_ds_bbox_yolo,
            'poly_yolo': self.create_ds_polygon_yolo,
            'poly4pts_yolo': self.create_ds_polygon_4pts_yolo
        }
        # if out_format == 'bbox_yolo':
        #     handler = self.create_ds_bbox_yolo
        # elif out_format == 'poly_yolo':
        #     handler = self.create_ds_polygon_yolo
        # elif out_format == 'poly4pts_yolo'

        files = os.listdir(self.path_input_dir)
        files = [file for file in files if file.endswith(".mp4")]
        # print(files)

        for i, file in enumerate(files):
            # if not file.endswith('.mp4'):
            #     continue
            video = os.path.join(self.path_input_dir, file)
            csv = os.path.join(self.path_input_dir, file.rsplit('.', 1)[0] + ".csv")
            if not os.path.exists(csv):
                print(f"{video} пропущен т.к. нет разметки")
                continue
            handler[out_format](video, csv, bar=True, proc_image=proc_image)
            print(f'Обработано {i + 1} видео из {len(files)}')
        print('Finish')
        self.print_infoproc_to_file()

    def print_infoproc_to_file(self):
        with open(os.path.join(self.path_out_dir, "class_name.txt"), "w") as f:
            print(self.class_id2class_name, file=f)
            print(list(self.class_name2class_id), "; count =", self.len_class, file=f)


if __name__ == "__main__":
    # all class name ['0', '1', '2']
    #class_ignore = ['0']

    out_dir = "D:/db/test_db"
    id_dir = r'D:\viame_dataset'
    vr = ViameToYolo(id_dir, img_size_out=None, path_out_dir=out_dir)
    vr.start_processing(out_format='bbox_yolo')
    print(vr.class_id2class_name)
    print(list(vr.class_name2class_id), "; count =", vr.len_class)
    with open(os.path.join(out_dir, "class_name.txt"), "w") as f:
        print(vr.class_id2class_name, file=f)
        print(list(vr.class_name2class_id), "; count =", vr.len_class, file=f)
    vr.get_all_class()

    # все классы
    # out_dir = "0207_box_yolo_ALL_class"
    # vr = ViameReader("data", img_size_out=1280, path_out_dir=out_dir)
    # vr.start_processing(out_format='bbox_yolo', proc_image=True)
    # print(vr.class_id2class_name)
    # print(list(vr.class_name2class_id), "; count =", vr.len_class)
    # with open(os.path.join(out_dir, "class_name.txt"), "w") as f:
    #     print(vr.class_id2class_name, file=f)
    #     print(list(vr.class_name2class_id), "; count =", vr.len_class, file=f)
    # vr.get_all_class()

