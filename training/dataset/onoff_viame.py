"""
Модуль содержит код для обработки разметки виама и возврата работающей/простаивающе техники
"""
import os

import cv2
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class OnOffViame(Dataset):
    """
    Класс для работы боксами разметки viame,
    классы должны быть названы по след формату [class_name]_[on (если работает)]
    например: truck_on_1, truck_2
    """
    col_id_to_name = {
        0: 'track_id', # id трека
        1: 'time_frame',
        2: 'frame',  # номер кадра
        3: 'x1', 4: 'y1', 5: 'x2', 6: 'y2',  # координаты бокса
        7: "detection_confidence",
        8: "target length (0 or -1 if invalid)",
        9: "name",  # имя класса
        10: "scope",
        11: 'attributes',
        12: "file",  # путь к видео файлу
        13: "worked",  # работает ли техника
        14: 'img_w', 15: 'img_h',  # размеры кадра
        16: 'w', 17: 'h'  # размеры бокса
    }
    def __init__(self, path_dir_dataset):
        """
        :param path_dir_dataset: путь к папке с видео файлами и разметкой в формате viame
        """
        self.path_dir_dataset = path_dir_dataset
        self.df, files_mp4 = self.get_big_df_from_viame(path_dir_dataset)
        self.video_file_list = files_mp4

    def _csv2df(self, path_csv, path_mp4):
        """read csv viame file to datafreme """
        with open(path_csv, mode='r', encoding='utf-8') as f:
            _desc = f.readline() + f.readline()  # первый две строки содержат описание колонок
            df = pd.read_csv(f, header=None, names=self.col_id_to_name.values(), engine='python')
            df.sort_values(by='frame', inplace=True)
            df[['name', 'worked']] = df.name.str.split(pat='_', n=1, expand=True)
            df['worked'] = df['worked'].str.startswith('on')
            df[['x1', 'x2', 'y1', 'y2', 'frame', 'track_id']] = df[['x1', 'x2', 'y1', 'y2', 'frame', 'track_id']].astype(int)
            df.w = df.x2 - df.x1
            df.h = df.y2 - df.y1

            df.file = path_mp4
            df.img_w, df.img_h = self._get_video_wh(path_mp4)
        return df

    def _get_video_wh(self, path_video):
        """ Возвращает размер кадра w h из видео """
        video = cv2.VideoCapture(path_video)  # Открываем видеофайл
        if not video.isOpened():
            raise ValueError("Не удалось открыть видеофайл")
        # Получаем размеры кадра
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()  # Закрываем видеофайл
        return int(width), int(height)

    def get_big_df_from_viame(self, path_dir_dataset):
        """ Читает все файлы разметки и возвращает 1 большой датафрейм и список видео файлов. """
        # get list target files
        files = os.listdir(path_dir_dataset)
        files_base = [file.rsplit('.', 1)[0] for file in files if file.endswith('.mp4')]
        files_mp4 = []
        files_csv = []
        for file in files_base:
            path_mp4 = os.path.join(path_dir_dataset, file + '.mp4')
            path_csv = os.path.join(path_dir_dataset, file + '.csv')
            # if not os.path.exists(path_mp4): не нужно
            if not os.path.exists(path_csv):
                print(f'file not found: {path_csv}')
                continue
            files_mp4.append(path_mp4)
            files_csv.append(path_csv)

        df_list = []
        for i in tqdm(range(len(files_csv)), desc="reading csv files"):
            df_list.append(self._csv2df(files_csv[i], files_mp4[i]))
        df = pd.concat(df_list)
        return df, files_mp4

    def get_clips_from_video(self, path_video, track_id_list=None, bar_name='') -> dict:
        """ Возвращает кроп видео ролик
        :param track_id_list: None - возвращает все клипы, list возвращает только указанные треки
        :return {track_id: list_crop_frames}
        """
        df = self.df[self.df.file == path_video]
        if track_id_list is not None and type(track_id_list) is list:
            df = df[df.track_id.isin(track_id_list)]

        track_id_list = list(df.track_id.unique())
        if len(track_id_list) == 0:
            print("Треки в видео не найдены")
            return None

        #####
        # Внимание! Возможно не обходимо делать клипы только для последовательных кадров.
        # Сейчас клип собирается из всех кадров трека
        #####

        # Создаем словарь для хранения координат кропов для каждого трека
        track_crops = {}

        # Вычисляем координаты кропов для каждого трека
        for track_id in track_id_list:
            track_df = df[df.track_id == track_id]  # Фильтруем только данные для текущего трека
            # Получаем минимальные и максимальные координаты боксов трека
            min_x = track_df['x1'].min()
            min_y = track_df['y1'].min()
            max_x = track_df['x2'].max()
            max_y = track_df['y2'].max()
            track_crops[track_id] = (min_x, min_y, max_x, max_y)

        # Читаем видеофайл
        video = cv2.VideoCapture(path_video)
        if not video.isOpened():
            print(f"Не удалось открыть видеофайл: {path_video}")
            return None

        # Инициализируем словарь для хранения кропов каждого трека на каждом кадре
        track_clips = {track_id: [] for track_id in track_id_list}

        # Получаем общее количество кадров видео
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Читаем и обрабатываем каждый кадр видео
        for frame_num in tqdm(range(total_frames), desc=bar_name, unit="frame"):
            ret, frame = video.read()
            if not ret:
                break

            # Обрабатываем каждый трек
            for track_id, crop_coords in track_crops.items():
                # кропим и добавляем в лист
                min_x, min_y, max_x, max_y = crop_coords
                clip_frame = frame[min_y:max_y, min_x:max_x]
                track_clips[track_id].append(clip_frame)

        video.release()
        return track_clips

    def save_all_clips(self, path_output):
        """
        Создает в папке path_output две дирректории:
            on - для работающей техники worked == True
            off  для не работающей
        Видео ролики именуются по след шаблину [имя файла из df.file]_[name]_[track_id]_[on или off].mp4
        """

        # Создаем директории "on" и "off"
        on_dir = os.path.join(path_output, "on")
        off_dir = os.path.join(path_output, "off")
        os.makedirs(on_dir, exist_ok=True)
        os.makedirs(off_dir, exist_ok=True)

        for i, video_path in enumerate(self.video_file_list):
            # print(f"[{i+1}/{len(video_path)}", end=' ')
            track_frames = self.get_clips_from_video(video_path, track_id_list=None,
                                                     bar_name=f'{i+1}/{len(self.video_file_list)} processing')

            # Извлекаем имя файла без расширения из video_path
            file_name = os.path.splitext(os.path.basename(video_path))[0]

            for track_id, clip_frames in tqdm(track_frames.items(), desc='saving'):
                # Проверяем значение self.df.worked для данного video_path и track_id
                track_info = self.df[(self.df.file == video_path) & (self.df.track_id == track_id)]
                worked = track_info["worked"].iloc[0]

                # Определяем путь для сохранения видео в соответствующую директорию
                save_path = os.path.join(on_dir if worked else off_dir ,
                                         f"{file_name}_"
                                         f"{track_info['name'].iloc[0]}_"
                                         f"{track_id}_"
                                         f"{'on' if worked else 'off'}.mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 30,
                                      (clip_frames[0].shape[1], clip_frames[0].shape[0]))
                # Записываем каждый кадр кропа в видео
                for frame in clip_frames:
                    out.write(frame)
                out.release()

        print("Сохранение видео роликов завершено.")



if __name__ == "__main__":
    path = r"G:\Мой диск\5min\get"
    ds = OnOffViame(path_dir_dataset=path)
    path_out = r"C:\workspace\hakaton\hak2023_Digital_breakthrough_ekat\ds_onoff"
    ds.save_all_clips(path_out)