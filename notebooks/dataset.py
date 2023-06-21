import random
import time

import imgaug.augmenters as iaa
import cv2
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, WeightedRandomSampler
from trainer.modules import GetOut
from PIL import Image

from scipy import ndimage

def select_uniform_indices(lst, k):
    n = len(lst)
    step = n / k  # Вычисляем шаг между индексами

    indices = []
    for i in range(k):
        index = int(i * step)
        indices.append(index)

    return indices


def rot(frames, angle=(-5,5), mode='onerand'):
    '''allrand, onerand, normalrand'''
    # ang = int(np.random.uniform(*angle))
    n = len(frames) if type(frames) in [list, tuple] else 1
    if mode == 'allrand':
        ang_list = [int(np.random.uniform(*angle)) for _ in range(n)]
    elif mode == 'onerand':
        ang_list = [int(np.random.uniform(*angle))] * n
    elif mode == 'normalrand':
        ang_list = list(range(angle[0], angle[1]+1, (angle[1]-angle[0])//n))

    if type(frames) in [list, tuple]:
        for i, img in enumerate(frames):
            frames[i] = ndimage.rotate(img, ang_list[i])
        return frames
    frames = ndimage.rotate(frames, ang_list[0])
    return frames

def convert_temp_kelvin(frames, temp=(3000, 14000)):
    param = temp
    state = time.time_ns() # все кадры с одними настройками
    if type(frames) in [list, tuple]:
        for i, fr in enumerate(frames):
            frames[i] = iaa.ChangeColorTemperature(param, random_state=state)(images=[fr])[0]
        return frames
    return iaa.ChangeColorTemperature(param, random_state=state)(images=[frames])[0]


def gaussian_noise(frames, scale=(0, 0.2)):
    sc = np.random.uniform(*scale)
    if type(frames) in [list, tuple]:
        aug = iaa.AdditiveGaussianNoise(scale=sc * 255)
        frames = aug(images=frames)
        return frames

    aug = iaa.AdditiveGaussianNoise(scale=sc * 255)
    frames = aug(images=[frames])[0]
    return frames


def consrast_brightness(frames, alpha=(0.5, 2.5), beta=(-20, 100)):
    alpha, beta = np.random.uniform(*alpha), int(np.random.uniform(*beta))
    if type(frames) in [list, tuple]:
        frames = [cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                  for img in frames]
        return frames
    frames = cv2.convertScaleAbs(frames, alpha=alpha, beta=beta)
    return frames


def motion(frames, k=(10, 17)):
    k = int(np.round(np.random.uniform(*k)))
    if type(frames) in [list, tuple]:
        frames = iaa.MotionBlur(k=k, angle=[-90, 90])(images=frames)
        return frames

    frames = iaa.MotionBlur(k=k, angle=[-90, 90])(images=[frames])[0]
    return frames

def mulHueSatur(frames):
    param = (0.5, 1.5)
    state = time.time_ns() # все кадры с одними настройками
    if type(frames) in [list, tuple]:
        for i, fr in enumerate(frames):
            frames[i] = iaa.MultiplyHueAndSaturation(param, random_state=state)(images=[fr])[0]
        return frames
    return iaa.MultiplyHueAndSaturation(param, random_state=state)(images=[frames])[0]

def clouds(frames):
    aug = iaa.Clouds()
    if type(frames) in [list, tuple]:
        frames = aug(images=frames)
        return frames
    return aug(images=[frames])[0]

def fog(frames):
    aug = iaa.Fog()
    if type(frames) in [list, tuple]:
        frames = aug(images=frames)
        return frames
    return aug(images=[frames])[0]

def resize_into(frames, shifted_scale=0.15):
    # shifted_scale = 0.1
    if type(frames) in [list, tuple]:
        h, w, _ = frames[0].shape
        x_sh, y_sh = np.random.rand(2) * np.array([w, h]) * shifted_scale # точки сдвига в пределах 10% изображения
        x_sh, y_sh = int(x_sh), int(y_sh)
        x_end, y_end = np.random.rand(2) * np.array(
            [w, h]) * shifted_scale  # точки сдвига в пределах 10% изображения
        x_end, y_end = int(x_end), int(y_end)
        for i, img in enumerate(frames):
            canva = np.zeros_like(img)
            img = cv2.resize(img, (w - x_sh - x_end, h - y_sh - y_end))
            new_h, new_w, _ = img.shape
            canva[y_sh:y_sh+new_h, x_sh:x_sh+new_w] = img
            frames[i] = canva
        return frames

    h, w, _ = frames.shape
    x_sh, y_sh = np.random.rand(2) * np.array([w, h]) * 0.1  # точки сдвига в пределах 10% изображения
    x_sh, y_sh = int(x_sh), int(y_sh)
    img = frames.copy()
    canva = np.zeros_like(img)
    img = cv2.resize(img, (w - x_sh, h - y_sh))
    new_w, new_h, _ = img.shape
    frames[x_sh:x_sh + new_w, y_sh:y_sh + new_h] = img
    return frames


def cartoon(frames,):
    aug = iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
                  saturation=2.0, edge_prevalence=1.0)
    if type(frames) in [list, tuple]:
        frames = aug(images=frames)
        return frames
    return aug(images=[frames])[0]

def grayscale(frames, scale=(0,1)):
    scale = np.random.uniform(*scale)
    if type(frames) in [list, tuple]:
        res = []
        for img in frames:
            gray = img.astype(float) * np.array([0.299, 0.587, 0.114])
            gray = gray.sum(axis=-1)[..., None]
            diff2gray = img - gray
            img_gray_scale = img - diff2gray * scale
            res.append(img_gray_scale.astype(np.uint8))
        return res
    gray = frames.astype(float) * np.array([0.299, 0.587, 0.114])
    gray = gray.sum(axis=-1)[..., None]
    diff2gray = frames - gray
    img_gray_scale = frames - diff2gray * scale
    return img_gray_scale.astype(np.uint8)

def get_countours(frames):
    # thresh = 100
    if type(frames) not in [tuple, list]:

        img_grey = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
        countour_list = []
        for thresh in [75, 125, 175]:
            # получим картинку, обрезанную порогом
            ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

            # надем контуры
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # создадим пустую картинку
            img_contours = np.zeros(img_grey.shape)

            # отобразим контуры
            cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
            countour_list.append(img_contours)

        # cv2.imshow('contours', np.stack(countour_list, axis=-1))  # выводим итоговое изображение в окно
        #
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return np.stack(countour_list, axis=-1)




class CustomDataset(Dataset):
    # df columns ['relative_path', 'class', 'sequence_name', 'width', 'height','location']
    def __init__(self, df, class_name2id, transform=None,
                 model_extractor=None, device='cuda', low_high=(35, 65), #out_list=None,
                 mode='train', sample_mode='random', half=False, p_flip=0.5):
        """
            low_high - указывает диапазон количества отобранных рандомно кадров из последовательности
                None - будут взяты все кадры.
        """
        self.half = half
        self.p_flip = p_flip
        torch.random.manual_seed(256)
        self.df = df
        self.transform = transform
        self.idx2seq = sorted(list(self.df.sequence_name.unique()))
        self.model_extractor = model_extractor.to(device)
        # self.model_extractor.eval()
        self.device = device
        self.class_name2id = class_name2id
        if type(low_high) is int:
            low_high = (low_high, low_high + 1)
        self.low_high = low_high
        self.mode = mode
        self.sample_mode = sample_mode

        # if out_list is not None:
        #     self.out_list = out_list
        # else:
        #     self.out_list = [None] * 4
        # model_extractor.trunk_output.block4 = \
        #     GetOut(model_extractor.trunk_output.block4, self.out_list, 3)
        # model_extractor.trunk_output.block3 = \
        #     GetOut(model_extractor.trunk_output.block3, self.out_list, 2)
        # model_extractor.trunk_output.block2 = \
        #     GetOut(model_extractor.trunk_output.block2, self.out_list, 1)
        # model_extractor.trunk_output.block1 = \
        #     GetOut(model_extractor.trunk_output.block1, self.out_list, 0)


    def __getitem__(self, index):
        frames, label_id = self.get_rnd_frames_tf(index)
        onehot_label = torch.zeros(4, dtype=torch.float)
        onehot_label[label_id] = 1


        # augmentation
        if random.random() < self.p_flip:
            frames = torch.flip(frames, dims=[3])  # flip horizontal


        if not self.model_extractor:
            return frames, onehot_label

        # # print(frames[0, ...].shape, 213232)
        # m = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # m = -m / std
        # std = 1.0 / std
        # image = frames[1, ...]
        # image = torchvision.transforms.Normalize(mean=m, std=std)(image)
        # # print(image.shape, 213232)
        # image = torchvision.transforms.ToPILImage()(image)
        # cv2.imshow('', np.array(image)[:,:,::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # print(frames.shape, frames[0, 0, 0, 0])

        frames = frames.float().to(self.device)
        # with torch.no_grad():
        if self.half:
            frames = frames.half()
            if self.mode == 'train':
                self.model_extractor.train()
            else:
                self.model_extractor.eval()
        b2, b3, b4 = self.model_extractor(frames)

        # b2 = self.out_list[-3] # 144x28x28
        # b3 = self.out_list[-2]
        # b4 = self.out_list[-1]

        b2_std = torch.std(b2, dim=0)
        b3_std = torch.std(b3, dim=0)
        # b4 = torch.mean(b4, dim=0)
        b4 = b3[np.random.randint(b3.shape[0])]
        # b3_mean = b3[list(range(0, b3.shape[0], 2))].mean(dim=0)
        # b3_mean = b3[[0, b3.shape[0]//2, -1]].mean(dim=0)
        # b3_mean = b3.mean(dim=0)
        # b4_std = torch.std(b4, dim=0)

        # print(b3_std.shape)
        # print(b4_std.shape)
        # out_mean = torch.mean(out, dim=0)
        # feat_std = torch.std(feature, dim=0)

        # return (out_std, feat_std), onehot_label

        # return torch.concatenate([out_std, out_mean], dim=0), onehot_label
        # return out[1] - out[0], onehot_label

        return (b2_std, b3_std, b4), onehot_label


    def get_rnd_frames_tf(self, index):
        """Возвращает из видео набор кадров c уагментацией и трансформацией и с учетом парамера low_high и метку класса.
        """
        # Получаем информацию о текущем элементе
        seq_name = self.idx2seq[index]
        df_seq = self.df[self.df.sequence_name == seq_name]
        if self.low_high is not None:
            count_frame = np.random.randint(*self.low_high)
            count_frame = min(count_frame, len(df_seq))
            if self.sample_mode == 'random':
                rows = df_seq.sample(count_frame)
            else:
                indx = select_uniform_indices(df_seq, k=count_frame+2)
                rows = df_seq.iloc[indx[1:-1]]
            rows = rows.copy().sort_values(by='relative_path')
        else:
            rows = self.df[self.df.sequence_name == seq_name]


        # Читаем изображения кадров
        frames = []
        for frame_path in rows['relative_path']:
            # чтение и перевод в ргб
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # get_countours(frames[0])

        # аугментируем ролик
        p_aug = 0.1

        if random.random() < p_aug:
            frames = convert_temp_kelvin(frames)
        if random.random() < p_aug:
            frames = consrast_brightness(frames)
        if random.random() < p_aug:
            frames = gaussian_noise(frames, scale=(0, 0.2))
        if random.random() < p_aug:
            frames = motion(frames)
        if random.random() < p_aug:
            frames = grayscale(frames)
        if random.random() < p_aug/2:
            frames = mulHueSatur(frames)
        if random.random() < p_aug:
            frames = resize_into(frames, shifted_scale=0.2)

        if random.random() < p_aug:
            frames = rot(frames, mode='onerand', angle=(-15, 15))

        if rows['class'].iloc[0] == 'bridge_down' and random.random() < p_aug/2:
            frames = fog(frames)

        # frames[len(frames)//2] = grayscale(frames[len(frames)//2], scale=(1, 1))

        # get_countours(frames[0])
        # get_countours(frames[1])

        # print(rows['relative_path'].str[-40:])
        # for img in frames:
        # # image = np.array(frames)
        # # print(image.shape)
        # # image = image.mean(axis=0) - image[0]
        # # image = image.astype(np.uint8)
        # # print(image.shape)
        #     cv2.imshow('', img[..., ::-1])
        #     k = cv2.waitKey(0)
        #     if k == 27:
        #         cv2.destroyAllWindows()
        #         break
        # cv2.destroyAllWindows()

        # Преобразуем список кадров в тензор
        if self.transform:
            frames = [self.transform(fr) for fr in frames]

        frames = torch.stack(frames)

        label = rows['class'].iloc[0]
        # Возвращаем кадры и метку/id класса
        return frames, self.class_name2id[label]


    def __len__(self):
        return len(self.idx2seq)

    def get_sampler(self):

        labels = []
        for seq in self.idx2seq:
            labels.append(self.df[self.df.sequence_name == seq]['class'].iloc[0])
        # for i in tqdm(range(len(ds_train)), desc='create sampler'):
        #     _, label = ds_train[i]
        #     cls_id = torch.argmax(label)
        #     labels.append(int(cls_id))
        class_count = {c: labels.count(c) for c in set(labels)}
        class_weights = [1.0 / class_count[c] for c in labels]

        # Создание сэмплера с взвешенной случайной выборкой
        # print('Создание сэмплера с взвешенной случайной выборкой')
        sampler = WeightedRandomSampler(class_weights, self.__len__(), replacement=True)
