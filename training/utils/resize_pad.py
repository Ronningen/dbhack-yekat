import cv2
import numpy as np

def create_resized_image(image, target_size, background_color):
    """
    Ресайзит изображение к целевому размеру с соотношением сторон и добавлением паддингов.

    :param image: изображение/массив H W C
    :param target_size: W H целевой
    :param background_color: цвет подложки
    :return:
    """
    # Определение размеров исходного изображения
    image_height, image_width = image.shape[:2]

    # Вычисление соотношения сторон и определение новых размеров изображения
    image_ratio = image_width / image_height
    target_width, target_height = target_size
    target_ratio = target_width / target_height

    if target_ratio > image_ratio:
        new_width = int(target_height * image_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / image_ratio)

    # Ресайз исходного изображения
    resized_image = cv2.resize(image, (new_width, new_height))

    # Создание подложки с заданным цветом пикселя
    background = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    # Вычисление координат для вставки ресайзнутого изображения в центр подложки
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2
    end_x = start_x + new_width
    end_y = start_y + new_height

    # Вставка ресайзнутого изображения в центр подложки
    background[start_y:end_y, start_x:end_x] = resized_image

    return background

if __name__ == "__main__":
    # пример использования
    img = cv2.imread(r"C:\Users\annza\Downloads\1.jpg")
    target_size = (224, 224)  # Целевой размер для ресайза
    background_color = (0, 0, 255)  # Цвет пикселя для подложки (BGR)

    resized_image = create_resized_image(img, target_size, background_color)

    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()