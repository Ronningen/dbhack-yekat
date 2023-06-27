import cv2
from tqdm import tqdm

def accelerate_video(video_path, start_frame=0, end_frame=None, step=10):
    """
    Ускоряет видео - создает видео из кадров исходного с шагом step

    :param video_path: путь к видео файлу
    :param start_frame: начало
    :param end_frame: конец
    :param step: шаг
    """
    # Открываем видеофайл для чтения
    video = cv2.VideoCapture(video_path)

    # Получаем информацию о видео
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определяем начальный и конечный кадры
    if end_frame is None:
        end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        end_frame = min(end_frame, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Создаем новое имя файла для сохранения ускоренного видео
    output_path = video_path.rsplit('.', 1)[0] + f"_acc_{start_frame}_{end_frame}_{step}.mp4"

    # Создаем объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Формат кодека для сохранения видео (здесь используется MP4)
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Устанавливаем текущий кадр на начальный
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Обрабатываем кадры с заданным шагом и записываем их в новое видео
    frame_counter = start_frame
    pbar = tqdm(total=(end_frame - start_frame) // step + 1)
    while frame_counter <= end_frame:
        ret, frame = video.read()

        if not ret:
            break

        output_video.write(frame)

        # Пропускаем заданное количество кадров
        frame_counter += step
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        pbar.update(1)

    pbar.close()

    # Закрываем видеофайлы
    video.release()
    output_video.release()

    print("Ускоренное видео сохранено в", output_path)



if __name__ == "__main__":
    video_path = r"C:\workspace\hakaton\hak2023_Digital_breakthrough_ekat\9Rc5BSwckWI_2106.mp4"

    start_frame = 0
    end_frame = None
    step = 30*10  # каждые 10 sec

    accelerate_video(video_path, start_frame, end_frame, step)
