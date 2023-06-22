import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO


if __name__ == '__main__':
    # weight = 'yolov8m.pt'
    weight = './runs/detect/attempt_2/weights/last.pt'
    model = YOLO(weight)

    model.train(
        data='datasetv2.yaml',
        # imgsz=1280,
        # batch=5,
        # cache=False,
        # patience=5,
        # name="attempt_2",
        epochs=10,
        resume=True
    )

