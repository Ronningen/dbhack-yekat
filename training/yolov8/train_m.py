import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


if __name__ == '__main__':
    # weight = 'yolov8m.pt'
    weight = './runs/detect/attempt_22/weights/best.pt'
    model = YOLO(weight)

    model.train(
        data='datasetv2.yaml',
        imgsz=1280,
        batch=4,
        cache=False,
        patience=5,
        name="attempt_3",
        degrees=15,
        mixup=0.5,
        epochs=10
    )

