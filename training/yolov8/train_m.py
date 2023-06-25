import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


if __name__ == '__main__':
    weight = 'yolov8m.pt'
    # weight = './runs/detect/attempt_3/weights/best.pt'
    model = YOLO(weight)

    model.train(
        data='netrics2.yaml',
        imgsz=1280,
        batch=4,
        cache=False,
        patience=5,
        name="net_val_1",
        degrees=15,
        mixup=0.5,
        shear=0.1,
        perspective=0.0005,
        epochs=10
    )

