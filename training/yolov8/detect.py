import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


if __name__ == '__main__':
    weight = "C:\workspace\hakaton\hak2023_Digital_breakthrough_ekat\dbhack-yekat\training\yolov8\runs\detect\net_12\weights\best.pt"
    model = YOLO(weight)

    model.track(
        source=r"G:\.shortcut-targets-by-id\12eFEjs53W4IJWpGf4InEf2cT-SbM3y_s\netrics\dataset_79200-79500.mp4",
        imgsz=1280,
        conf=0.1,
    )
