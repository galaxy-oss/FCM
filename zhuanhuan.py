from ultralytics import YOLO


model = YOLO("/home/zyh/code/xy/ultralytics-main-new/runs/detect/train7/weights/hongwai_best.pt")
model.export(format="engine",device=0)  