from ultralytics import YOLO
import os
from ultralytics import RTDETR
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
if __name__ == '__main__':
    
    model = YOLO("yolov8s.yaml")  

    results = model.train(data='dates_cizhuan_6714.yaml', epochs=300, batch=8, name='',mosaic=1,patience=300,optimizer='SGD')


