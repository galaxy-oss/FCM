from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(
"/home/zyh/code/xy/ultralytics-main-new/runs/detect/train7-baseline/weights/hongwai_best.engine")  # build a new model from YAML
    # Validate the model
    model.val(val=True, data='dates_cizhuan_6714.yaml', split='val', batch=1)
   

