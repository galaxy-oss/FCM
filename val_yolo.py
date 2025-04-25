from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(
    "/home/zyh/code/xy/ultralytics-main/runs/detect/S-baseline/weights/S-baseline_best.pt")  # build a new model from YAML
    # Validate the model
    model.val(val=True, data='dates_cizhuan_6714.yaml', split='val', batch=1, save_json=True)