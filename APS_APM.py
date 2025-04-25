from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    pred_json ="/home/zyh/code/xy/ultralytics-main/runs/detect/val2/predictions.json"
    anno_json ="/home/zyh/data/xy/my_dates_6417/valid/annotations_VisDrone_test_ok.json"


    cocoGt = COCO(anno_json)
    cocoDt = cocoGt.loadRes(pred_json)


    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')


    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    print(cocoEval.stats)