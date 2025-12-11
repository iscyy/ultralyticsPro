import sys
import argparse
import os


from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    model = YOLO(yaml)

    model.info()
    
    # 旋转检测训练
    results = model.train(
        data='dota8.yaml', 
        epochs=2, 
        imgsz=640, 
        workers=2, 
        batch=2
        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\11\yolo11-obb.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default=r'', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)