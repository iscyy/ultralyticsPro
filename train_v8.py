import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    results = model.train(data='coco128.yaml', 
                        epochs=2, 
                        imgsz=320, 
                        workers=0, 
                        batch=1,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\v8\yolov8.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)