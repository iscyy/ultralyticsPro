import sys
import argparse
import os

from ultralytics import YOLO


def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    model = YOLO(yaml)

    model.info()

    results = model.train(data='mnist160', 
                        epochs=100, 
                        imgsz=640, 
                        workers=2, 
                        batch=8,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\11\yolo11-cls.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)