import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO('yolov10n.pt')

    model.predict(r'ultralytics\assets\zidane.jpg', save=True, imgsz=640, conf=0.5)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'yolov10n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)