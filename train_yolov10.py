import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg 
    model = YOLO(yaml) # 直接加载yaml文件训练
    # model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(r'runs\detect\train5\weights\last.pt') # 加载yaml配置文件的同时，加载权重进行训练

    model.info()

    results = model.train(data='coco128.yaml',  # 训练参数均可以重新设置
                        epochs=10, 
                        imgsz=640, 
                        workers=2, 
                        batch=2,
                        device='0',
                        # ...在这里添加需要修改的参数
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics\cfg\models\v10\yolov10s.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov10n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)