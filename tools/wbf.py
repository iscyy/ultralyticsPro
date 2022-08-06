import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from ensemble_boxes import * # install ensemble_boxes

def xywh2x1y1x2y2(bbox):
    x1 = bbox[0] - bbox[2]/2
    x2 = bbox[0] + bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    y2 = bbox[1] + bbox[3]/2
    return ([x1,y1,x2,y2])

def x1y1x2y22xywh(bbox):
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return ([x,y,w,h])

IMG_PATH = '/Datasets/images/'
TXT_PATH = './runs/val/'

OUT_PATH = './runs/wbf_labels/'


MODEL_NAME = os.listdir(TXT_PATH)
# MODEL_NAME = ['test1','test2']

# ===============================
# Default WBF config (you can change these)
iou_thr = 0.67 #0.67
skip_box_thr = 0.01
# skip_box_thr = 0.0001
sigma = 0.1
# boxes_list, scores_list, labels_list, weights=weights,
# ===============================

image_ids = os.listdir(IMG_PATH)
for image_id in tqdm(image_ids, total=len(image_ids)):
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []
    for name in MODEL_NAME:
        box_list = []
        score_list = []
        label_list = []
        txt_file = TXT_PATH + name + '/labels/' + image_id.replace('jpg', 'txt')
        if os.path.exists(txt_file):
        # if os.path.getsize(txt_file) > 0:
            txt_df = pd.read_csv(txt_file,header=None,sep=' ').values

            for row in txt_df:
                box_list.append(xywh2x1y1x2y2(row[1:5]))
                score_list.append(row[5])
                label_list.append(int(row[0]))
            boxes_list.append(box_list)
            scores_list.append(score_list)
            labels_list.append(label_list)
            weights.append(1.0)
        else:
            continue
            # print(txt_file)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    out_file = open(OUT_PATH + image_id.replace('jpg', 'txt'), 'w')  

    for i,row in enumerate(boxes):
        img = Image.open(IMG_PATH + image_id)
        img_size = img.size
        bbox = x1y1x2y22xywh(row)
        out_file.write(str(int(labels[i]+1)) + ' ' +" ".join(str(x) for x in bbox) + " " + str(round(scores[i],6)) + '\n')
    out_file.close()