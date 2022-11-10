## YOLOAir :  Make the improvement of the YOLO model faster, more convenient

English | [简体中文](./README.md)

The YOLOAir algorithm library is a combination toolbox of a series of YOLO detection algorithms based on PyTorch. Used to **combine different modules to build different networks**.  

<div align='center'>
    <img width="850" src="https://img-blog.csdnimg.cn/11d5f1f403f74cff9c2f70a0c3b92de4.png"></a>
</div>

✨Built-in model network structures such as YOLOv5, YOLOv7, YOLOX, YOLOR, Transformer, Scaled_YOLOv4, YOLOv3, YOLOv4, YOLO-Facev2, TPH-YOLOv5, YOLOv5Lite, PicoDet (continuously updated🚀)...

**✨Module componentization**: Help users customize and quickly combine Backbone, Neck, Head, diversify network models, help scientific research improve detection algorithms, model improvements, and network permutation and combination 🏆. Build powerful network models.

**✨Unified model code framework, unified application method, unified parameter adjustment, unified improvement, easy module combination, and more powerful network model building**.
  

```

██╗   ██╗ ██████╗ ██╗      ██████╗      █████╗     ██╗    ██████╗ 
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ██╔══██╗    ██║    ██╔══██╗
 ╚████╔╝ ██║   ██║██║     ██║   ██║    ███████║    ██║    ██████╔╝
  ╚██╔╝  ██║   ██║██║     ██║   ██║    ██╔══██║    ██║    ██╔══██╗
   ██║   ╚██████╔╝███████╗╚██████╔╝    ██║  ██║    ██║    ██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝
```


✨Based on the YOLOv5 code framework, and synchronously adapt to the **stable YOLOv5_v6.1 update**, synchronize the v6.1 deployment ecosystem. Before using this project, you can learn about the YOLOv5 library.

______________________________________________________________________

[Features🚀](#main-features) • [Usage🍉](#usage) • [Documentation📒](https://github.com/iscyy/yoloair) • [issues🌟](https://github.com/iscyy/yoloair/issues/new) • [News💪](#news-)

![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

#### support
![](https://img.shields.io/badge/Support-YOLOv5-red) ![](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![](https://img.shields.io/badge/Support-YOLOX-yellow) ![](https://img.shields.io/badge/Support-YOLOv4-green) ![](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4)
![](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![](https://img.shields.io/badge/Support-Attention-green)

Project address: https://github.com/iscyy/yoloair


______________________________________________________________________

### Main features🚀

🚀 Support more YOLO series algorithm model improvements (continuous update...)

The YOLOAir algorithm library summarizes a variety of mainstream YOLO series detection models, and a set of codes integrates multiple models:
- Built-in integrated YOLOv5 model network structure, YOLOv7 model network structure, YOLOR model network structure, YOLOX model network structure, Scaled_YOLOv4 model network structure, YOLOv4 model network structure, YOLOv3 model network structure, YOLO-FaceV2 model network structure, TPH-YOLOv5 model network structure The structure, YOLOv5-Lite model network structure, PicoDet model network structure, etc. are continuously updated...

|||
:-------------------------:|:-------------------------:
<img src='docs/image/test.jpg' width='500px'>  |  <img src='docs/image/zebra.jpg' width='500px'>

The above multiple detection algorithms use a unified model code framework, which is integrated in the YOLOAir library, with a unified task form and a unified application method. 
🌟It is convenient for researchers to improve the algorithm model of the paper, compare models, and realize the diversification of network combinations. including lightweight models and models with higher accuracy, which can be reasonably selected according to the scene, and achieve a balance between accuracy and speed. 
🌟the library supports decoupling of different structures and module components to make modules componentized. By combining different module components, users can customize and build different detection models according to different data sets or different business scenarios.

🔥🔥🔥 Heavy! ! ! As a supplement to the open source project of the attention mechanism, a 6300+ 🌟Star attention mechanism algorithm code base is strongly recommended 👉[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)，The summary is very comprehensive, including various Attention, Self-Attention and other codes. The code is concise and easy to read, and one line of code implements the Attention mechanism. Welcome everyone to play!

______________________________________________________________________

🚀Support loading the official pre-training weights of YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOR and other networks for transfer learning

🚀 Support more Backbone

- `CSPDarkNet series`  
`ResNet series`  
`RepBlock series`  
`ShuffleNet series`  
`Ghost series`  
`MobileNet series`  
`ConvNext series`  
`RepLKNet series`  
`RegNet series`  
`EfficientNet series`  
`Transformer series`  
`CNN and Transformer, etc.`  
Continuously updated 🎈  

🚀 Support more Necks

- The neck includes mainstream structures such as FPN, PANet, and BiFPN.
 Continuously updated 🎈

🚀Support more detection heads
-  YOLOv4、YOLOv5 Head、
-  YOLOR Implicit Head、
-  YOLOX Decoupled Head、DetectX Head
-  Adaptively Spatial Feature Fusion Head(ASFF)、
-  YOLOv7 IAuxDetect Head, IDetect Head；

🚀Support for more plug-and-play attention mechanisms
- Plug-and-play use of attention mechanisms anywhere in the network  
Self Attention  
Contextual Transformer  
Bottleneck Transformer  
S2-MLP Attention  
SK Attention  
CBAM Attention  
SE Attention  
Coordinate attention  
NAM Attention  
GAM attention  
ECA Attention  
Shuffle Attention  
DANet Attention and other mainstream  attention mechanisms  

🚀Support for more IoU loss functions
- Loss functions such as CIoU, DIoU, GIoU, EIoU, SIoU, alpha IOU, etc.  

🚀 More spatial pyramid pooling structures  
- SPP / SPPF / ASPP / RFB / SPPCSPC, etc.  

🚀Support more NMS
- NMS、Merge-NMS、DIoU-NMS、Soft-NMS、CIoU_NMS、DIoU_NMS、GIoU_NMS、EIoU_NMS、SIoU_NMS、Soft-SIoUNMS、Soft-CIoUNMS、Soft-DIoUNMS、Soft-EIoUNMS、Soft-GIoUNMS, etc.  

🚀Support for more data augmentation
- Mosaic、Copy paste、Random affine(Rotation, Scale, Translation and Shear)、MixUp、Augment HSV(Hue, Saturation, Value、Random horizontal flip

🚀Support more Loss
- ComputeLoss、ComputeNWDLoss、ComputeLoss(X)、ComputeLoss(v6)、ComputeLossAuxOTA(v7)、ComputeLossOTA(v7), etc.

🚀Supports Anchor-base and Anchor-Free

🚀Supports Weighted Box Fusion (WBF)

🚀 Built-in multiple network model modular components
- Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost etc.
Detailed code search in **./models/common.py file**🔍👉[Corresponding module link](https://github.com/iscyy/yoloair/blob/main/models/common.py)

- 👉[Network Model Structure Diagram](https://github.com/iscyy/yoloair/blob/main/docs/document/model_.md) 

The above component modules use a unified model code framework, unified task form, and unified application method. **Module componentization**🚀 can help users customize and quickly combine Backbone, Neck, and Head, diversify network models, and help scientific research improve detection algorithms. Build stronger network models.

### Built-in network model configuration support ✨

🚀Including yaml configuration of YOLOv3, YOLOv4, Scaled_YOLOv4, YOLOv5, YOLOv7, YOLOX, YOLOR, Transformer, YOLO-FaceV2, PicoDet, YOLOv5-Lite, TPH-YOLOv5 ** other algorithm models such as improved network structure** File Summary (Recommended 🌟🌟🌟🌟🌟)

### News! <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="ingishvcn" width="20">

Support SPD-Conv✅  
Support HorNet Network ✅  
Support ConvNext module ✅  
🔥Add the difference of YOLOv3, YOLOv4, YOLOv5, YOLOv7, Scaled-YOLOv4 network [pre-training weight link 🔗](https://github.com/iscyy/yoloair/releases)  
Updating...
______________________________________________________________________
## Usage🍉

**About the code.** Follow the design principle of [YOLOv5](https://github.com/ultralytics/yolov5).  
The original version was created based on YOLOv5(v6.1)

### Install

Clone repo and install requirements.txt in a **Python>=3.7.0** environment, including **PyTorch>=1.7**. 
```bash
$ git clone https://github.com/iscyy/yoloair.git
$ cd yoloair
$ pip install -r requirements.txt
```

### Training

```bash
$ python train.py --data coco128.yaml --cfg configs/yolov5/yolov5s.yaml
```

### Inference

`detect.py` runs inference on a variety of sources,  saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
```

### fusion
If you use different models to infer on the dataset, you can use the wbf.py file to integrate the results via weighted box fusion.
You just need to set img path and txt path in wbf.py file.
```bash
$ python wbf.py
```
______________________________________________________________________
### Performance
| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| YOLOv5n                   | 640                   | 28.0                    | 45.7               | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| YOLOv5s                   | 640                   | 37.4                    | 56.8               | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| YOLOv5m                   | 640                   | 45.4                    | 64.1               | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| YOLOv5l                   | 640                   | 49.0                    | 67.3               | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| YOLOv5x                   | 640                   | 50.7                    | 68.9               | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                      |                       |                         |                    |                              |                               |                                |                    |                        |
| YOLOv5n6                 | 1280                  | 36.0                    | 54.4               | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| YOLOv5s6                 | 1280                  | 44.8                    | 63.7               | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| YOLOv5m6                 | 1280                  | 51.3                    | 69.3               | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| YOLOv5l6                 | 1280                  | 53.7                    | 71.3               | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| YOLOv5x6<br>+ TTA | 1280<br>1536          | 55.0<br>**55.8**        | 72.7<br>**72.7**   | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

<details><summary> <b>Expand</b> </summary>

* The original version was created based on YOLOv5

</details>

______________________________________________________________________

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | deconv kernel size<br><sup> | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| YOLOv5s                   | 640                   | 33.7                    | 52.9               | -                       | **5.6**                       | **2.2**                        | **7.23**            | **16.5**                |
| YOLOv5s-deconv-exp1                   | 640                   | 33.4                    | 52.5               | 2                       | **5.6**                       | 2.4                        | 7.55            | 18.2                |
| YOLOv5s-deconv-exp2                   | 640                   | **34.7**                    | **54.2**               | 4                           | 5.8                           | 2.5                            | 8.54                | 23.2                   |
<details><summary> <b>Expand</b> </summary>

* The training process depends on 4xV100 GPU
```
# train
python -m torch.distributed.run --nproc_per_node 4 train.py --device 0,1,2,3 --data data/coco.yaml --hyp data/hyps/hyp.scratch-low.yaml  --cfg path/to/model.yaml --batch 512 --epochs 300 --weights ''
# val
python val.py --verbose --data data/coco.yaml --conf 0.001 --iou 0.65 --batch 1 --weights path/to/model.pt
```

* There is a gap between the mAP of YOLOv5s and the official one, here is just for comparison
</details>

______________________________________________________________________

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------------------|--------------------|------------------------|
| YOLOv5s                   | 640                   | 37.4                       | 56.6                        | **7.226**            | **16.5**                |
| YOLOv5s-deconv             | 640                   | **37.8**                       | **57.1**                        | 7.232            | **16.5**                |

<details><summary> <b>Expand</b> </summary>

* tested the 4x4 depthwise-separable deconv by setting the groups as input_channel
* their params number and FLOPS are nearly the same while the new model's mAP is about 0.4 higher than the origin.
</details>

______________________________________________________________________

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------------------|--------------------|------------------------|
| YOLOv5s                   | 640                   | 37.2                       | 56.0                        | **7.2**            | **16.5**                |
| YOLOv5s-C3GC-backbone             | 640                   | **37.7**                       | **57.3**                        | 7.5            | 16.8                |

<details><summary> <b>Expand</b> </summary>

* The original version was created based on YOLOv5-6.0
</details>

______________________________________________________________________

### YOLO network model specific improvement method tutorial and principle reference

More module improvement explanation tutorials are continuously updated...

### YOLOv5 official tutorial ✨
Sync with YOLOv5 framework

- [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 RECOMMENDED
- [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️ RECOMMENDED
- [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) 🌟 NEW
- [TFLite, ONNX, CoreML, TensorRT Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
- [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
- [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
- [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)
- [Architecture Summary](https://github.com/ultralytics/yolov5/issues/6998) 🌟 NEW
- [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)
- [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)  🌟 NEW
- [ClearML Logging](https://github.com/ultralytics/yolov5/tree/master/utils/loggers/clearml) 🌟 NEW
- [Deci Platform](https://github.com/ultralytics/yolov5/wiki/Deci-Platform) 🌟 NEW

</details>

### Future enhancement ✨
In the future, we will continue to build and improve the YOLOAir ecosystem
Improve and integrate more YOLO series models, continue to combine different modules to build more different network models
Horizontal expansion and introduction of associative techniques, such as semi-supervised learning, etc.
Follow up: YOLO-mask & YOLO-pose

______________________________________________________________________

## Statement
<details><summary> <b>Expand</b> </summary>

* The content of this site is only for sharing notes. If some content is infringing, please sending email.

* If you have any question, please discuss with me by sending email.
</details>

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)  
[https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)  
[https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)  
[https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)   
[https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)  
[https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)  
[https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)   
[https://github.com/xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)  
[https://gitee.com/SearchSource/yolov5_yolox](https://gitee.com/SearchSource/yolov5_yolox)  
[https://github.com/Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2)  
[https://github.com/positive666/yolov5_research/](https://github.com/positive666/yolov5_research)  
[https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)  
[https://github.com/Gumpest/YOLOv5-Multibackbone-Compression](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)  
[https://github.com/cv516Buaa/tph-yolov5](https://github.com/cv516Buaa/tph-yolov5)
Paper:[https://arxiv.org/abs/2208.02019](https://arxiv.org/abs/2208.02019)  

</details>
