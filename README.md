## YOLOAir: Make the improvement of the YOLO model faster, more convenient

简体中文 | [English](./README_EN.md)

YOLOAir算法库 是一个基于 PyTorch 的一系列 YOLO 检测算法组合工具箱。用来**组合不同模块构建不同网络**。  

<div align='center'>
    <img src='docs/image/logo1.png' width='500px'>
</div>

内置YOLOv5、YOLOv7、YOLOX、YOLOR、Transformer、Scaled_YOLOv4、YOLOv3、YOLOv4、YOLO-Facev2、TPH-YOLOv5、YOLOv5Lite、PicoDet等模型网络结构(持续更新中🚀)...

**模块组件化**：帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法、模型改进，网络排列组合🏆。构建强大的网络模型。

**统一模型代码框架、统一应用方式、统一调参、统一改进、易于模块组合、构建更强大的网络模型**。
  

```

██╗   ██╗ ██████╗ ██╗      ██████╗      █████╗     ██╗    ██████╗ 
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ██╔══██╗    ██║    ██╔══██╗
 ╚████╔╝ ██║   ██║██║     ██║   ██║    ███████║    ██║    ██████╔╝
  ╚██╔╝  ██║   ██║██║     ██║   ██║    ██╔══██║    ██║    ██╔══██╗
   ██║   ╚██████╔╝███████╗╚██████╔╝    ██║  ██║    ██║    ██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝
```


基于 YOLOv5 代码框架，并同步适配 **稳定的YOLOv5_v6.1更新**, 同步v6.1部署生态。使用这个项目之前, 您可以先了解YOLOv5库。  

[特性🚀](#Mainfeatures) • [使用🍉](#Usage) • [文档📒](https://github.com/iscyy/yoloair) • [报告问题🌟](https://github.com/iscyy/yoloair/issues/new)

![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

#### 支持
![](https://img.shields.io/badge/Support-YOLOv5-red) ![](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![](https://img.shields.io/badge/Support-YOLOX-yellow) ![](https://img.shields.io/badge/Support-YOLOv4-green) ![](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4)
![](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![](https://img.shields.io/badge/Support-Attention-green)

项目地址: https://github.com/iscyy/yoloair

部分改进说明演示: [芒果汁没有芒果](https://blog.csdn.net/qq_38668236?type=blog)

______________________________________________________________________

### 主要特性🚀

🚀支持更多的YOLO系列算法模型改进(持续更新...)

YOLOAir 算法库汇总了多种主流YOLO系列检测模型，一套代码集成多种模型: 
- 内置集成 YOLOv5 模型网络结构、YOLOv7 模型网络结构、 YOLOR 模型网络结构、YOLOX 模型网络结构、Scaled_YOLOv4 模型网络结构、YOLOv4 模型网络结构、YOLOv3 模型网络结构、YOLO-FaceV2模型网络结构、TPH-YOLOv5模型网络结构、YOLOv5-Lite模型网络结构、PicoDet模型网络结构等持续更新中...

|||
:-------------------------:|:-------------------------:
<img src='docs/image/test.jpg' width='500px'>  |  <img src='docs/image/zebra.jpg' width='500px'>

- 以上多种检测算法使用统一模型代码框架，**集成在 YOLOAir 库中，统一任务形式、统一应用方式**。🌟便于科研者用于论文算法模型改进，模型对比，实现网络组合多样化。🌟工程算法部署落地更便捷，包含轻量化模型和精度更高的模型，根据场景合理选择，在精度和速度俩个方面取得平衡。同时该库支持解耦不同的结构和模块组件，让模块组件化，通过组合不同的模块组件，用户可以根据不同数据集或不同业务场景自行定制化构建不同检测模型。

🔥🔥🔥 重磅！！！作为注意力机制的开源项目补充，强烈推荐一个6300+🌟Star的注意力机制算法代码库👉[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)，里面汇总整理很全面，包含各种Attention、Self-Attention等代码，代码简洁易读，一行代码实现Attention机制。欢迎大家来玩呀！

🚀支持加载YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOR等网络的官方预训练权重进行迁移学习

🚀支持更多Backbone

-  CSPDarkNet系列  
-  RepBlock系列  
-  HorNet系列  
-  ResNet系列  
-  RegNet 系列  
-  ShuffleNet系列  
-  Ghost系列  
-  MobileNet系列  
-  EfficientNet系列  
-  ConvNext系列  
-  RepLKNet系列  
-  自注意力Transformer系列  
-  CNN和Transformer结合系列  
持续更新中🎈

🚀支持更多Neck
- FPN  
- PANet  
- BiFPN等主流结构。  
 持续更新中🎈

🚀支持更多检测头Head  
-  YOLOv4、YOLOv5 Head检测头、
-  YOLOR 隐式学习Head检测头、
-  YOLOX的解耦合检测头Decoupled Head、DetectX Head
-  自适应空间特征融合 检测头ASFF Head、
-  YOLOv7检测头IAuxDetect Head, IDetect Head等；
-  其他不同检测头

🚀支持更多即插即用的注意力机制Attention
- 在网络任何部分即插即用式使用注意力机制
- Self Attention  
- Contextual Transformer  
- Bottleneck Transformer  
- S2-MLP Attention  
- SK Attention  
- CBAM Attention  
- SE Attention  
- Coordinate attention  
- NAM Attention  
- GAM Attention  
- ECA Attention  
- Shuffle Attention  
- CrissCrossAttention  
- Coordinate attention  
- SOCAttention  
- SimAM Attention 
持续更新中🎈  

🚀更多空间金字塔池化结构  
- SPP
- SPPF
- ASPP
- RFB
- SPPCSPC  
持续更新中🎈    

🚀支持更多Loss   
- ComputeLoss(v5)  
- ComputeLoss(X)  
- ComputeLossAuxOTA(v7)  
- ComputeLossOTA(v7)  
- ComputeNWDLoss  
- 其他Loss

🚀支持Anchor-base和Anchor-Free  

🚀支持多种正负样本分配  

🚀支持加权框融合(WBF)  

🚀 内置多种网络模型模块化组件  
Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3HB, C3RFEM, MultiSEAM, SEAM, C3STR, SPPCSPC, RepConv, BoT3, Air, CA, CBAM, Involution, Stem, ResCSPC, ResCSPB, ResXCSPB, ResXCSPC, BottleneckCSPB, BottleneckCSPC, ASPP, BasicRFB, SPPCSPC_group, HorBlock, CNeB,C3GC ,C3C2, nn.ConvTranspose2d, DWConvblock, RepVGGBlock, CoT3, ConvNextBlock, SPPCSP, BottleneckCSP2, DownC, BottleneckCSPF, RepVGGBlock, ReOrg, DWT, MobileOne,HorNet...等详细代码 **./models/common.py文件** 内  

🚀支持更多IoU损失函数  
- CIoU  
- DIoU  
- GIoU  
- EIoU  
- SIoU  
- alpha IOU  
持续更新中🎈    

🚀支持更多NMS  
- NMS  
- Merge-NMS  
- Soft-NMS  
- CIoU_NMS  
- DIoU_NMS  
- GIoU_NMS  
- EIoU_NMS  
- SIoU_NMS  
- Soft-SIoUNMS、Soft-CIoUNMS、Soft-DIoUNMS、Soft-EIoUNMS、Soft-GIoUNMS等;    
持续更新中🎈    

🚀支持更多数据增强  
- Mosaic、Copy paste、Random affine(Rotation, Scale, Translation and Shear)、MixUp、Augment HSV(Hue, Saturation, Value、Random horizontal flip

网络模型结构图: [模型🔗](https://github.com/iscyy/yoloair/blob/main/docs/document/model_.md) 

以上组件模块使用统一模型代码框架、统一任务形式、统一应用方式，**模块组件化**🚀 可以帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法，构建更强大的网络模型。

### 内置网络模型配置支持✨

🚀包括基于 YOLOv5、YOLOv7、YOLOX、YOLOR、YOLOv3、YOLOv4、Scaled_YOLOv4、Transformer、YOLO-FaceV2、PicoDet、YOLOv5-Lite、TPH-YOLOv5 等**其他多种改进网络结构等算法模型**的模型配置文件
______________________________________________________________________

### 更新 🌟

支持 SPD-Conv  
支持 HorNet 网络  
支持 ConvNext 模块  
支持 CNeBlock  
支持 C3HBLock  
支持 C3GCBLock  
支持 C3C2BLock
持续更新中🎈 

______________________________________________________________________

### 技术交流 <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20">

|FightingCV公众号|YOLOAir目标检测交流群( 答案:  yoloair )|
:-------------------------:|:-------------------------:
<img src='https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b6f5057da9a8410fa22dcc7566548193~tplv-k3u1fbpfcp-watermark.image?' width='200px'>  |  <img src='https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1882e8cf3f804918a043d70de1b70e31~tplv-k3u1fbpfcp-watermark.image' width='200px'> 

- FightingCV每天分享前沿论文动态(公众号回复加群, 添加小助手, 加入微信交流群)  

- YOLOAir目标检测交流群
______________________________________________________________________

## 使用🍉

**About the code.** Follow the design principle of [YOLOv5](https://github.com/ultralytics/yolov5).  
The original version was created based on YOLOv5(v6.1)

### 安装

在**Python>=3.7.0** 的环境中克隆版本仓并安装 requirements.txt，包括**PyTorch>=1.7**。

```bash
$ git clone https://github.com/iscyy/yoloair.git  # 克隆
$ cd YOLOAir
$ pip install -r requirements.txt  # 安装
```

### 训练

```bash
$ python train.py --data coco128.yaml --cfg configs/yolov5/yolov5s.yaml #默认为yolo
```

### 推理

`detect.py` 在各种数据源上运行推理, 并将检测结果保存到 `runs/detect` 目录。

```bash
$ python detect.py --source 0  # 网络摄像头
                          img.jpg  # 图像
                          vid.mp4  # 视频
                          path/  # 文件夹
                          path/*.jpg  # glob
```

### 融合
如果您使用不同模型来推理数据集，则可以使用 wbf.py文件 通过加权框融合来集成结果。
您只需要在 wbf.py文件 中设置 img 路径和 txt 路径。
```bash
$ python wbf.py
```
______________________________________________________________________

### Performance
| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) | Weights
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|------------------------|
| YOLOv5n                   | 640                   | 28.0                    | 45.7               | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                | [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt)
| YOLOv5s                   | 640                   | 37.4                    | 56.8               | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   | [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt)
| YOLOv5m                   | 640                   | 45.4                    | 64.1               | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   | [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt)
| YOLOv5l                   | 640                   | 49.0                    | 67.3               | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  | [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt)
| YOLOv5x                   | 640                   | 50.7                    | 68.9               | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  | [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt)
|                                                                                                      |                       |                         |                    |                              |                               |                                |                    |                        |
| YOLOv5n6                 | 1280                  | 36.0                    | 54.4               | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |[YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n6.pt)
| YOLOv5s6                 | 1280                  | 44.8                    | 63.7               | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |[YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt)
| YOLOv5m6                 | 1280                  | 51.3                    | 69.3               | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |[YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m6.pt)
| YOLOv5l6                 | 1280                  | 53.7                    | 71.3               | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |[YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l6.pt)
| YOLOv5x6<br>+ TTA | 1280<br>1536          | 55.0<br>**55.8**        | 72.7<br>**72.7**   | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |[YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt)

<details><summary> <b>Expand</b> </summary>

* The original version was created based on YOLOv5(6.1)

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


### YOLO网络模型具体改进方式教程及原理参考

- 11.[改进YOLOv5系列：11.ConvNeXt结合YOLO | CVPR2022 多种搭配，即插即用 | Backbone主干CNN模型](https://blog.csdn.net/qq_38668236/article/details/126454548)

- 10.[改进YOLOv5系列：10.最新ECCV2022 | HorNet即插即用、Backbone主干、递归门控卷积的高效高阶空间交互](https://blog.csdn.net/qq_38668236/article/details/126410711)

- 9.[改进YOLOv5系列：9.BoTNet Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126333061)

- 8.[改进YOLOv5系列：8.增加ACmix结构的修改,自注意力和卷积集成](https://blog.csdn.net/qq_38668236/article/details/126302599)

- 7.[改进YOLOv5系列：7.修改DIoU-NMS,SIoU-NMS,EIoU-NMS,CIoU-NMS,GIoU-NMS](https://blog.csdn.net/qq_38668236/article/details/126243834)

- 6.[改进YOLOv5系列：6.修改Soft-NMS,Soft-CIoUNMS,Soft-SIoUNMS](https://blog.csdn.net/qq_38668236/article/details/126245080)

- 5.[改进YOLOv5系列：5.CotNet Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126226726)

- 4.[改进YOLOv5系列：4.YOLOv5_最新MobileOne结构换Backbone修改](https://blog.csdn.net/qq_38668236/article/details/126157859)

- 3.[改进YOLOv5系列：3.Swin Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126122888?spm=1001.2014.3001.5502)

- 2.[改进YOLOv5系列：2.PicoDet结构的修改](https://blog.csdn.net/qq_38668236/article/details/126087343?spm=1001.2014.3001.5502)

- 1.[改进YOLOv5系列：1.多种注意力机制结合YOLO应用](https://blog.csdn.net/qq_38668236/article/details/126086716)


更多模块详细解释教程持续更新中...

______________________________________________________________________

### YOLOv5官方教程✨
与YOLOv5框架同步

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 推荐
- [获得最佳训练效果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️ 推荐
- [使用 Weights & Biases 记录实验](https://github.com/ultralytics/yolov5/issues/1289)  🌟 新
- [Roboflow：数据集、标签和主动学习](https://github.com/ultralytics/yolov5/issues/4975)  🌟 新
- [多GPU训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)  ⭐ 新
- [TFLite, ONNX, CoreML, TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [测试时数据增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型集成](https://github.com/ultralytics/yolov5/issues/318)
- [模型剪枝/稀疏性](https://github.com/ultralytics/yolov5/issues/304)
- [超参数进化](https://github.com/ultralytics/yolov5/issues/607)
- [带有冻结层的迁移学习](https://github.com/ultralytics/yolov5/issues/1314) ⭐ 新
- [架构概要](https://github.com/ultralytics/yolov5/issues/6998) ⭐ 新

</details>

### 未来增强✨
后续会持续建设和完善 YOLOAir 生态  
完善集成更多 YOLO 系列模型，持续结合不同模块，构建更多不同网络模型  
横向拓展和引入关联技术等等  
跟进：YOLO-mask & YOLO-pose  

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
