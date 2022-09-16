## YOLO Air : Makes improvements easy again

<div align="center">
<p>YOLOAir 算法库🚀 
是一个基于 PyTorch 的一系列 YOLO 检测算法组合工具箱。</p>
<p>统一模型代码框架、统一应用、统一改进、易于模块组合、构建更强大的网络模型。</p>
  <p>
    <a align="center" href="https://github.com/iscyy/yoloair" target="_blank">
      <img width="850" src="docs/image/yoloair.png"></a>
    <br><br>
  </p>

  简体中文 | [English](./README_EN.md)

  ![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

  #### **支持**

![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv5-red) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv6-blueviolet) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOX-yellow) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOE-007d65) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv4-green) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-TOOD-6a6da9) 
![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLO-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOv2-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOEPlus-d5c59f) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-MLP-ff69b4) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Attention-green)

[特性🚀](#主要特性) • [使用🍉](#使用) • [文档📒](https://github.com/iscyy/yoloair) • [报告问题🌟](https://github.com/iscyy/yoloair/issues/new) • [更新💪](#-to-do) • [讨论✌️](https://github.com/iscyy/yoloair/discussions)

</div>

**模型多样化**：基于不同网络模块构建不同检测网络模型。

**模块组件化**：帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法、模型改进，网络排列组合🏆。构建强大的网络模型。

**统一模型代码框架、统一应用方式、统一调参、统一改进、易于模块组合、构建更强大的网络模型**。


内置YOLOv5、YOLOv7、YOLOv6、YOLOX、YOLOR、Transformer、PP-YOLO、PP-YOLOv2、PP-YOLOE、PP-YOLOEPlus、Scaled_YOLOv4、YOLOv3、YOLOv4、YOLO-Facev2、TPH-YOLO、YOLOv5Lite、SPD-YOLO、SlimNeck-YOLO、PicoDet等模型网络结构(持续更新中🚀)...

```
██╗   ██╗ ██████╗ ██╗      ██████╗      █████╗     ██╗    ██████╗ 
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ██╔══██╗    ██║    ██╔══██╗
 ╚████╔╝ ██║   ██║██║     ██║   ██║    ███████║    ██║    ██████╔╝
  ╚██╔╝  ██║   ██║██║     ██║   ██║    ██╔══██║    ██║    ██╔══██╗
   ██║   ╚██████╔╝███████╗╚██████╔╝    ██║  ██║    ██║    ██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝
```

基于 YOLOv5 代码框架，并同步适配 **稳定的YOLOv5_v6.1更新**, 同步v6.1部署生态。使用这个项目之前, 您可以先了解YOLOv5库。  

**Star🌟、Fork** 不迷路，同步更新。![](https://img.shields.io/github/stars/iscyy/yoloair?style=social)

项目地址🌟: https://github.com/iscyy/yoloair


部分改进说明教程🌟: [https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation](https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation)

______________________________________________________________________

### 🌟 Beta Supported Updates
下列功能 YOLOAir-Beta版本内测 已支持🔥

- 完成更新支持 **20+ 种 Transformer系列及其变种主干网络**🌟、**多种MLP网络🌟** 以及 **大部分重参数化结构模型网络** 🚀🚀🚀
- 完成更新支持 **图神经网络**🌟 在YOLO中的应用
- 完成更新支持 多模态🔥 在YOLO中的应用
- 完成更新支持 30+ 种 **Attention注意力机制**🌟、
- 完成更新支持 **多种Head检测头**
- 完成更新支持 YOLOv6模型-v2.0 paper 版本
- 完成更新支持 **YOLO系列网络模型热力图可视化**
(GardCAM、GardCAM++等)支持YOLOv3、YOLOv3-SPP、YOLOv4、YOLOv5、YOLOR、YOLOv7、Scaled_YOLOv4、TPH-YOLO、SPD-YOLO、PP-YOLO以及自定义网络模型等模型
- 支持Adaptive Training Sample Selection 标签分配策略和 Task Alignment Learning 标签分配策略
- 完成更新集成 PPYOLOE 算法模型
- 完成更新集成 TOOD 算法
- 其他 各种
______________________________________________________________________

### 🌟 To Do

- 更新支持多任务, 包括目标检测、分类、分割任务
______________________________________________________________________

### 主要特性🚀

🚀支持更多的YOLO系列算法模型改进(持续更新...)

YOLOAir 算法库汇总了多种主流YOLO系列检测模型，一套代码集成多种模型: 
- 内置集成 YOLOv5 模型网络结构、YOLOv7 模型网络结构、 YOLOv6 模型网络结构、PP-YOLO 模型网络结构、PP-YOLOv2 模型网络结构、PP-YOLOE 模型网络结构、PP-YOLOEPlus 模型网络结构、YOLOR 模型网络结构、YOLOX 模型网络结构、ScaledYOLOv4 模型网络结构、YOLOv4 模型网络结构、YOLOv3 模型网络结构、YOLO-FaceV2模型网络结构、TPH-YOLOv5模型网络结构、SPD-YOLO模型网络结构、SlimNeck-YOLO模型网络结构、YOLOv5-Lite模型网络结构、PicoDet模型网络结构等持续更新中...

|目标检测|目标检测|
:-------------------------:|:-------------------------:
<img src='docs/image/test.jpg' width='500px'   alt='ingishvcn'>  |  <img src='docs/image/zebra.jpg' width='500px'   alt='ingishvcn'>

|目标分类|目标分割|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/b1ca7795b70c4b6086b5e6b43b687c1b.jpeg#pic_center' width='500px'   alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/f0a70ccc62864f3996dc9b6ecbbd0c34.jpeg#pic_center' width='500px'   alt='ingishvcn'>

|||
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/e7f5b7c2bf974961a2183436b56c09a3.png#pic_center' width='500px'   alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/0ce7c7584f2149c980d7e292fc1fcd24.jpeg#pic_center' width='500px'   alt='ingishvcn'>

- 以上多种检测算法 和 其他相关任务 使用统一模型代码框架，**集成在 YOLOAir 库中，统一任务形式、统一应用方式**。🌟便于科研者用于论文算法模型改进，模型对比，实现网络组合多样化。🌟工程算法部署落地更便捷，包含轻量化模型和精度更高的模型，根据场景合理选择，在精度和速度俩个方面取得平衡。同时该库支持解耦不同的结构和模块组件，让模块组件化，通过组合不同的模块组件，用户可以根据不同数据集或不同业务场景自行定制化构建不同检测模型。

🚀支持加载YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOR、Scaled_YOLO等网络的官方预训练权重进行迁移学习

🚀支持更多Backbone

-  CSPDarkNet系列   
-  HorNet系列  
-  ResNet系列  
-  RegNet 系列  
-  ShuffleNet系列  
-  Ghost系列  
-  MobileNet系列  
-  EfficientNet系列  
-  ConvNext系列  
-  RepLKNet系列  
-  重参数化系列  
-  RepVGG系列  
-  RepMLP系列  
-  ACNet系列
-  RepConv系列  
-  OREPA系列
-  Mobileone系列  
-  自注意力Transformer系列  
-  MobileViT系列
-  BoTNet-Transfomrer  
-  CoTNet-Transfomrer  
-  Swin-Transfomrer  
-  Swin-Transfomrer(v2)  
-  以及其他trans系列
持续更新中🎈🚀🚀🚀
注: (YOLOAir(Beta版本)🔥 已经完成更新 20+ 种Transformer系列主干网络、多种MLP网络 以及 绝大部分重参数化结构模型网络)

🚀支持更多Neck
- FPN  
- PANet  
- RepPAN
- BiFPN
- CSPPAN
- SlimNeck
- ELANPAN  
 持续更新中🎈

🚀支持更多检测头Head  
-  YOLOv4、YOLOv5 Head检测头;
-  YOLOR 隐式学习Head检测头;
-  YOLOX的解耦合检测头Decoupled Head、DetectX Head;
-  自适应空间特征融合 检测头ASFF Head;
-  YOLOv6-v2.0 Efficient decoupled head;
-  YOLOv7检测头IAuxDetect Head, IDetect Head等;
-  PPYOLOE Efficient Task-aligned head with DFL and VFL

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

- 🔥🔥🔥 重磅！！！作为开源项目补充，推荐一个注意力算法代码库[External-Attention](https://github.com/xmu-xiaoma666/External-Attention-pytorch)，里面汇总整理很全面，包含各种Attention等代码，代码简洁易读，一行代码实现Attention机制。

🚀更多空间金字塔池化结构  
- SPP
- SPPF
- ASPP
- RFB
- SPPCSPC  
- SPPFCSPC  
- SimSPPF
持续更新中🎈    

🚀支持更多Loss   
- ComputeLoss  
- ComputeLoss(v5)  
- ComputeLoss(v6)  
- ComputeLoss(X)  
- ComputeLossAuxOTA(v7)  
- ComputeLossOTA(v7)  
- ComputeNWDLoss  
- 其他Loss

🚀支持 Anchor-base 和 Anchor-Free  

- 🌟 YOLOv5、YOLOv7、YOLOv3、YOLOv4、YOLOR、ScaledYOLOv4、PPYOLO、PPYOLOv2、Improved-YOLOv5、Improved-YOLOv7

- 🌟 YOLOX、YOLOv6 (Paper)、PPYOLOE、PPYOLOE+

🚀支持多种标签分配策略  
- Multi Anchor策略
- YOLOv5 标签分配策略
- SimOTA 标签分配策略
- YOLOv7 标签分配策略
- Adaptive Training Sample Selection 标签分配策略
- Task Alignment Learning 标签分配策略
- 其他改进的标签分配策略
持续更新中🎈  

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

🚀 YOLO系列网络模型热力图可视化(GardCAM、GardCAM++等)
支持YOLOv3、、YOLOv3-SPP、YOLOv4、YOLOv5、YOLOR、YOLOv7Scaled_YOLOv4、TPH-YOLO、SPD-YOLO以及自定义网络模型等模型 (YOLOAir(Beta版本内测)🔥已支持)

🚀主流网络模型结构图汇总: [模型🔗](https://github.com/iscyy/yoloair/blob/main/docs/document/model_.md) 

以上组件模块使用统一模型代码框架、统一任务形式、统一应用方式，**模块组件化**🚀 可以帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法，构建更强大的网络模型。

说明: 以上主要特性支持 包括 Main版本 和 Beta版本，部分特性暂时只完成更新在 Beta 中，不是所有更新都直接在 Main 中，后续 Beta 版本内测之后逐渐加入到 Main版本中。

______________________________________________________________________

### 内置网络模型配置支持✨

🚀包括基于 YOLOv5、YOLOv7、YOLOX、YOLOR、YOLOv3、YOLOv4、Scaled_YOLOv4、PPYOLO、PPYOLOE、PPYOLOEPlus、Transformer、YOLO-FaceV2、PicoDet、YOLOv5-Lite、TPH-YOLOv5、SPD-YOLO等**其他多种改进网络结构等算法模型**的模型配置文件
______________________________________________________________________


### 预训练权重🚀

- YOLOv5
https://github.com/ultralytics/yolov5/releases/tag/v6.1

- YOLOv4
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOv3
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOR
https://github.com/iscyy/yoloair/releases/tag/v1.0

- Scaled_YOLO
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOv7
https://github.com/iscyy/yoloair/releases/tag/v1.0

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

### 论文分享 <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20">

FightingCV公众号每天分享前沿论文动态  
(公众号回复加群,添加小助手加交流群)   
<img src='https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b6f5057da9a8410fa22dcc7566548193~tplv-k3u1fbpfcp-watermark.image?' width='140px'>   
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

[https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation🌟🌟🌟🌟🌟](https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation)

更多模块改进说明教程持续更新中...

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

______________________________________________________________________


### 未来增强✨
后续会持续建设和完善 YOLOAir 生态  
完善集成更多 YOLO 系列模型，持续结合不同模块，构建更多不同网络模型  
横向拓展和引入关联技术等等   

______________________________________________________________________

## Citation✨

```python
@article{2022yoloair,
  title={{YOLOAir}: Makes improvements easy again},
  author={iscyy},
  repo={github https://github.com/iscyy/yoloair},
  year={2022}
}
```

## Statement
<details><summary> <b>Expand</b> </summary>

* The content of this site is only for sharing notes. If some content is infringing, please sending email.

* If you have any question, please discuss with me by sending email.
</details>

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
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
[https://github.com/Gumpest/YOLOv5-Multibackbone](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)  
[https://github.com/cv516Buaa/tph-yolov5](https://github.com/cv516Buaa/tph-yolov5)

</details>