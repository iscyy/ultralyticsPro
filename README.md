# YOLOAir:  新的Air, Air, Air

YOLOAir算法库 是一个基于 PyTorch 的一系列 YOLO 检测算法组合工具箱。用来**组合不同模块构建不同网络**。  

<div align='center'>
    <img src='docs/image/logo1.png' width='500px'>
</div>

内置YOLOv5、YOLOv7、YOLOX、YOLOR、Transformer、Scaled_YOLOv4、YOLOv3、YOLOv4、YOLO-Facev2、TPH-YOLOv5、YOLOv5-Lite、PP-YOLO、PicoDet等算法模型(更新中)...

**模块组件化**：帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法、模型改进，网络排列组合🏆。构建强大的网络模型。

**统一模型代码框架、统一应用方式、统一调参、、统一部署、易于模块组合、构建更强大的网络模型**。  

基于 YOLOv5 代码框架，并同步适配 **稳定的YOLOv5_v6.1更新**, 同步v6.1部署生态。使用这个项目之前, 您可以先了解YOLOv5库。  

简体中文 | [English](./README_EN.md)

[特性🚀](#Mainfeatures) • [使用🍉](#Usage) • [文档📒](https://github.com/iscyy/yoloair) • [报告问题🌟](https://github.com/iscyy/yoloair/issues/new)

![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

#### 支持
![](https://img.shields.io/badge/Support-YOLOv5-red) ![](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![](https://img.shields.io/badge/Support-YOLOX-yellow) ![](https://img.shields.io/badge/Support-YOLOv4-green) ![](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4)
![](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![](https://img.shields.io/badge/Support-Attention-green)


项目地址: https://github.com/iscyy/yoloair

### 主要特性🚀

🚀支持更多的YOLO系列算法模型(持续更新...)

YOLOAir 算法库汇总了多种主流YOLO系列检测模型，一套代码集成多种模型: 
- 内置集成 YOLOv5 模型网络结构
- 内置集成 YOLOR 模型网络结构
- 内置集成 YOLOX 模型网络结构
- 内置集成 YOLOv7 模型网络结构
- 内置集成 Scaled_YOLOv4 模型网络结构
- 内置集成 YOLOv4 模型网络结构
- 内置集成 YOLOv3 模型网络结构
- 以及优秀的部分改进模型
- YOLO-FaceV2模型网络结构
- TPH-YOLOv5模型网络结构
- YOLOv5-Lite模型网络结构
- PPYOLO模型网络结构
- PicoDet模型网络结构
...

以上多种检测算法使用统一模型代码框架，**集成在 YOLOAir 库中，统一任务形式、统一应用方式**。🌟便于科研者用于论文算法模型改进，模型对比，实现网络组合多样化。🌟工程算法部署落地更便捷，包含轻量化模型和精度更高的模型，根据场景合理选择，在精度和速度俩个方面取得平衡。同时该库支持解耦不同的结构和模块组件，让模块组件化，通过组合不同的模块组件，用户可以根据不同数据集或不同业务场景自行定制化构建不同检测模型。

🚀模型支持导出ONNX进行TensorRT推理，落地部署。

🚀支持加载YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOR等网络的官方预训练权重进行迁移学习

🚀支持更多Backbone

- `CSPDarkNet系列`、(多种)
`ResNet系列`、(多种)
`ShuffleNet系列`、(多种)
`Ghost系列`、(多种)
`MobileNet系列`、(多种)
`ConvNext系列`、
`RepLKNet系列`、
`RepBlock系列`、(多种)
`自注意力Transformer系列`、(多种)
持续更新中🎈

🚀支持更多Neck

- neck包含FPN、PANet、BiFPN等主流结构。
 持续更新中🎈

🚀支持更多检测头Head  
-  YOLOv4、YOLOv5 Head检测头、
-  YOLOvR 隐式学习Head检测头、
-  YOLOX的解耦合检测头Decoupled Head、
-  自适应空间特征融合 检测头ASFF Head、
-  YOLOv7检测头IAuxDetect Head, IDetect Head等；

🚀支持更多即插即用的注意力机制
- 在网络任何部分即插即用式使用注意力机制，例如SE、CBAM、CA、GAM、ECA...等多种主流注意力机制  
[详情](https://github.com/iscyy/yoloair/blob/main/docs/document/attention.md)

🚀支持更多IoU损失函数
- CIoU、DIoU、GIoU、EIoU、SIoU、alpha IOU等损失函数；  
[详情](https://blog.csdn.net/qq_38668236?type=blog)

🚀支持更多NMS  
- NMS、Merge-NMS、DIoU-NMS、Soft-NMS、CIoU_NMS、DIoU_NMS、GIoU_NMS、EIoU_NMS、SIoU_NMS等;  
[详情](https://blog.csdn.net/qq_38668236?type=blog)

🚀支持更多数据增强
- Mosaic、Copy paste、Random affine(Rotation, Scale, Translation and Shear)、MixUp、Augment HSV(Hue, Saturation, Value、Random horizontal flip

🚀支持更多Loss
- ComputeLoss、ComputeNWDLoss、ComputeXLoss、ComputeLossAuxOTA(v7)、ComputeLossOTA(v7)等
[详情](https://blog.csdn.net/qq_38668236?type=blog)

🚀支持加权框融合(WBF)

🚀 内置多种网络模型模块化组件
- Conv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, GhostConv, 等
详细代码 **./models/common.py文件** 内搜索🔍👉[对应模块链接](https://github.com/iscyy/yoloair/blob/main/models/common.py)

- 👉[网络模型结构图](https://github.com/iscyy/yoloair/blob/main/docs/document/model_.md) 

以上组件模块使用统一模型代码框架、统一任务形式、统一应用方式，**模块组件化**🚀 可以帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法，构建更强大的网络模型。

### 内置网络模型配置支持✨

🚀包括YOLOv3、YOLOv4、Scaled_YOLOv4、YOLOv5、YOLOv6、YOLOv7、YOLOX、YOLOR、Transformer、YOLO-FaceV2、PicoDet、YOLOv5-Lite、PPYOLO、TPH-YOLOv5、**其他多种改进网络结构等算法模型**的yaml配置文件汇总

[多种内置yaml网络模型配置(推荐🌟🌟🌟🌟🌟)](https://blog.csdn.net/qq_38668236/article/details/126237396)

🚀用户可自行改进网络

YOLOv7官方仓库目前一直在更新

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

### Benchmark
Updating...

### YOLO网络模型具体改进方式教程及原理

- 1.[改进YOLOv5系列：1.YOLOv5_CBAM注意力机制修改(其他注意力机制同理)](https://blog.csdn.net/qq_38668236/article/details/126086716)

- 2.[改进YOLOv5系列：2.PicoDet结构的修改🍀](https://blog.csdn.net/qq_38668236/article/details/126087343?spm=1001.2014.3001.5502)

- 3.[改进YOLOv5系列：3.Swin Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126122888?spm=1001.2014.3001.5502)

- 4.[改进YOLOv5系列：4.YOLOv5_最新MobileOne结构换Backbone修改🍀](https://blog.csdn.net/qq_38668236/article/details/126157859)

- 5.[改进YOLOv5系列：5.CotNet Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126226726)

- 6.[改进YOLOv5系列：6.修改Soft-NMS,Soft-CIoUNMS,Soft-SIoUNMS](https://blog.csdn.net/qq_38668236/article/details/126245080)

- 7.[改进YOLOv5系列：7.修改DIoU-NMS,SIoU-NMS,EIoU-NMS,CIoU-NMS,GIoU-NMS](https://blog.csdn.net/qq_38668236/article/details/126243834)

- 1.[手把手带你调参Yolo v5 (v6.1)（一）](https://blog.csdn.net/weixin_43694096/article/details/124378167)🌟强烈推荐

- 2.[手把手带你调参Yolo v5 (v6.1)（二）](https://blog.csdn.net/weixin_43694096/article/details/124411509?spm=1001.2014.3001.5502)🚀

- 3.[如何快速使用自己的数据集训练Yolov5模型](https://blog.csdn.net/weixin_43694096/article/details/124457787)

- 4.[手把手带你Yolov5 (v6.1)添加注意力机制(一)（并附上30多种顶会Attention原理图）](https://blog.csdn.net/weixin_43694096/article/details/124443059?spm=1001.2014.3001.5502)🌟

- 5.[手把手带你Yolov5 (v6.1)添加注意力机制(二)（在C3模块中加入注意力机制）](https://blog.csdn.net/weixin_43694096/article/details/124695537)

- 6.[Yolov5如何更换激活函数？](https://blog.csdn.net/weixin_43694096/article/details/124413941?spm=1001.2014.3001.5502)

- 7.[Yolov5 (v6.1)数据增强方式解析](https://blog.csdn.net/weixin_43694096/article/details/124741952?spm=1001.2014.3001.5502)

- 8.[Yolov5更换上采样方式( 最近邻 / 双线性 / 双立方 / 三线性 / 转置卷积)](https://blog.csdn.net/weixin_43694096/article/details/125416120)🍀

- 9.[Yolov5如何更换EIOU / alpha IOU / SIoU？](https://blog.csdn.net/weixin_43694096/article/details/124902685)🍀

- 10.[Yolov5更换主干网络之《旷视轻量化卷积神经网络ShuffleNetv2》](https://blog.csdn.net/weixin_43694096/article/details/126109839?spm=1001.2014.3001.5501)

- 11.[YOLOv5应用轻量级通用上采样算子CARAFE](https://blog.csdn.net/weixin_43694096/article/details/126148795)

更多模块详细解释持续更新中。。。

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
横向拓展和引入关联技术，如半监督学习等等  
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


Paper:
[https://arxiv.org/abs/2208.02019](https://arxiv.org/abs/2208.02019)  

</details>
