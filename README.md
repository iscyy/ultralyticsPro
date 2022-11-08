## YOLO Air : Makes improvements easy again

<div align="center">
  <p>
    <a align="center" href="https://github.com/iscyy/yoloair" target="_blank">
      <img width="850" src="https://img-blog.csdnimg.cn/11d5f1f403f74cff9c2f70a0c3b92de4.png"></a>
    <br><br>
  </p>
  <p>YOLOAir是一个基于PyTorch的YOLO算法库。统一模型代码框架、统一应用、统一改进、易于模块组合、构建更强大的网络模型。</p>

  简体中文 | [English](./README_EN.md)

  ![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

  #### **支持**

![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv5-red) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv6-blueviolet) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOX-yellow) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOE-007d65) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv4-green) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-TOOD-6a6da9) 
![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLO-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOv2-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOEPlus-d5c59f) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-MLP-ff69b4) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Attention-green)

[特性🚀](#主要特性) • [使用🍉](#使用) • [文档📒](https://github.com/iscyy/yoloair) • [报告问题🌟](https://github.com/iscyy/yoloair/issues/new) • [更新💪](#-to-do) • [讨论✌️](https://github.com/iscyy/yoloair/discussions)  
• [YOLO改进方式教程及原理参考🚀](#yolo网络模型具体改进方式教程及原理参考)•


![https://github.com/iscyy/yoloair](https://img-blog.csdnimg.cn/f7045ecc4f90430cafc276540dddd702.gif#pic_center)

</div>

## Introduction

**模型多样化**: 基于不同网络模块构建不同检测网络模型。

**模块组件化**: 帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法、模型改进，网络排列组合🏆。构建强大的网络模型。

**统一模型代码框架、统一应用方式、统一调参、统一改进、集成多任务、易于模块组合、构建更强大的网络模型**。

内置集成YOLOv5、YOLOv7、YOLOv6、YOLOX、YOLOR、Transformer、PP-YOLO、PP-YOLOv2、PP-YOLOE、PP-YOLOEPlus、Scaled_YOLOv4、YOLOv3、YOLOv4、YOLO-Face、TPH-YOLO、YOLOv5Lite、SPD-YOLO、SlimNeck-YOLO、PicoDet等模型网络结构... 
集成多种检测算法 和 相关多任务模型 使用统一模型代码框架，**集成在 YOLOAir 库中，统一应用方式**。便于科研者用于论文算法模型改进，模型对比，实现网络组合多样化。包含轻量化模型和精度更高的模型，根据场景合理选择，在精度和速度俩个方面取得平衡。同时该库支持解耦不同的结构和模块组件，让模块组件化，通过组合不同的模块组件，用户可以根据不同数据集或不同业务场景自行定制化构建不同检测模型。

支持集成多任务，包括目标检测、实例分割、图像分类、姿态估计、人脸检测、目标跟踪等任务

<img src='https://img-blog.csdnimg.cn/1589c7f744004401b9d88132de35abe8.jpeg#pic_center' alt='ingishvcn'>

**Star🌟、Fork** 不迷路，同步更新。![](https://img.shields.io/github/stars/iscyy/yoloair?style=social)

项目地址🌟: https://github.com/iscyy/yoloair


部分改进说明教程🌟: [https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation](https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation)
______________________________________________________________________

### 🌟 Beta Supported Updates
下列功能 YOLOAir-Beta版本内测 已支持🔥
- 其他 功能🌟
______________________________________________________________________

### 🌟 To Do

- 支持多任务, 包括 目标检测、分类、分割、姿态估计、人脸检测、目标跟踪 等任务
______________________________________________________________________

### 主要特性🚀

🚀支持更多的YOLO系列算法模型改进(持续更新...)

YOLOAir 算法库汇总了多种主流YOLO系列检测模型，一套代码集成多种模型: 

- 内置集成 YOLOv5 模型网络结构、YOLOv7 模型网络结构、 YOLOv6 模型网络结构、PP-YOLO 模型网络结构、PP-YOLOE 模型网络结构、PP-YOLOEPlus 模型网络结构、YOLOR 模型网络结构、YOLOX 模型网络结构、ScaledYOLOv4 模型网络结构、YOLOv4 模型网络结构、YOLOv3 模型网络结构、YOLO-FaceV2模型网络结构、TPH-YOLOv5模型网络结构、SPD-YOLO模型网络结构、SlimNeck-YOLO模型网络结构、YOLOv5-Lite模型网络结构、PicoDet模型网络结构等持续更新中...


🚀支持更多的网络模型组件
- 如表

<details open>
<div align="center">
  <b>YOLOAir Structural Module Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Head</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>CSPDarkNet系列</li>
        <li>HorNet系列</li>
        <li>ResNet系列</li>
        <li>RegNet系列</li>
        <li>ShuffleNet系列</li>
        <li>RepLKNet系列</li>
        <li>MobileNet系列</li>
        <li>EfficientNet系列</li>
        <li>ConvNext系列</li>
        <li>Ghost系列</li>
        <li>RepVGG系列</li>
        <li>transformer系列</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>PANet</li>
        <li>RepPAN</li>
        <li>BiFPN</li>
        <li>CSPPAN</li>
        <li>SlimNeck</li>
        <li>ELANPAN</li>
        <li>GSNeck</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>YOLOv4Head</li>
          <li>YOLOv5Head</li>
          <li>YOLOv6 Efficient decoupled head </li>
          <li>YOLOv7 IDetect Head</li>
          <li>YOLOv7 IAuxDetect Head</li>
          <li>YOLOX DetectX Head</li>
          <li>自适应空间特征融合 检测头ASFF Head</li>
          <li>YOLOR 隐式学习 Head</li>
          <li>EH-Head</li>
          <li>Dual Weighting Head</li>
          <li>FCOS Head</li>
          <li>Dynamic Head</li>
          ...
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>


<details open>
<div align="center">
  <b>YOLOAir Module Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Attention注意力机制</b>
      </td>
      <td>
        <b>标签分配策略</b>
      </td>
      <td>
        <b>IoU损失函数</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>Self Attention</li>
        <li>Contextual Transformer</li>
        <li>Bottleneck Transformer</li>
        <li>S2-MLP Attention</li>
        <li>AxialAttention</li>
        <li>DANet</li>
        <li>DoubleAttention</li>
        <li>GlobalContextBlock</li>
        <li>ParNetAttention</li>
        <li>PolarizedSelfAttention</li>
        <li>SpatialGroupEnhance copy</li>
        <li>TripletAttention</li>
        <li>SK Attention</li>
        <li>CBAM Attention</li>
        <li>SE Attention</li>
        <li>Coordinate attention</li>
        <li>NAM Attention</li>
        <li>GAM Attention</li>
        <li>ECA Attention</li>
        <li>Shuffle Attention</li>
        <li>CrissCrossAttention</li>
        <li>SOCAttention</li>
        <li>SimAM Attention</li>
        <li>CrissCrossAttention</li>
      </ul>
      </td>
      <td>
      <ul>
          <li>Multi Anchor策略</li>
          <li>YOLOv5 标签分配策略</li>
          <li>SimOTA 标签分配策略</li>
          <li>YOLOv7 标签分配策略</li>
          <li>FreeAnchor 标签分配策略</li>
          <li>AutoAssign 标签分配策略</li>
          <li>Dual Weighting 标签分配策略</li>
          <li>Probabilistic Anchor Assignment 锚分配策略</li>
          <li>其他改进的标签分配策略</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>CIoU</li>
          <li>DIoU</li>
          <li>GIoU</li>
          <li>EIoU</li>
          <li>SIoU</li>
          <li>alpha IOU</li>      
          ...
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>


<details open>
<div align="center">
  <b>Module Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>NMS</b>
      </td>
      <td>
        <b>数据增强</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>ComputeLoss(v3)</li>
        <li>ComputeLoss(v4)</li>
        <li>ComputeLoss(v5)</li>
        <li>ComputeLoss(v6)</li>
        <li>ComputeLoss(X)</li>
        <li>ComputeLossAuxOTA(v7)</li>
        <li>ComputeLossOTA(v7)</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>NMS</li>
        <li>Merge-NMS</li>
        <li>Soft-NMS</li>
        <li>CIoU_NMS</li>
        <li>DIoU_NMS</li>
        <li>GIoU_NMS</li>
        <li>EIoU_NMS</li>
        <li>SIoU_NMS</li>
        <li>Soft-SIoUNMS</li>
        <li>Soft-CIoUNMS</li>
        <li>Soft-DIoUNMS</li>
        <li>Soft-EIoUNMS</li>
        <li>Soft-GIoUNMS</li>
        <li>SimSPPF</li>
        <li>SimSPPF</li>
        <li>SimSPPF</li>  
      </ul>
      </td>
      <td>
        <ul>
          <li>Mosaic</li>
          <li>Copy paste</li>
          <li>Random affine(Rotation, Scale, Translation and Shear)</li>
          <li>MixUp</li>
          <li>HSV</li>
          <li>Random horizontal flip</li> 
          ...
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>


<details open>
<div align="center">
  <b>YOLOAir Module Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>空间金字塔池化结构</b>
      </td>
      <td>
        <b>模块化组件</b>
      </td>
      <td>
        <b>AnchorBased / AnchorFree</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li>SPP</li>
        <li>SPPF</li>
        <li>ASPP</li>
        <li>RFB</li>
        <li>SPPCSPC</li>
        <li>SPPFCSPC</li>
        <li>SimSPPF</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3HB, C3RFEM, MultiSEAM, SEAM, C3STR, SPPCSPC, RepConv, BoT3, Air, CA, CBAM, Involution, Stem, ResCSPC, ResCSPB, ResXCSPB, ResXCSPC, BottleneckCSPB, BottleneckCSPC, ASPP, BasicRFB, SPPCSPC_group, HorBlock, CNeB,C3GC ,C3C2, nn.ConvTranspose2d, DWConvblock, RepVGGBlock, CoT3, ConvNextBlock, SPPCSP, BottleneckCSP2, DownC, BottleneckCSPF, RepVGGBlock, ReOrg, DWT, MobileOne,HorNet...</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>YOLOv5、YOLOv7、YOLOv3、YOLOv4、YOLOR、ScaledYOLOv4、PPYOLO、PPYOLOv2、Improved-YOLOv5、Improved-YOLOv7</li> 
          <li>YOLOX、YOLOv6 (Paper)、PPYOLOE、PPYOLOE+</li>
          ...
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>
</details>


<details>

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
注: (YOLOAir(Beta)🔥 已经完成更新 20+ 种Transformer系列主干网络、多种MLP网络 以及 绝大部分重参数化结构模型网络)

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
- AutoAssign 标签分配策略🌟
- Dual Weighting 标签分配策略🌟
- FreeAnchor 标签分配策略🌟
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

</details>

🚀支持加载YOLOv3、YOLOv4、YOLOv5、YOLOv7、YOLOR、Scaled_YOLO等网络的官方预训练权重进行迁移学习

🚀 YOLO系列网络模型热力图可视化(GardCAM、GardCAM++等)
支持YOLOv3、、YOLOv3-SPP、YOLOv4、YOLOv5、YOLOR、YOLOv7Scaled_YOLOv4、TPH-YOLO、SPD-YOLO以及自定义网络模型等模型 (YOLOAir(Beta版本内测)🔥已支持)

🚀主流网络模型结构图汇总: [模型🔗](https://github.com/iscyy/yoloair/blob/main/docs/document/model_.md) 

以上组件模块使用统一模型代码框架、统一任务形式、统一应用方式，**模块组件化**🚀 可以帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法，构建更强大的网络模型。

说明: 以上主要特性支持 包括 Main版本 和 Beta版本，部分特性暂时只完成更新在 Beta 中，不是所有更新都直接在 Main 中，后续 Beta 版本内测之后逐渐加入到 Main版本中。

______________________________________________________________________

### 内置网络模型配置支持✨

🚀包括基于 YOLOv5、YOLOv7、YOLOX、YOLOR、YOLOv3、YOLOv4、Scaled_YOLOv4、PPYOLO、PPYOLOE、PPYOLOEPlus、Transformer、YOLO-FaceV2、PicoDet、YOLOv5-Lite、TPH-YOLOv5、SPD-YOLO等**其他多种改进网络结构等算法模型**的模型配置文件
______________________________________________________________________

### 效果预览🚀

|目标检测|目标分割|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/0b04579f80d145d7bd2e854753e9f367.jpeg' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/adb10e3c47e440f9acf4a183df9acf05.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|图像分类|实例分割|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/b1ca7795b70c4b6086b5e6b43b687c1b.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d29f6d6fa0624c5cacf107bd5d1a5fa2.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|目标分割|目标跟踪|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/0ce7c7584f2149c980d7e292fc1fcd24.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d9ae8953fb394a74a6b1096a401fc315.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|姿态估计|人脸检测|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/01f41103dc6c416aaeeb4577b87bb363.gif#pic_center' width='300px' height='260px' alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d18a095621b64da69d2a712fa5613976.gif#pic_center' width='300px' height='260px'   alt='ingishvcn'>

|热力图01|热力图02|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/eef8f911702242a5bb3e10a2e3188ca6.jpeg#pic_center' width='300px' height='180px' alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/a22986632c25462cbe6abddc75a01ca5.jpeg#pic_center' width='300px' height='180px'   alt='ingishvcn'>


![yolo](https://img-blog.csdnimg.cn/b962fcd1bfa841399226ca564f22a385.gif#pic_center)
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


## YOLO网络模型具体改进方式教程及原理参考

- [手把手调参最新 YOLOv7 模型 推理部分（一）](https://blog.csdn.net/qq_38668236/article/details/126563528)🌟

- [手把手调参最新 YOLOv7 模型 训练部分（二）](https://blog.csdn.net/qq_38668236/article/details/126566495)🌟

- [论文解读｜清华&Meta提出HorNet，用递归门控卷积进行高阶空间相互作用](https://blog.csdn.net/qq_38668236/article/details/126796291)(推荐)

- [改进YOLOv5系列：30.结合即插即用的动态卷积ODConv](https://blog.csdn.net/qq_38668236/article/details/127097170)(🚀最新推荐)

- [改进YOLOv5系列：首发结合 RepLKNet 构建 最新 RepLKDeXt 结构｜CVPR2022 超大卷积核, 越大越暴力,大到31x31, 涨点高效](https://blog.csdn.net/qq_38668236/article/details/127196758)(🚀最新推荐)

- [改进YOLOv7系列：首发结合最新Centralized Feature Pyramid集中特征金字塔，通过COCO数据集验证强势涨点](https://blog.csdn.net/qq_38668236/article/details/127217449)(🚀最新推荐)

- [改进YOLOv7系列：首发结合最新Transformer视觉模型MOAT结构：交替移动卷积和注意力带来强大的Transformer视觉模型，超强的提升](https://blog.csdn.net/qq_38668236/article/details/127249237)(🚀最新推荐)

- [改进YOLOv5, YOLOv7系列：1.YOLO超全注意力机制汇总 | S2A, SE,SimAM, SKA,ShA, SOCA, CA, CBAM, CrissCrossA, NAM, GAM等](https://blog.csdn.net/qq_38668236/article/details/126086716)🌟(强烈推荐)

- [改进YOLOv5系列：29.YOLOv5 结合 极简又强大的RepVGG 重参数化模型结构](https://blog.csdn.net/qq_38668236/article/details/126735476)🌟

- [改进YOLOv7系列：28.YOLOv7 结合 Swin Transformer V2结构，Swin Transformer V2：通向视觉大模型之路](https://blog.csdn.net/qq_38668236/article/details/126735304)🌟

- [改进YOLOv5系列：27.YOLOv5 结合 Swin Transformer V2结构，Swin Transformer V2：通向视觉大模型之路](https://blog.csdn.net/qq_38668236/article/details/126735107)🌟

- [改进YOLOv7系列：26.CVPR2022. ConvNeXt结合YOLOv7 | 基于ConvNeXt结构 构建 CNeB 模块](https://blog.csdn.net/qq_38668236/article/details/126727055)🌟

- [改进YOLOv7系列：25.YOLOv7 加入RepVGG模型结构，重参数化 极简架构](https://blog.csdn.net/qq_38668236/article/details/126715391)🌟

- [改进YOLOv7系列：24.添加SimAM注意力机制](https://blog.csdn.net/qq_38668236/article/details/126695253)🌟

- [改进YOLOv7系列：23.YOLOv7添加CBAM注意力机制](https://blog.csdn.net/qq_38668236/article/details/126652075)🌟

- [改进YOLOv7系列：22.最新HorNet结合YOLOv7应用](https://blog.csdn.net/qq_38668236/article/details/126626414)🌟

- [改进YOLOv7系列：21.添加ShuffleAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126621483)🌟

- [改进YOLO系列：4.YOLOv5_最新MobileOne结构换Backbone修改，超轻量型架构，移动端仅需1ms推理！苹果最新移动端高效主干网络](https://blog.csdn.net/qq_38668236/article/details/126157859)🌟

- [改进YOLO系列：10.最新HorNet结合YOLO应用首发！ | 多种搭配，即插即用 | Backbone主干、递归门控卷积的高效高阶空间交互](https://blog.csdn.net/qq_38668236/article/details/126410711)🌟

- [改进YOLO系列：9.BoTNet Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126333061)🌟

- [改进YOLO系列：8.增加ACmix结构的修改,自注意力和卷积集成](https://blog.csdn.net/qq_38668236/article/details/126302599)🌟

- [改进YOLO系列：7.修改DIoU-NMS,SIoU-NMS,EIoU-NMS,CIoU-NMS,GIoU-NMS](https://blog.csdn.net/qq_38668236/article/details/126243834)🌟

- [改进YOLO系列：6.修改Soft-NMS,Soft-CIoUNMS,Soft-SIoUNMS](https://blog.csdn.net/qq_38668236/article/details/126245080)🌟

- [改进YOLO系列：5.CotNet Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126226726)🌟

- [改进YOLO系列：4.YOLOv5_最新MobileOne结构换Backbone修改](https://blog.csdn.net/qq_38668236/article/details/126157859)🌟

- [改进YOLO系列：3.Swin Transformer结构的修改](https://blog.csdn.net/qq_38668236/article/details/126122888?spm=1001.2014.3001.5502)🌟

- [改进YOLO系列：2.PicoDet结构的修改](https://blog.csdn.net/qq_38668236/article/details/126087343?spm=1001.2014.3001.5502)🌟

- [改进YOLO系列：1.多种注意力机制修改](https://blog.csdn.net/qq_38668236/article/details/126086716)🌟

- [改进YOLO系列：12.添加ShuffleAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126498783)🌟

- [改进YOLO系列：13.添加CrissCrossAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503278)🌟

- [改进YOLO系列：14.添加S2-MLPv2注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503447)🌟

- [改进YOLO系列：15.添加SimAM注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503550)🌟

- [改进YOLO系列：16.添加SKAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503616)🌟

- [改进YOLO系列：17.添加NAMAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503665)🌟

- [改进YOLO系列：18.添加SOCA注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503809)🌟

- [改进YOLO系列：18.添加CBAM注意力机制](https://blog.csdn.net/qq_38668236/article/details/126503939)🌟

- [改进YOLO系列：19.添加SEAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126504116)🌟

- [改进YOLO系列：20.添加GAMAttention注意力机制](https://blog.csdn.net/qq_38668236/article/details/126504421)🌟

<div align='center'>
    <img src='https://img-blog.csdnimg.cn/88f8adee63e54ae1906b5bac1699b2bd.jpeg#pic_center'  width="600"  alt='ingishvcn'>
</div>

______________________________________________________________________


### YOLOv5教程✨
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

</details>
