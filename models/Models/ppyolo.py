import numpy as np
import torch
import torch.nn as nn

#PPYOLOE-L
class ResSPP(nn.Module):   #res SPP

    def __init__(self, c1 = 1024 ,c2 = 384,n = 3,act='swish',k = (5,9,13)):
        super(ResSPP, self).__init__()
        c_ = c2
        if c2 == 1024:
            c_ = c2//2
        self.conv1 = ConvBNLayer(c1, c_, 1, act=act)  # CBR
        self.basicBlock_spp1 = BasicBlock(c_, c_,shortcut=False)
        self.basicBlock_spp2 = BasicBlock(c_, c_, shortcut=False)
        self.spp =nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv2 = ConvBNLayer(c_*4, c_, 1, act=act)
        self.basicBlock_spp3 = BasicBlock(c_, c_, shortcut=False)
        self.basicBlock_spp4 = BasicBlock(c_, c_, shortcut=False)
        self.n = n



    def forward(self, x):
        y1 = self.conv1(x)
        if self.n == 3:
            y1 = self.basicBlock_spp1(y1)
            y1 = self.basicBlock_spp2(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp3(y1)
        elif self.n == 1:
            y1 = self.basicBlock_spp1(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
        elif self.n == 2:
            y1 = self.basicBlock_spp1(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp2(y1)
        elif self.n == 4:
            y1 = self.basicBlock_spp1(y1)
            y1 = self.basicBlock_spp2(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp3(y1)
            y1 = self.basicBlock_spp4(y1)
        return y1

#https://github.com/Nioolek/PPYOLOE_pytorch
#https://github.com/PaddlePaddle/PaddleDetection
class CSPResNet(nn.Module):

    def __init__(self,c1,c2,n,conv_down,infor = 'backbone',act='swish'):
        super(CSPResNet, self).__init__()
        self.backbone = CSPResStage(BasicBlock, c1, c2, n,conv_down,infor, act=act)

    def forward(self, x):
        x = self.backbone(x)
        return x


class CSPResNet_CBS(nn.Module):

    def __init__(self,c1=3,c2=64,use_large_stem=True,act='swish'):
        super(CSPResNet_CBS, self).__init__()
        if use_large_stem:
            self.stem = nn.Sequential(
                (ConvBNLayer(c1, c2 // 2, 3, stride=2, padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2 // 2,3,stride=1,padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2,3,stride=1,padding=1,act = act))
            )
        else:
            self.stem = nn.Sequential(
                (ConvBNLayer(3, c2 // 2, 3, stride=2, padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2,3,stride=1,padding=1,act = act)))

    def forward(self, x):
        x = self.stem(x)
        return x

class ConvBNLayer(nn.Module):  #CBS,CBR

    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act='swish'):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

#CSPRes
class CSPResStage(nn.Module):
    def __init__(self, block_fn, c1, c2, n, stride, infor = "backbone",act='relu', attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (c1 + c2) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(c1, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)  #CBR 1x1,BN,RELU
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)  # CBR 1x1,BN,RELU
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for i in range(n)]) #n Res Block
        if attn: #effective SE
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, c2, 1, act=act)#CBR

        if infor == "neck":
            _c2 = c2//2
            self.conv1 = ConvBNLayer(c1, _c2, 1, act=act)
            self.conv2 = ConvBNLayer(c1, _c2, 1, act=act)
            self.attn = None #neck中无effective SE
            self.conv3 = ConvBNLayer(c2, c2, 1, act=act)
            self.blocks = nn.Sequential(*[block_fn(_c2, _c2, act=act, shortcut=False) for i in range(n)])  # n Res Block,no shortcut in neck

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        if self.deploy == False:
            self.conv1 = ConvBNLayer(
                ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.conv2 = ConvBNLayer(
                ch_in, ch_out, 1, stride=1, padding=0, act=None)
        else:
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        self.act = get_activation(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if self.deploy:
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def switch_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__(self.conv1)
        self.__delattr__(self.conv2)
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

#EffectiveSELayer
class EffectiveSELayer(nn.Module):
    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

#Res Blocks = CBS + RepVGG Block  concat
class BasicBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x+y
        else:
            return y

def identity(x):
    return x

__all__ = [

    'mish',
    'silu',
    'swish',
    'identity',
]
def mish(x):
    return nn.mish(x)

def relu(x):
    return nn.relu(x)
def silu(x):
    return nn.silu(x)


def swish(x):
    return x * nn.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError("Unsupported act type: {}".format(name))

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)
