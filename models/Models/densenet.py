"""Pytorch Densenet implementation w/ tweaks
This file is a copy of https://github.com/pytorch/vision 'densenet.py' (BSD-3-Clause) with
fixed kwargs passthrough and addition of dynamic global avg/max pool.
"""
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.jit.annotations import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import BatchNormAct2d, create_norm_act_layer, BlurPool2d, create_classifier


__all__ = ['DenseNet']


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.conv0', 'classifier': 'classifier',
    }


default_cfgs = {
    'densenet121': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenet121_ra-50efcf5c.pth'),
    'densenet121d': _cfg(url=''),
    'densenetblur121d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/densenetblur121d_ra-100dcfbc.pth'),
    'densenet169': _cfg(url='https://download.pytorch.org/models/densenet169-b2777c0a.pth'),
    'densenet201': _cfg(url='https://download.pytorch.org/models/densenet201-c1103571.pth'),
    'densenet161': _cfg(url='https://download.pytorch.org/models/densenet161-8d451a50.pth'),
    'densenet264': _cfg(url=''),
    'densenet264d_iabn': _cfg(url=''),
    'tv_densenet121': _cfg(url='https://download.pytorch.org/models/densenet121-a639ec97.pth'),
}


class DenseLayer(nn.Module):
    def __init__(
            self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d,
            drop_rate=0., memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bottleneck_fn(self, xs):
        # type: (List[torch.Tensor]) -> torch.Tensor
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, x):
        # type: (List[torch.Tensor]) -> bool
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        def closure(*xs):
            return self.bottleneck_fn(xs)

        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, x):  # noqa: F811
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)

        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=nn.ReLU,
            drop_rate=0., memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d, aa_layer=None):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


'''
██╗   ██╗ ██████╗ ██╗      ██████╗      █████╗     ██╗    ██████╗ 
╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ██╔══██╗    ██║    ██╔══██╗
 ╚████╔╝ ██║   ██║██║     ██║   ██║    ███████║    ██║    ██████╔╝
  ╚██╔╝  ██║   ██║██║     ██║   ██║    ██╔══██║    ██║    ██╔══██╗
   ██║   ╚██████╔╝███████╗╚██████╔╝    ██║  ██║    ██║    ██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝     ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝
'''

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(
            self, idx = 0,layer_name = "", growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000, in_chans=3, global_pool='avg',
            bn_size=4, stem_type='', norm_layer=BatchNormAct2d, aa_layer=None, drop_rate=0,
            memory_efficient=False, aa_stem_only=True):
        self.num_classes = num_classes
        self.layer_name = layer_name
        self.drop_rate = drop_rate
        self.idx = idx
        super(DenseNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type  # 3x3 deep stem
        num_init_features = growth_rate * 2

        if aa_layer is None:
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            stem_pool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=num_init_features, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = growth_rate
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (growth_rate // 4)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (growth_rate // 4)
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)),
                ('norm0', norm_layer(stem_chs_1)),
                ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)),
                ('norm1', norm_layer(stem_chs_2)),
                ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)),
                ('norm2', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))

            ####
            if self.layer_name == "densenet_input":
                self.densenet_input = nn.Sequential(OrderedDict([
                    ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('norm0', norm_layer(num_init_features)),
                    ('pool0', stem_pool),
                ]))  #1

            self.densenet_layer = [nn.Sequential(),nn.Sequential(),nn.Sequential(),nn.Sequential()]  #4
            self.densenet_transition_layer = [nn.Sequential(), nn.Sequential(), nn.Sequential()]  #3
        self.feature_info = [
            dict(num_chs=num_init_features, reduction=2, module=f'features.norm{2 if deep_stem else 0}')]
        current_stride = 4

        # DenseBlocks
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            module_name = f'denseblock{(i + 1)}'
            self.features.add_module(module_name, block)

            self.densenet_layer[i].add_module(module_name, block)
            num_features = num_features + num_layers * growth_rate
            transition_aa_layer = None if aa_stem_only else aa_layer
            if i != len(block_config) - 1:
                self.feature_info += [
                    dict(num_chs=num_features, reduction=current_stride, module='features.' + module_name)]
                current_stride *= 2
                trans = DenseTransition(
                    num_input_features=num_features, num_output_features=num_features // 2,  #256  128
                    norm_layer=norm_layer, aa_layer=transition_aa_layer)
                self.features.add_module(f'transition{i + 1}', trans)
                self.densenet_transition_layer[i].add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2


        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))
        # if i == 3:
        #     self.densenet_layer[3].add_module('norm5', norm_layer(num_features))

        self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.norm5')]
        self.num_features = num_features

        # Linear layer
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        if self.layer_name == "densenet_input":
            x = self.densenet_input(x)
            return x
        elif self.layer_name == "densenet_layer":
            x = self.densenet_layer[self.idx](x)
            return x
        elif self.layer_name == "densenet_transition_layer":
            x = self.densenet_transition_layer[self.idx](x)
            return x
        else:
            x = self.densenet_input(x)
            x = self.densenet_layer[0](x)
            x = self.densenet_transition_layer[0](x)
            x = self.densenet_layer[1](x)
            x = self.densenet_transition_layer[1](x)
            x = self.densenet_layer[2](x)
            x = self.densenet_transition_layer[3](x)
            x = self.densenet_layer[3](x)
            return x
        # x = self.forward_features(x)
        # x = self.global_pool(x)
        # both classifier and block drop?
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        # x = self.classifier(x)



class Densenet121(nn.Module):
    """Constructs a Res2Net-50  model."""

    def __init__(self,cin = 3,cout = 64,idx = 0,layer_name = ""):

        super(Densenet121 , self).__init__()
        self.cout = cout
        self.idx = idx
        self.layer_name = layer_name
        self.densenet121  = DenseNet(idx = self.idx,layer_name = self.layer_name,growth_rate=32, block_config=(6, 12, 24, 16))
    def forward(self, x):
        x = self.densenet121(x)
        return x

class Densenet169(nn.Module):
    """Constructs a Res2Net-50  model."""

    def __init__(self,cin = 3,cout = 64,idx = 0,layer_name = ""):

        super(Densenet169 , self).__init__()
        self.cout = cout
        self.idx = idx
        self.layer_name = layer_name
        self.densenet169  = DenseNet(idx = self.idx,layer_name = self.layer_name,growth_rate=32, block_config=(6, 12, 32, 32))
    def forward(self, x):
        x = self.densenet169(x)
        return x
class Densenet201(nn.Module):
    """Constructs a Res2Net-50  model."""

    def __init__(self,cin = 3,cout = 64,idx = 0,layer_name = ""):

        super(Densenet201 , self).__init__()
        self.cout = cout
        self.idx = idx
        self.layer_name = layer_name
        self.densenet201  = DenseNet(idx = self.idx,layer_name = self.layer_name,growth_rate=32, block_config=(6, 12, 48, 32))
    def forward(self, x):
        x = self.densenet201(x)
        return x

#more densenet models

# @register_model
# def densenetblur121d(pretrained=False, **kwargs):
#     r"""Densenet-121 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenetblur121d', growth_rate=32, block_config=(6, 12, 24, 16), pretrained=pretrained, stem_type='deep',
#         aa_layer=BlurPool2d, **kwargs)
#     return model
#
#
# @register_model
# def densenet121d(pretrained=False, **kwargs):
#     r"""Densenet-121 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenet121d', growth_rate=32, block_config=(6, 12, 24, 16), stem_type='deep',
#         pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def densenet169(pretrained=False, **kwargs):
#     r"""Densenet-169 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenet169', growth_rate=32, block_config=(6, 12, 32, 32), pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def densenet201(pretrained=False, **kwargs):
#     r"""Densenet-201 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenet201', growth_rate=32, block_config=(6, 12, 48, 32), pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def densenet161(pretrained=False, **kwargs):
#     r"""Densenet-161 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenet161', growth_rate=48, block_config=(6, 12, 36, 24), pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def densenet264(pretrained=False, **kwargs):
#     r"""Densenet-264 model from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'densenet264', growth_rate=48, block_config=(6, 12, 64, 48), pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def densenet264d_iabn(pretrained=False, **kwargs):
#     r"""Densenet-264 model with deep stem and Inplace-ABN
#     """
#     def norm_act_fn(num_features, **kwargs):
#         return create_norm_act_layer('iabn', num_features, act_layer='leaky_relu', **kwargs)
#     model = _create_densenet(
#         'densenet264d_iabn', growth_rate=48, block_config=(6, 12, 64, 48), stem_type='deep',
#         norm_layer=norm_act_fn, pretrained=pretrained, **kwargs)
#     return model
#
#
# @register_model
# def tv_densenet121(pretrained=False, **kwargs):
#     r"""Densenet-121 model with original Torchvision weights, from
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     """
#     model = _create_densenet(
#         'tv_densenet121', growth_rate=32, block_config=(6, 12, 24, 16), pretrained=pretrained, **kwargs)
#     return model





