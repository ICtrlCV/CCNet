#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/5 10:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(name="silu", inplace=True):
    """
        获取激活函数
    Args:
        name: 激活函数名称
        inplace:

    Returns: 激活函数

    """
    if name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "relu6":
        act = nn.ReLU6(inplace=inplace)
    elif name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported activation function type: {}".format(name))
    return act


class BaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups=1, bias=False, act="silu"):
        """
            初始化基础卷积块
        Args:
            in_channel: 输入通道
            out_channel: 输出通道
            kernel_size: 卷积核大小
            stride: 步长
            groups: 组
            bias: 偏移
            act: 激活函数
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = get_activation_function(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPFBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=5, act="silu"):
        """
            初始化SPPFBottleneck，该方法等价于SPP
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_sizes: 卷积核大小
            act:
        """
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, mid_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(mid_channels * 4, out_channels, kernel_size=1, stride=1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = torch.cat((x, y1, y2, self.m(y2)), dim=1)
        out = self.conv2(out)
        return out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        """
            CSPLayer实现
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            n: bottleneck个数
            shortcut: 短连接
            expansion: 通道扩张倍数
            act: 激活函数
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, kernel_size=1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class AMFFCSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        """
            AMFFCSPLayer实现，优化Bottleneck
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            n: bottleneck个数
            shortcut: 短连接
            expansion: 通道扩张倍数
            act: 激活函数
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, kernel_size=1, stride=1, act=act)
        module_list = [
            AMFFBottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class SpaceToDepth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        """
            将宽度和高度信息集中到通道空间中
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_size: 卷积核
            stride: 步长
            act: 激活函数
        """
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # 通道变为4倍，长宽减半
        top_left = x[..., ::2, ::2]
        top_right = x[..., ::2, 1::2]
        bottom_left = x[..., 1::2, ::2]
        bottom_right = x[..., 1::2, 1::2]
        x = torch.cat((top_left, top_right, bottom_left, bottom_right), dim=1)
        x = self.conv(x)
        return x


class FocusReplaceConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding, groups=1, bias=False, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding,
                              groups=groups, bias=bias, )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation_function(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu", ):
        """
            标准bottleneck
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            shortcut: 短连接
            expansion: 扩张通道倍数
            act:
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, kernel_size=3, stride=1, act=act)

        # 使用短连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y


class AMFFBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        """
            AMFF-YOLOX中提出的Bottleneck
            详细参考论文：https://doi.org/10.3390/electronics12071662
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=1,
                               bias=False)
        self.act = get_activation_function(act, inplace=True)
        # 使用短连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn(y)
        y = self.conv2(y)
        y = self.act(y)

        if self.use_add:
            y = y + x
        return y


class MobileViT(nn.Module):
    def __init__(self, in_channel=1024, out_channel=1024, d_model=512, dim_feedforward=2048, nhead=2,
                 num_encoder_layers=6, num_decoder_layers=6):
        """
            MobileViT复现
        Args:
            in_channel: 输入通道
            out_channel: 输出通道
            d_model: 编码器解码器输入中预期特征的数量
            dim_feedforward: 前馈网络模型的维度
            nhead: 多头注意力模型中的头数
            num_encoder_layers: 编码器中的子编码器层数
            num_decoder_layers: 解码器中的子编码器层数
        """
        super(MobileViT, self).__init__()
        self.d_model = d_model
        # 3x3
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                               padding=1, groups=out_channel)
        # 1x1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=d_model, kernel_size=1, groups=1)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

        # 1x1
        self.conv3 = nn.Conv2d(in_channels=d_model, out_channels=out_channel, kernel_size=1, groups=1)
        # 3x3
        self.conv4 = nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=3, stride=1,
                               padding=1, groups=1)

    def forward(self, x):
        x_clone = x.clone()
        h, w = x.shape[2:]

        # 局域性表征
        y = self.conv1(x)
        y = self.conv2(y)

        # 全局性表征
        y = y.permute(0, 2, 3, 1)
        y = y.view(-1, h * w, self.d_model)
        y = self.transformer(y, y)
        y = y.view(-1, h, w, self.d_model)
        y = y.permute(0, 3, 1, 2)

        # 融合
        y = self.conv3(y)
        y = torch.cat([x, y], dim=1)
        y = self.conv4(y)
        return y


def asff_auto_pad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ASFFConv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, act=True):
        super(ASFFConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, asff_auto_pad(kernel, padding),
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = ASFFConv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = ASFFConv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = ASFFConv(self.inter_dim, int(1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = ASFFConv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = ASFFConv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = ASFFConv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = ASFFConv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = ASFFConv(int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = ASFFConv(self.inter_dim, int(256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = ASFFConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = ASFFConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = ASFFConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = ASFFConv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        # max feature
        global level_0_resized, level_1_resized, level_2_resized
        x_level_0 = x[2]
        # mid feature
        x_level_1 = x[1]
        # min feature
        x_level_2 = x[0]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
