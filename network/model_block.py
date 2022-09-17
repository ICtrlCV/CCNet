#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/5 10:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""

import torch
import torch.nn as nn


# 激活函数
def get_activation_function(name="silu", inplace=True):
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


# 基础卷积块
class BaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, k_size, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = get_activation_function(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# SPPFBottleneck
class SPPFBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=5, act="silu"):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, mid_channels, k_size=1, stride=1, act=act)
        self.conv2 = BaseConv(mid_channels * 4, out_channels, k_size=1, stride=1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = torch.cat((x, y1, y2, self.m(y2)), dim=1)
        out = self.conv2(out)
        return out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
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


class SpaceToDepth(nn.Module):
    """
        将宽度和高度信息集中到通道空间中
    """

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # 通道变为4倍，长宽减半
        x = torch.cat((x[..., ::2, ::2], x[..., ::2, 1::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    # 标准bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu", ):
        super().__init__()

        # 原版
        # expansion=0.5
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)

        # 使用短连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        # 原版
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y


class MobileViT(nn.Module):
    """
        MobileViT复现
    """

    def __init__(self, in_channel=1024, out_channel=1024, d_model=512, dim_feedforward=2048, nhead=2,
                 num_encoder_layers=6, num_decoder_layers=6):
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


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM注意力机制
class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
