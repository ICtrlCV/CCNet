#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/5 10:50
    @Author : chairc
    @Site   : https://github.com/chairc
"""

import torch
import torch.nn as nn


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
        x = torch.cat((x[..., ::2, ::2], x[..., ::2, 1::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        x = self.conv(x)
        return x


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


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        """
            通道注意力机制
        Args:
            in_channel: 输入通道
            ratio: 比率
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
            空间注意力机制
        Args:
            kernel_size: 卷积核
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channel, ratio=8, kernel_size=7):
        """
            CBAM注意力机制
        Args:
            in_channel: 通道数
            ratio: 比率
            kernel_size: 卷积核
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
