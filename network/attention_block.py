import torch
from torch import nn


# CBAM注意力机制
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


# ECA注意力机制
class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2, kernel_size=3):
        """
        ECA注意力机制
        Args:
            channel: 通道
            b: 自适应参数
            gamma: 自适应参数
            kernel_size: 卷积核
        """
        super(ECA, self).__init__()
        # https://github.com/BangguWu/ECANet/issues/24#issuecomment-664926242
        # 自适应内核不容易实现，所以固定kernel_size=3
        # 由于输入通道原因不能正确确认内核大小，可能会引起错误
        # kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        # kernel_size = kernel_size if kernel_size % 2 else kernel_size + yolox-s-1600-15

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局空间信息的特征
        y = self.avg_pool(x)
        # ECA模块的两个不同分支
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 多尺度信息融合
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# CA注意力
class CA(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        """
        CA注意力机制

        源码仓库: https://github.com/houqb/CoordAttention/blob/main/mbv2_ca.py
        原论文: https://arxiv.org/abs/2103.02907
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            groups: 组
        """
        super(CA, self).__init__()
        mid_channels = max(8, in_channels // groups)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = HSwish()

    def forward(self, x):
        # [batch, channel, height, width]
        b, c, h, w = x.size()
        # C * H * 1
        x_h = self.pool_h(x)
        # C * 1 * W => C * W * 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # W + H
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # 分离x_h和x_w => C * H * 1 和 C * W * 1
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 转换x_w => C * 1 * W
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        # 乘法加权计算
        y = x * x_w * x_h

        return y


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
