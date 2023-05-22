# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/5/18 09:30
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from .model_block import BaseConv, SPPFBottleneck, AMFFCSPLayer, FocusReplaceConv, ASFF
from .attention_block import ECA


# AMFF-YOLOX adopts "Improved CSPDarknet + PANet + ASFF + ECA + Decoupled head" structure
# 可用于TensorRT端部署
class Net(nn.Module):
    def __init__(
            self,
            depth=1.0,
            width=1.0,
            num_classes=80,
            act="silu",
    ):
        """
            初始化模型
        Args:
            depth: 深度
            width: 宽度
            num_classes: 类别个数
            act: 激活函数
        """
        super().__init__()

        # 主干网络

        base_channels = int(width * 64)  # 64
        base_depth = max(round(depth * 3), 1)  # 3

        # stem
        # 原版为Focus，现替换为6 * 6卷积
        self.stem = FocusReplaceConv(3, base_channels, ksize=6, stride=2, padding=2)

        # dark2~dark5中的CSPLayer的n比例1:1:3:1
        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            AMFFCSPLayer(
                base_channels * 2,
                base_channels * 2,
                # bottleneck的个数
                n=base_depth,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            AMFFCSPLayer(
                base_channels * 4,
                base_channels * 4,
                # bottleneck的个数
                n=base_depth * 3,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            AMFFCSPLayer(
                base_channels * 8,
                base_channels * 8,
                # bottleneck的个数
                n=base_depth * 3,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPFBottleneck(base_channels * 16, base_channels * 16, act=act),
            AMFFCSPLayer(
                base_channels * 16,
                base_channels * 16,
                # bottleneck的个数
                n=base_depth,
                shortcut=False,
                act=act,
            ),
        )

        # 特征提取网络

        in_channels = [256, 512, 1024]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = AMFFCSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = AMFFCSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = AMFFCSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = AMFFCSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            act=act,
        )

        # 注意力通道大小
        self.neck_channels = [512, 256, 512, 1024]
        # ECA
        # 对应dark5输出的1024维度通道
        self.eca_1 = ECA(int(in_channels[2] * width))
        # 对应dark4输出的512维度通道
        self.eca_2 = ECA(int(in_channels[1] * width))
        # 对应dark3输出的256维度通道
        self.eca_3 = ECA(int(in_channels[0] * width))
        # FPN CSPLayer 512
        self.eca_fp1 = ECA(int(self.neck_channels[0] * width))
        # PAN CSPLayer pan_out2 downsample 256
        self.eca_pa1 = ECA(int(self.neck_channels[1] * width))
        # PAN CSPLayer pan_out1 downsample 512
        self.eca_pa2 = ECA(int(self.neck_channels[2] * width))
        # PAN CSPLayer pan_out0 1024
        self.eca_pa3 = ECA(int(self.neck_channels[3] * width))

        # ASFF
        self.asff_1 = ASFF(level=0, multiplier=width)
        self.asff_2 = ASFF(level=1, multiplier=width)
        self.asff_3 = ASFF(level=2, multiplier=width)

        # 解耦头

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channel=int(in_channels[i] * width), out_channel=int(256 * width), kernel_size=1, stride=1,
                         act=act))
            # 分类
            self.cls_convs.append(nn.Sequential(*[
                BaseConv(in_channel=int(256 * width), out_channel=int(256 * width), kernel_size=3, stride=1, act=act),
                BaseConv(in_channel=int(256 * width), out_channel=int(256 * width), kernel_size=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            # 预测
            self.reg_convs.append(nn.Sequential(*[
                BaseConv(in_channel=int(256 * width), out_channel=int(256 * width), kernel_size=3, stride=1, act=act),
                BaseConv(in_channel=int(256 * width), out_channel=int(256 * width), kernel_size=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        layer0 = self.stem(x)
        layer1 = self.dark2(layer0)
        layer2 = self.dark3(layer1)
        layer3 = self.dark4(layer2)
        layer4 = self.dark5(layer3)

        layer2_eca = self.eca_3(layer2)
        layer3_eca = self.eca_2(layer3)
        layer4_eca = self.eca_1(layer4)

        # dark5输出的特征层进行卷积
        fpn_out0 = self.lateral_conv0(layer4_eca)  # 1024->512/32
        # 进行上采样
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # 将dark4输出的特征层与上采样结果进行叠加
        f_out0 = torch.cat([f_out0, layer3_eca], 1)  # 512->1024/16
        # 将叠加后的的结果进行一个CSPLayer操作
        # 大小不变，通道缩减
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # ECA
        f_out0 = self.eca_fp1(f_out0)

        # 对CSPLayer结果进行基础卷积
        # 大小不变，通道缩减
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # 对输出的基础卷积进行上采样
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # 将dark3输出的特征层与上采样结果进行叠加
        f_out1 = torch.cat([f_out1, layer2_eca], 1)  # 256->512/8
        # 输出结果到YOLO HEAD
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # ECA
        pan_out2 = self.eca_pa1(pan_out2)

        # 对pan_out2输出结果进行下采样
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # 将pan_out1输出的下采样结果与CSPLayer输出fpn_out1的结果进行叠加
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # 对CSPLayer结果进行基础卷积
        # 特征提取（普通卷积 + 标准化 + 激活函数）
        # 输出结果到YOLO HEAD
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # ECA
        pan_out1 = self.eca_pa2(pan_out1)

        # 下采样，降低特征图大小
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # 将p_out0下采样结果与最下层输出的卷积进行叠加
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # 对CSPLayer结果进行基础卷积
        # 特征提取（普通卷积 + 标准化 + 激活函数）
        # 输出结果到YOLO HEAD
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # ECA
        pan_out0 = self.eca_pa3(pan_out0)

        pan_outputs = (pan_out2, pan_out1, pan_out0)
        # ASFF
        pan_out0 = self.asff_1(pan_outputs)
        pan_out1 = self.asff_2(pan_outputs)
        pan_out2 = self.asff_3(pan_outputs)

        neck_outputs = (pan_out2, pan_out1, pan_out0)
        outputs = []
        for k, x in enumerate(neck_outputs):
            # 利用1x1卷积进行通道整合
            x = self.stems[k](x)

            # 利用两个卷积标准化激活函数来进行特征提取
            cls_feat = self.cls_convs[k](x)

            # 判断特征点所属的种类
            # 80, 80, num_classes
            # 40, 40, num_classes
            # 20, 20, num_classes
            cls_output = self.cls_preds[k](cls_feat)

            # 利用两个卷积标准化激活函数来进行特征提取
            reg_feat = self.reg_convs[k](x)

            # 特征点的回归系数
            # reg_pred 80, 80, 4
            # reg_pred 40, 40, 4
            # reg_pred 20, 20, 4
            reg_output = self.reg_preds[k](reg_feat)

            # 判断特征点是否有对应的物体
            # obj_pred 80, 80, 1
            # obj_pred 40, 40, 1
            # obj_pred 20, 20, 1
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
