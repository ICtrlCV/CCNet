#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/5 17:41
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import numpy as np
import torch
from torchvision.ops import nms, boxes


def decode_outputs(outputs, input_shape):
    """
        对模型推理结果的解码器
    Args:
        outputs: 模型推理结果
        input_shape: 输入尺寸

    Returns: 解码后结果

    """
    grids = []
    strides = []
    # 将outputs后两位依次提取出来
    hw = [x.shape[-2:] for x in outputs]

    # 1. outputs输入前代表每个特征层的预测结果：
    # (batch_size, 4 + 1 + num_classes, 80, 80) -> (batch_size, 4 + 1 + num_classes, 6400)
    # (batch_size, 4 + 1 + num_classes, 40, 40) -> (batch_size, 4 + 1 + num_classes, 1600)
    # (batch_size, 4 + 1 + num_classes, 20, 20) -> (batch_size, 4 + 1 + num_classes, 400)
    # 2. 堆叠过程：
    # (batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400) -> (batch_size, 4 + 1 + num_classes, 8400)
    # 3. 堆叠后交换维度结果：
    # (batch_size, 8400, 4 + 1 + num_classes)
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

    # 获得每一个特征点属于每一个种类的概率
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        # 根据每个特征层的高宽生成网格点
        # torch.meshgrid:生成网格，可以用于生成坐标。
        # 函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数
        # grid_x与grid_y都是h.size行，w.size列的
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

        # (1, 6400, 2)
        # (1, 1600, 2)
        # (1, 400, 2)
        # 生成完的网格点进行堆叠
        # grid = torch.stack((grid_x, grid_y), dim=2).view(1, -1, 2)
        grid = torch.stack((grid_x, grid_y), dim=2).reshape(1, -1, 2)
        shape = grid.shape[:2]
        grids.append(grid)
        # 保存每一个特征层的步长
        # 创建一个shape[0] * shape[1]大小的填充值为input_shape[0] / h的矩阵
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))

    # 1. 将网格点堆叠：
    # (1, 6400, 2)
    # (1, 1600, 2)
    # (1, 400, 2)
    # 2. 堆叠结果：
    # (1, 8400, 2)
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())

    # 根据网格点进行解码
    # 调整后预测框中心，outputs前两位
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    # 调整后预测框宽高，outputs的3~4位
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

    # 归一化
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]

    # 输出结果为 中心点坐标 + 宽高 的形式
    return outputs


def post_process(prediction, num_classes, input_shape, image_info, conf_thres=0.5, nms_thres=0.4):
    """
        后处理
    Args:
        prediction: 预测值
        num_classes: 类别个数
        input_shape: 输入尺寸
        image_info: 图片信息
        conf_thres: 置信度阈值
        nms_thres: 非极大值抑制阈值

    Returns: 后处理结果

    """
    # 将预测结果的格式转换成左上角右下角的格式
    # prediction  [batch_size, num_anchors, 85]
    box_corner = prediction.new(prediction.shape)
    # 左上角x = 中心点x - 宽w的一半
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # 左上角y = 中心点x - 高h的一半
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # 右下角x = 中心点x + 宽w的一半
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # 右下角x = 中心点x + 高h的一半
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    # 对输入图片进行循环，一般只会进行一次
    for i, image_pred in enumerate(prediction):

        # 对种类预测部分取最大
        # torch.max会返回list中的最大是和索引值
        # class_conf  [num_anchors, 1]    种类置信度
        # class_pred  [num_anchors, 1]    种类
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], dim=1, keepdim=True)

        # 利用置信度>阈值的结果进行第一轮筛选，大于保留，小于就舍弃
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue

        # 预测结果进行堆叠
        # detections含有的元素为[num_anchors, 7]
        # 其中7中含有的内容为：x1, y1, x2, y2, obj_conf(是否包含物体置信度), class_conf(包含物体类型的预测框置信度), class_pred(预测框属于的种类)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), dim=1)
        detections = detections[conf_mask]

        # 非极大抑制
        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        output[i] = detections[nms_out_index]

        # 图片灰条调整
        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]

            # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
            box_yx = box_xy[..., ::-1]
            box_hw = box_wh[..., ::-1]
            input_shape = np.array(input_shape)
            image_shape = np.array([image_info[i][2], image_info[i][1]])

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)
            re_boxes = np.concatenate(
                [box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
            re_boxes *= np.concatenate([image_shape, image_shape], axis=-1)
            output[i][:, :4] = re_boxes
    return output
