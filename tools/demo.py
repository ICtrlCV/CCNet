#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/7 16:49
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import logging

import coloredlogs
import cv2
import numpy as np
import torch
from network.net import Net
from utils.bbox import decode_outputs, post_process
from utils.initialization import device_initializer, model_initializer
from utils.util import get_classes, draw_rectangle, load_model_weights, get_root_path

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


def inference(model, image, input_shape, conf_thres, nms_thres, device):
    """
        模型推理
    Args:
        model: 模型
        image: 图片
        input_shape: 输入尺寸
        conf_thres: 置信度阈值
        nms_thres: 非极大值抑制阈值
        device: 设备

    Returns: 预测后处理的图片

    """
    # 转化成RGB
    try:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        logger.error(f"The image {image} is not a 3 channel picture.")

    # resize
    ih, iw = image.shape[0], image.shape[1]
    h, w = input_shape

    image_info = [[0, iw, ih]]
    image_data = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    # 先进行image数据的归一化，再进行转置，交换通道(0 1 2)变为(2 0 1)，最后再添加一个新的batch_size维度
    image_data = np.array(image_data, dtype="float32")
    image_data /= 255.0
    image_data -= np.array([0.485, 0.456, 0.406])
    image_data /= np.array([0.229, 0.224, 0.225])
    image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

    with torch.no_grad():
        # np转换为Tensor，共享内存
        images = torch.from_numpy(image_data)
        images = images.to(device)

        # 将图像输入网络当中进行预测
        outputs = model(images)
        outputs = decode_outputs(outputs, input_shape)

        # 将预测框进行堆叠，然后进行非极大抑制
        results = post_process(outputs, num_classes, input_shape, image_info, conf_thres=conf_thres,
                               nms_thres=nms_thres)

    cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    # 判断是否检测到物体
    if results[0] is None:
        logger.warning("No image result.")
        return image
    draw_rectangle(results, image, iw, ih, input_shape, class_names)
    return image


if __name__ == "__main__":
    mode = "image"
    dataset = "NEUDET"
    model_path = os.path.join(root, f"weights/{dataset}/{dataset}.pth")
    input_shape = [224, 224]
    conf_thres = 0.5
    # nms阈值不建议修改
    nms_thres = 0.6
    class_names, num_classes = get_classes(os.path.join(root, f"datasets/{dataset}/classes.txt"))
    device = device_initializer()
    depth, width = model_initializer(type="s")

    model = Net(depth=depth, width=width, num_classes=num_classes, act="silu")
    model, _ = load_model_weights(model, model_path, device)
    model.to(device)
    model.eval()

    if mode == "image":
        img_path = os.path.join(root, "asserts/inclusion_1.jpg")
        image = cv2.imread(img_path)
        image = inference(model, image, input_shape, conf_thres, nms_thres, device)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    elif mode == "video":
        cap = cv2.VideoCapture(0)
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                frame = inference(model, frame, input_shape, conf_thres, nms_thres, device)
                cv2.imshow("image", frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
