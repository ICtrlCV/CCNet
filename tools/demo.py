#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/7 16:49
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
from collections import OrderedDict

import coloredlogs
import cv2
import numpy as np
import torch
from network.net import Net
from utils.bbox import decode_outputs, post_process
from utils.initialization import device_initializer
from utils.util import get_classes, draw_rectangle

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def inference(model, image, input_shape, conf_thres, nms_thres, device):
    # 转化成RGB
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    mode = "video"
    model_path = "../results/1662448384.497914/model_200.pth"
    input_shape = [224, 224]
    conf_thres = 0.5
    nms_thres = 0.6
    class_names, num_classes = get_classes("../datasets/NEUDET/classes.txt")
    device = device_initializer()

    model = Net(depth=0.33, width=0.5, num_classes=6, act="silu")
    model_dict = model.state_dict()
    weights_dict = torch.load(model_path, map_location=device)
    weights_dict = {k: v for k, v in weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(weights_dict)
    model.load_state_dict(OrderedDict(model_dict))
    model.to(device)
    model.eval()

    if mode == "image":
        img_path = "../asserts/inclusion_1.jpg"
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
