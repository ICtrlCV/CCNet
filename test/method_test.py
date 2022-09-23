#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/6 9:54
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import logging
import json
import socket
import coloredlogs
from torch.utils.data import DataLoader

from network.net import Net
from utils.datasets import Datasets
from utils.initialization import model_initializer
from utils.util import get_classes, get_val_lines, get_root_path, replace_path_str, get_gt_dir

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


def test_lr():
    import matplotlib.pyplot as plt
    import torch
    from math import cos, pi

    def warmup_cos_lr(fn_optimizer, current_epoch, fn_max_epoch, fn_lr_min=0.0, fn_lr_max=0.1, warmup=True):
        warmup_epoch = 5 if warmup else 0
        if current_epoch < warmup_epoch:
            lr = fn_lr_max * current_epoch / warmup_epoch
        elif current_epoch < fn_max_epoch:
            lr = fn_lr_min + (fn_lr_max - fn_lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (fn_max_epoch - warmup_epoch))) / 2
        else:
            lr = fn_lr_min + (fn_lr_max - fn_lr_min) * (
                    1 + cos(pi * (current_epoch - fn_max_epoch) / fn_max_epoch)) / 2
        logger.info(lr)
        for param_group in fn_optimizer.param_groups:
            param_group["lr"] = lr

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    lr_max = 0.1
    lr_min = 0.00001
    max_epoch = 200
    lrs = []
    for epoch in range(200):
        warmup_cos_lr(fn_optimizer=optimizer, current_epoch=epoch, fn_max_epoch=max_epoch, fn_lr_min=lr_min,
                      fn_lr_max=lr_max,
                      warmup=True)
        # logger.info(optimizer.param_groups[0]["lr"])
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()

    plt.plot(lrs)
    plt.show()


def test_datasets():
    val_lines = get_val_lines(os.path.join(root, "datasets/NEUDET/val.txt"))
    input_shape = [224, 224]
    num_classes, class_len = get_classes(os.path.join(root, "datasets/NEUDET/classes.txt"))
    data_path = os.path.join(root, "datasets/NEUDET")
    epochs_num = 200
    datasets = Datasets(annotation=val_lines, input_shape=input_shape, num_classes=num_classes, train=False,
                        path=data_path, epoch=epochs_num, mosaic=False)
    dataloader = DataLoader(dataset=datasets, batch_size=4, shuffle=True, drop_last=True)
    for current_iter, batch in enumerate(dataloader):
        image, label, image_info = batch
        logger.info(current_iter, image.shape, label.shape, image_info.shape)


def test_model_type():
    type_list = ["s", "m", "l", "x"]
    for type in type_list:
        depth, width = model_initializer(type=type)
        logger.info(f"depth={depth}, width={width}")


def test_get_gt_dir():
    class_names, num_classes = get_classes(os.path.join(root, "datasets/NEUDET/classes.txt"))
    val_lines = get_val_lines(os.path.join(root, "datasets/NEUDET/val.txt"))
    save_gt_path = os.path.join(root, "results/1662448384.497914/gt")
    anno_path = os.path.join(root, "datasets/NEUDET/Annotations")
    get_gt_dir(save_gt_path, val_lines, anno_path, class_names)


def test_socket2springboot():
    model_path = os.path.join(root, "weights/NEUDET/NEUDET.pth")
    image1 = os.path.join(root, "asserts/inclusion_1.jpg")
    image2 = os.path.join(root, "asserts/patches_235.jpg")
    image3 = os.path.join(root, "asserts/rolled-in_scale_264.jpg")
    msg_json = {"model_name": "Net", "model_path": model_path,
                "dataset": "NEUDET",
                "input_shape": [224, 224], "conf_thres": 0.5, "nms_thres": 0.6,
                "image_path": [image1, image2, image3]}
    logger.info(msg_json)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # host = "127.0.1.1"
    # host = "192.168.16.1"
    host = socket.gethostname()
    client_socket.bind((host, 12346))
    client_socket.connect((host, 12345))
    msg = json.dumps(msg_json)
    client_socket.send(msg.encode("utf-8"))
    client_socket.send("over".encode("utf-8"))
    client_socket.close()
    logger.info("Send message successfully!")


def test_socket2springboot_repeatedly():
    model_path = os.path.join(root, "weights/NEUDET/NEUDET.pth")
    image1 = os.path.join(root, "asserts/inclusion_1.jpg")
    image2 = os.path.join(root, "asserts/patches_235.jpg")
    image3 = os.path.join(root, "asserts/rolled-in_scale_264.jpg")
    msg_json = {"model_name": "Net", "model_path": model_path,
                "dataset": "NEUDET",
                "input_shape": [224, 224], "conf_thres": 0.5, "nms_thres": 0.6,
                "image_path": [image1, image2, image3]}
    logger.info(msg_json)
    port_list = [12346, 12347, 12348, 12349, 12350, 12351, 12352, 12353, 12354, 12355, 12356, 12357, 12358, 12359,
                 12360, 12361]
    for port in port_list:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # host = "127.0.1.1"
        # host = "192.168.16.1"
        host = socket.gethostname()
        client_socket.bind((host, port))
        client_socket.connect((host, 12345))
        msg = json.dumps(msg_json)
        client_socket.send(msg.encode("utf-8"))
        client_socket.send("over".encode("utf-8"))
        client_socket.close()


def test_get_root():
    current_path = replace_path_str(os.path.abspath(os.path.dirname(__file__)))
    logger.info(current_path)
    root = current_path[:current_path.find("Net/") + len("Net/")]
    logger.info(root)


def test_socket2springboot_different_channel():
    model_path = os.path.join(root, "weights/NEUDET/NEUDET.pth")
    image1 = os.path.join(root, "asserts/inclusion_1.jpg")
    image2 = os.path.join(root, "asserts/different_channel.gif")
    image3 = os.path.join(root, "asserts/rolled-in_scale_264.jpg")
    msg_json = {"model_name": "Net", "model_path": model_path,
                "dataset": "NEUDET",
                "input_shape": [224, 224], "conf_thres": 0.5, "nms_thres": 0.6,
                "image_path": [image1, image2, image3]}
    logger.info(msg_json)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    client_socket.bind((host, 12346))
    client_socket.connect((host, 12345))
    msg = json.dumps(msg_json)
    client_socket.send(msg.encode("utf-8"))
    client_socket.send("over".encode("utf-8"))
    client_socket.close()
    logger.info("Send message successfully!")


if __name__ == "__main__":
    # test_lr()
    # test_datasets()
    # test_model_type()
    # test_get_gt_dir()
    test_socket2springboot()
    # test_socket2springboot_repeatedly()
    # test_get_root()
