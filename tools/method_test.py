#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/6 9:54
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import json
import socket
from torch.utils.data import DataLoader

from network.net import Net
from eval import get_gt_dir
from utils.datasets import Datasets
from utils.initialization import device_initializer, model_initializer
from utils.util import get_classes, get_train_lines, get_val_lines


def lr_test():
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
        print(lr)
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
        # print(optimizer.param_groups[0]["lr"])
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()

    plt.plot(lrs)
    plt.show()


def datasets_test():
    datasets = Datasets()
    dataloader = DataLoader(dataset=datasets, batch_size=4, shuffle=True, drop_last=True)
    for current_iter, (x, y) in enumerate(dataloader):
        print(current_iter, x.shape, y.shape)


def init_test():
    model_initializer()


def init_model_param():
    depth, width = model_initializer("s")
    print(depth, width)


def test_get_gt_dir():
    class_names, num_classes = get_classes("../datasets/NEUDET/classes.txt")
    val_lines = get_val_lines("../datasets/NEUDET/val.txt")
    save_gt_path = "../results/1662448384.497914/gt"
    anno_path = "../datasets/NEUDET/Annotations"
    get_gt_dir(save_gt_path, val_lines, anno_path, class_names)


def test_socket2springboot():
    msg_json = {"model_name": "Net", "model_path": "../results/1662448384.497914/model_200.pth",
                "dataset": "NEUDET",
                "input_shape": [224, 224], "conf_thres": 0.5, "nms_thres": 0.6,
                "image_path": ["../asserts/inclusion_1.jpg", "../asserts/patches_235.jpg",
                               "../asserts/rolled-in_scale_264.jpg"]}
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.bind(("192.168.16.1", 12346))
    client_socket.connect(("192.168.16.1", 12345))
    msg = json.dumps(msg_json)
    client_socket.send(msg.encode("utf-8"))
    client_socket.send("over".encode("utf-8"))
    client_socket.close()
    print("发送成功")


if __name__ == "__main__":
    # lr_test()
    # datasets_test()
    # init_test()
    # init_model_param()
    # test_get_gt_dir()
    test_socket2springboot()
