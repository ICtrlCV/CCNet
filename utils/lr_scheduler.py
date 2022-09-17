#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/6 9:37
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math


def set_optimizer_lr(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
