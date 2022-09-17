#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/12 17:05
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer():
    """
        This function init the program when it is the first running.
    :return:
    """
    logger.info("Init program, it is checking the basic setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        logger.info(device_dict)
    else:
        logger.warning("The device is using cpu.")
        device = torch.device("cpu")
    return device


def model_initializer(type="s"):
    """
        This Enum is defined the selection of model, such as depth and width.
    """
    depth_dict = {"tiny": 0.33, "s": 0.33, "m": 0.67, "l": 1.00, "x": 1.33}
    width_dict = {"tiny": 0.375, "s": 0.50, "m": 0.75, "l": 1.00, "x": 1.25}
    depth = depth_dict.get(type)
    width = width_dict.get(type)
    if depth is None or width is None:
        depth = 0.33
        width = 0.50
    return depth, width
