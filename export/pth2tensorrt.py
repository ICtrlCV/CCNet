#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/8 10:43
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import argparse
import logging
import os
import shutil

import coloredlogs
import torch
from torch2trt import torch2trt
import tensorrt as trt

from network.net import Net
from utils.initialization import model_initializer, device_initializer
from utils.util import get_classes, get_root_path

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


@torch.no_grad()
def main(arg_lists):
    """
        仅限于简单且TensorRT支持结构，暂且不支持SpaceToDepth和MobileViT结构
        若要转换当前结构，仅考虑torch -> onnx -> engine
    Args:
        arg_lists: 参数列表
    """
    model_type = arg_lists.model_type
    class_path = arg_lists.class_path
    model_path = arg_lists.model_path
    model_name = arg_lists.model_name
    trt_path = os.path.join(model_path, "pth2trt")
    save_path = arg_lists.save_path
    batch_size = arg_lists.batch_size
    test_size = arg_lists.test_size
    workspace = arg_lists.workspace
    if not os.path.exists(trt_path):
        os.makedirs(trt_path)

    depth, width = model_initializer(model_type)
    class_names, num_classes = get_classes(class_path)
    device = device_initializer()

    model = Net(depth=depth, width=width, num_classes=num_classes)
    model_param = torch.load(os.path.join(model_path, f"{model_name}.pth"), map_location="cpu")
    model.load_state_dict(model_param["model"])
    logger.info("Loaded the weights finish.")
    model.eval()
    model.to(device)
    x = torch.ones(1, 3, test_size[0], test_size[1], device=device)
    model_trt = torch2trt(model, [x], fp16_mode=False, log_level=trt.Logger.INFO,
                          max_workspace_size=(1 << workspace), max_batch_size=batch_size)
    torch.save(model_trt.state_dict(), os.path.join(save_path, f"{model_name}_trt.pth"))
    logger.info("Convert tensorrt model finish")
    engine_path = os.path.join(save_path, f"{model_name}_trt.engine")
    engine_path_copy = os.path.join(trt_path, f"{model_name}_trt.engine")
    with open(engine_path) as f:
        f.write(model_trt.engine.serialize())
    shutil.copyfile(engine_path, engine_path_copy)
    logger.info("Convert tensorrt model engine file is saved for C++.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="s")
    parser.add_argument("--class_path", type=str, default=f"{root}datasets/NEUDET/classes.txt")
    parser.add_argument("--model_path", type=str, default=f"{root}results/test")
    parser.add_argument("--model_name", type=str, default="model_best")
    parser.add_argument("--save_path", type=str, default=f"{root}demo/tensorrt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=list, default=[224, 224])
    parser.add_argument("--workspace", type=int, default=32)
    args = parser.parse_args()
    main(args)
