#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/13 9:28
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import argparse
import logging
import os

import coloredlogs
import torch
import torch.onnx

from network.net import Net
from utils.initialization import device_initializer, model_initializer
from utils.util import get_root_path, get_classes

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


@torch.no_grad()
def main(arg_lists):
    model_type = arg_lists.model_type
    class_path = arg_lists.class_path
    model_path = arg_lists.model_path
    model_name = arg_lists.model_name
    onnx_path = os.path.join(model_path, "pth2onnx")
    batch_size = arg_lists.batch_size
    test_size = arg_lists.test_size

    device = device_initializer()
    class_names, num_classes = get_classes(class_path)
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)

    depth, width = model_initializer(model_type)
    model = Net(depth=depth, width=width, num_classes=num_classes)
    model_param = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_param["model"])
    logger.info("Loaded the weights finish.")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(batch_size, 1, test_size[0], test_size[1], requires_grad=True, device=device)
    torch.onnx.export(model, dummy_input, f"{onnx_path}/{model_name}.onnx", opset_version=11, verbose=False,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    logger.info("Convert onnx model finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="s")
    parser.add_argument("--class_path", type=str, default=f"{root}datasets/NEUDET/classes.txt")
    parser.add_argument("--model_path", type=str, default=f"{root}results/test")
    parser.add_argument("--model_name", type=str, default="model_best")
    parser.add_argument("--save_path", type=str, default=f"{root}demo/onnx")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=list, default=[640, 640])
    parser.add_argument("--workspace", type=int, default=32)
    args = parser.parse_args()
    main(args)
