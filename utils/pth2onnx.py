#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/13 9:28
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import torch
from network.net import Net
from utils.util import get_root_path

root = get_root_path()

result_name = "1657616363.2346666"
model_name = "model_100"
model_root = os.path.join(root,f"results/{result_name}")
model_path = f"{model_root}/{model_name}.pth"
onnx_path = f"{model_root}/pth2onnx"
if not os.path.exists(onnx_path):
    os.mkdir(onnx_path)

model = Net()
model_load = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(model_load)
model.cuda()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 1, 32, 64, device=device)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, f"{onnx_path}/{model_name}.onnx", opset_version=11, verbose=False,
                  input_names=input_names, output_names=output_names)
