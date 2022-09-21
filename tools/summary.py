#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/4 17:45
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from torchsummary import summary
from network.net import Net as Net

# from network.net_old import Net as Net

if __name__ == "__main__":
    model = Net(depth=0.33, width=0.5, num_classes=1, act="silu").train().cuda()
    print(model)
    print("==========================================================================================")
    summary(model, (3, 640, 640))
