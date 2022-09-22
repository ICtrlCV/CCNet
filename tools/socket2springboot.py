#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/6 16:17
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import logging
import socket
import threading
import json

import coloredlogs
import cv2
import numpy as np
import torch

from network.net import Net
# from network.net_old import Net
from utils.bbox import decode_outputs, post_process
from utils.initialization import device_initializer
from utils.util import get_classes, load_model_weights, get_root_path

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


def inference(model_name, model_path, dataset, input_shape, conf_thres, nms_thres, image_paths):
    if model_name == "Net":
        msg_dict = {"image": [], "annotations": []}
        model_path = model_path
        class_names, num_classes = get_classes(os.path.join(root, f"datasets/{dataset}/classes.txt"))
        device = device_initializer()
        if device == "cpu":
            # 如果服务器只有cpu，限制1个核跑
            torch.set_num_threads(1)

        # 需要切换到老网络，直接修改import，将net改为net_old
        model = Net(depth=0.33, width=0.5, num_classes=num_classes, act="silu")
        model, _ = load_model_weights(model, model_path, device)
        model.to(device)
        model.eval()
        for image_id, image_path in enumerate(image_paths):
            # 读取图片
            image = cv2.imread(image_path)
            # 写入json
            image_dict = {"image_path": image_path, "image_id": image_id}
            msg_dict["image"].append(image_dict)
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
            top_label = np.array(results[0][:, 6], dtype="int32")
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            for i, c in list(enumerate(top_label)):
                predicted_class = class_names[int(c)]
                box = top_boxes[i]
                conf = top_conf[i]
                # [y0,x0,y1,x1]
                top, left, bottom, right = box
                ann_dict = {"image_id": image_id, "box": [int(left), int(top), int(right), int(bottom)],
                            "predicted_class": predicted_class,
                            "conf": conf}
                msg_dict["annotations"].append(ann_dict)
        return_set = msg_dict
    else:
        return_set = {}
    return return_set


def main():
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 12345
    # 将套接字与本地主机和端口绑定
    server_socket.bind((host, port))
    # 设置监听最大连接数
    server_socket.listen(5)
    # 获取本地服务器的连接信息
    local_server_address = server_socket.getsockname()
    logger.info(f"服务器地址:{str(local_server_address)}")
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        client_socket, address = server_socket.accept()
        logger.info(f"连接地址:{str(address)}")
        try:
            # 为每一个请求开启一个处理线程
            t = ServerThreading(client_socket)
            t.start()
            pass
        except Exception as identifier:
            logger.error(identifier)
            break
    server_socket.close()


class ServerThreading(threading.Thread):
    def __init__(self, client_socket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = client_socket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        logger.info("开启线程.....")
        try:
            # 接受数据
            msg = ""
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith("over"):
                    msg = msg[:-4]
                    break
            # 解析json格式的数据
            re = json.loads(msg)
            # 提取传入的json值信息
            model_name = re["model_name"]
            model_path = re["model_path"]
            dataset = re["dataset"]
            input_shape = re["input_shape"]
            conf_thres = re["conf_thres"]
            nms_thres = re["nms_thres"]
            image_paths = []
            image_path = re["image_path"]
            for i in range(len(image_path)):
                image_paths.append(image_path[i])
            # 调用模型选择器处理请求并获取模型返回值
            res = inference(model_name, model_path, dataset, input_shape, conf_thres, nms_thres, image_paths)
            send_msg = res
            logger.info(f"当前返回值为:{send_msg}")
            # 发送数据
            self._socket.send(f"{send_msg}".encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            logger.error(identifier)
            pass
        finally:
            self._socket.close()
        logger.info("任务结束.....")

        pass

    def __del__(self):
        pass


if __name__ == "__main__":
    main()
