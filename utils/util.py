#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/7 14:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import logging
import os

import coloredlogs
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET

from collections import OrderedDict

from tqdm import tqdm


def get_classes(classes_path):
    # 打开类的路径并每次读取一行
    with open(classes_path, encoding="utf-8") as c_f:
        class_names = c_f.readlines()
    c_f.close()
    # 获取类名并移除换行符
    class_names = [c.strip() for c in class_names]
    class_len = len(class_names)
    # 返回类名和多少个类
    return class_names, class_len


def get_train_lines(train_path):
    with open(train_path, encoding="utf-8") as t_f:
        train_lines = t_f.readlines()
    t_f.close()
    train_lines = [c.strip() for c in train_lines]
    return train_lines


def get_val_lines(val_path):
    with open(val_path, encoding="utf-8") as v_f:
        val_lines = v_f.readlines()
    v_f.close()
    val_lines = [c.strip() for c in val_lines]
    return val_lines


def get_test_lines(test_path):
    with open(test_path, encoding="utf-8") as t_f:
        test_lines = t_f.readlines()
    t_f.close()
    test_lines = [c.strip() for c in test_lines]
    return test_lines


def load_model_weights(model, model_path, device):
    model_dict = model.state_dict()
    weights = torch.load(model_path, map_location=device)
    weights_dict = weights["model"]
    best_ap5095 = weights["ap5095"]
    weights_dict = {k: v for k, v in weights_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(weights_dict)
    model.load_state_dict(OrderedDict(model_dict))
    return model, best_ap5095


def get_gt_dir(ground_truth_results_path, anno_lines, voc_annotations_path, class_names):
    if not os.path.exists(ground_truth_results_path):
        os.makedirs(ground_truth_results_path)
    for line in tqdm(anno_lines):
        line_content = line.split()
        # 获取图片原名
        file_name, file_extend = os.path.splitext(line_content[0])
        with open(f"{ground_truth_results_path}/{file_name}.txt", "w") as fw:
            root = ET.parse(f"{voc_annotations_path}/{file_name}.xml").getroot()
            for obj in root.findall("object"):
                difficult_flag = False
                if obj.find("difficult") is not None:
                    difficult = obj.find("difficult").text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find("name").text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find("bndbox")
                left = bndbox.find("xmin").text
                top = bndbox.find("ymin").text
                right = bndbox.find("xmax").text
                bottom = bndbox.find("ymax").text

                if difficult_flag:
                    fw.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                else:
                    fw.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        fw.close()


def get_color(i):
    color = [(0, 113, 188), (216, 82, 24), (236, 176, 31), (125, 46, 141), (118, 171, 47), (76, 189, 237),
             (161, 19, 46), (76, 76, 76), (153, 153, 153), (255, 0, 0), (255, 127, 0), (190, 190, 0), (0, 255, 0),
             (0, 0, 255), (170, 0, 255), (84, 84, 0), (84, 170, 0), (84, 255, 0), (170, 84, 0), (170, 170, 0),
             (170, 255, 0), (255, 84, 0), (255, 170, 0), (255, 255, 0), (0, 84, 127), (0, 170, 127), (0, 255, 127),
             (84, 0, 127), (84, 84, 127), (84, 170, 127), (84, 255, 127), (170, 0, 127), (170, 84, 127),
             (170, 170, 127), (170, 255, 127), (255, 0, 127), (255, 84, 127), (255, 170, 127), (255, 255, 127),
             (0, 84, 255), (0, 170, 255), (0, 255, 255), (84, 0, 255), (84, 84, 255), (84, 170, 255), (84, 255, 255),
             (170, 0, 255), (170, 84, 255), (170, 170, 255), (170, 255, 255), (255, 0, 255), (255, 84, 255),
             (255, 170, 255), (84, 0, 0), (127, 0, 0), (170, 0, 0), (212, 0, 0), (255, 0, 0), (0, 42, 0), (0, 84, 0),
             (0, 127, 0), (0, 170, 0), (0, 212, 0), (0, 255, 0), (0, 0, 42), (0, 0, 84), (0, 0, 127), (0, 0, 170),
             (0, 0, 212), (0, 0, 255), (0, 0, 0), (36, 36, 36), (72, 72, 72), (109, 109, 109), (145, 145, 145),
             (182, 182, 182), (218, 218, 218), (0, 113, 188), (80, 182, 188), (127, 127, 0)]
    return color[i]


def draw_rectangle(results, image, iw, ih, input_shape, class_names):
    # 对预测结果进行拆分
    top_label = np.array(results[0][:, 6], dtype="int32")
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]

    # 设置字体与边框厚度

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    # 图像绘制
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        conf = top_conf[i]

        top, left, bottom, right = box

        # 左上坐标
        top = max(0, np.floor(top).astype("int32"))
        left = max(0, np.floor(left).astype("int32"))
        # 右下左边
        bottom = min(ih, np.floor(bottom).astype("int32"))
        right = min(iw, np.floor(right).astype("int32"))

        color = get_color(int(c))
        text = '{}:{:.1f}%'.format(predicted_class, conf * 100)
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_color = (0, 0, 0) if np.mean(color) > 128 else (255, 255, 255)

        # 绘制预测框
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness=thickness + 1)
        rect_text_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        cv2.rectangle(
            image, (left, top + 1), (left + txt_size[0] + 1, top + int(1.5 * txt_size[1])),
            rect_text_color, -1
        )
        cv2.putText(image, text, (left, top + txt_size[1]), font, 0.4, txt_color, thickness=thickness)
    return image


def replace_path_str(path_str):
    path_str = path_str.replace("\\", "/")
    return path_str


def get_root_path(project_name="Net"):
    current_path = replace_path_str(os.path.abspath(os.path.dirname(__file__)))
    root = current_path[:current_path.find(f"{project_name}/") + len(f"{project_name}/")]
    return root
