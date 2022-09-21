#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/9/1 20:27
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import logging
import random
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)
# voc_classes = ["defect"]
voc_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

num_classes = len(voc_classes)

train_val_test_percent = 1
train_val_percent = 0.9

file_dir = "NEUDET"

voc_img_path = f"../datasets/{file_dir}/JPEGImages"
voc_annotations_path = f"../datasets/{file_dir}/Annotations"
voc_image_sets_path = f"../datasets/{file_dir}"


def read_all_annotations():
    # 标注信息写入到txt文件中
    with open(f"{voc_image_sets_path}/all.txt", "w", encoding="utf-8") as list_file:
        for img in os.listdir(voc_img_path):
            file_name, file_extend = os.path.splitext(img)
            list_file.write(img)
            with open(f"{voc_annotations_path}/{file_name}.xml", encoding="utf-8") as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()
                for obj in root.iter("object"):
                    difficult = 0
                    if obj.find("difficult") is not None:
                        difficult = obj.find("difficult").text
                    cls = obj.find("name").text
                    if cls not in voc_classes or int(difficult) == 1:
                        continue
                    cls_id = voc_classes.index(cls)
                    # 获取到标注框坐标
                    xml_box = obj.find("bndbox")
                    x1 = int(float(xml_box.find("xmin").text))
                    y1 = int(float(xml_box.find("ymin").text))
                    x2 = int(float(xml_box.find("xmax").text))
                    y2 = int(float(xml_box.find("ymax").text))
                    position = (x1, y1, x2, y2)
                    list_file.write(" " + ",".join([str(a) for a in position]) + "," + str(cls_id))
                list_file.write("\n")
    list_file.close()


def write_train_val_test_annotations():
    random.seed(0)
    with open(f"../datasets/{file_dir}/all.txt") as anno_r:
        anno_lines = anno_r.readlines()
    anno_r.close()
    num_annotations = len(anno_lines)
    anno_list = range(num_annotations)
    # 训练集+验证集与测试集比例
    num_train_val_test = int(num_annotations * train_val_test_percent)
    # 训练集与验证集比例
    num_train_val = int(num_train_val_test * train_val_percent)

    train_val_data = random.sample(anno_list, num_train_val_test)
    train_data = random.sample(train_val_data, num_train_val)

    logger.info(f"Train size is {num_train_val}, val size is {num_train_val_test - num_train_val}")
    f_train_val = open(f"{voc_image_sets_path}/trainval.txt", "w")
    f_test = open(f"{voc_image_sets_path}/test.txt", "w")
    f_train = open(f"{voc_image_sets_path}/train.txt", "w")
    f_val = open(f"{voc_image_sets_path}/val.txt", "w")
    f_classes = open(f"{voc_image_sets_path}/classes.txt", "w")

    for i in anno_list:
        name = anno_lines[i]
        if i in train_val_data:
            f_train_val.write(name)
            if i in train_data:
                f_train.write(name)
            else:
                f_val.write(name)
        else:
            f_test.write(name)

    for cls in voc_classes:
        f_classes.write(f"{cls}\n")

    f_train_val.close()
    f_train.close()
    f_val.close()
    f_test.close()
    f_classes.close()


def create_annotations():
    read_all_annotations()
    write_train_val_test_annotations()


if __name__ == "__main__":
    create_annotations()
