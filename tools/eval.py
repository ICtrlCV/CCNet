#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/8 9:04
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import logging
import argparse

import coloredlogs
import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm

from network.net import Net
from utils.datasets import Datasets, dataset_collate
from utils.bbox import decode_outputs, post_process
from utils.initialization import device_initializer, model_initializer
from utils.util import get_val_lines, get_classes, load_model_weights, get_root_path, get_gt_dir
from utils.map import get_map, get_coco_map

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
root = get_root_path()


def check_detection_file(val_lines, save_box_path):
    # bug fix
    # 检查是否有没有生成的文件，预测时可能有些文件的框预测不出来，所以需要手动添加文件
    for val_line in val_lines:
        line_content = val_line.split()
        line_name, line_extend = os.path.splitext(line_content[0])
        if not os.path.exists(f"{save_box_path}/{line_name}.txt"):
            with open(f"{save_box_path}/{line_name}.txt", "w") as f:
                f.write("0 0 0 0 0 0\n")
            f.close()


def evaluate(model, val_dataloader, input_shape, num_classes, device, save_box_path, class_names,
             conf_thres=0.001, nms_thres=0.6):
    """
        评估方法
    Args:
        model: 模型
        val_dataloader: 验证集
        val_lines: 验证集列表
        input_shape: 输入尺寸
        num_classes: 类别个数
        device: 设备
        save_box_path: 保存box路径
        class_names: 类别名称
        conf_thres: 置信度阈值
        nms_thres: 非极大值抑制阈值

    Returns:

    """
    model.eval()
    for current_iter, batch in enumerate(tqdm(val_dataloader)):
        image, _, image_info = batch

        with torch.no_grad():
            # 计算eval模型mAP
            image = torch.Tensor(image).type(torch.FloatTensor).to(device)

            outputs = model(image)
            # 算出所有框的box中心点坐标 + 宽高
            outputs = decode_outputs(outputs, input_shape)
            # 处理结果
            outputs = post_process(outputs, num_classes, input_shape, image_info, conf_thres=conf_thres,
                                   nms_thres=nms_thres)
        for i, output in enumerate(outputs):
            image_name = image_info[i][0]
            file_name, file_extend = os.path.splitext(image_name)
            with open(f"{save_box_path}/{file_name}.txt", "w") as f:
                if output is None:
                    logger.warning("No image result.")
                    return

                output_label = np.array(output[:, 6], dtype="int32")
                output_conf = output[:, 4] * output[:, 5]
                output_boxes = output[:, :4]
                for i, c in list(enumerate(output_label)):
                    predicted_class = class_names[int(c)]
                    box = output_boxes[i]
                    conf = str(output_conf[i])

                    top, left, bottom, right = box

                    if predicted_class not in class_names:
                        continue
                    f.write(
                        f"{predicted_class} {conf[:6]} {str(int(left))} {str(int(top))} {str(int(right))} {str(int(bottom))}\n"
                    )
            f.close()


def evaluate_one_img(model, val_lines, input_shape, num_classes, device, data_path, save_box_path, class_names,
                     conf_thres=0.001, nms_thres=0.6):
    """
        单张图片评估
    Args:
        model: 模型
        val_lines: 验证集列表
        input_shape: 输入尺寸
        num_classes: 类别个数
        device: 设备
        data_path: 数据集路径
        save_box_path: 保存box路径
        class_names: 类别名称
        conf_thres: 置信度阈值
        nms_thres: 非极大值抑制阈值

    Returns:

    """
    model.eval()
    for line in val_lines:
        line = line.split()
        image = Image.open(f"{data_path}/JPEGImages/{line[0]}")
        line_name, line_extend = os.path.splitext(line[0])
        iw, ih = image.size
        with open(f"{save_box_path}/{line_name}.txt", "w") as f:
            image_shape = [[0, iw, ih]]
            image = image.convert("RGB")
            image_data = image.resize((input_shape[1], input_shape[0]), Image.BICUBIC)
            image_data = np.array(image_data, dtype="float32")
            image_data /= 255.0
            image_data -= np.array([0.485, 0.456, 0.406])
            image_data /= np.array([0.229, 0.224, 0.225])
            image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)

            with torch.no_grad():
                images = torch.from_numpy(image_data)
                images = images.to(device)
                outputs = model(images)
                outputs = decode_outputs(outputs, input_shape)
                results = post_process(outputs, num_classes, input_shape, image_shape, conf_thres=conf_thres,
                                       nms_thres=nms_thres)

                if results[0] is None:
                    logger.warning("No image result.")
                    return

                top_label = np.array(results[0][:, 6], dtype="int32")
                top_conf = results[0][:, 4] * results[0][:, 5]
                top_boxes = results[0][:, :4]

            for i, c in list(enumerate(top_label)):
                predicted_class = class_names[int(c)]
                box = top_boxes[i]
                score = str(top_conf[i])

                top, left, bottom, right = box

                if predicted_class not in class_names:
                    continue
                f.write(
                    f"{predicted_class} {score[:6]} {str(int(left))} {str(int(top))} {str(int(right))} {str(int(bottom))}\n"
                )
        f.close()


def compute_mAP(eval_type, class_names, conf_thres=0.5, save_path=f"{root}results"):
    """
        计算mAP
    Args:
        eval_type: 评估类型 voc或者coco
        class_names: 类别名称
        conf_thres: 置信度阈值
        save_path: 保存临时文件路径

    Returns: mAP50 mAP5095

    """
    if eval_type == "voc":
        min_overlaps = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        voc_results = []
        mAP_sum = 0
        for min_overlap in min_overlaps:
            voc_result = get_map(min_overlap=min_overlap, draw_plot=False, score_threhold=conf_thres, path=save_path)
            voc_results.append(voc_result)
            mAP_sum += voc_result
            print(f"mAP@{min_overlap} is {voc_result}")
        mAP50 = voc_results[0]
        mAP5095 = mAP_sum / len(voc_results)
        print(f"VOC mAP@0.5 is {mAP50}")
        print(f"VOC mAP@0.5:0.95 is {mAP5095}")
    elif eval_type == "coco":
        coco_results = get_coco_map(class_names, save_path)
        mAP5095 = coco_results[0]
        mAP50 = coco_results[1]
        print(f"COCO mAP@0.5 is {mAP50}")
        print(f"COCO mAP@0.5:0.95 is {mAP5095}")
    else:
        mAP50 = 0
        mAP5095 = 0
    return mAP50, mAP5095


def main(arg_list):
    """
        主函数
    Args:
        arg_list: 参数列表
    """
    model_type = arg_list.model_type
    depth, width = model_initializer(model_type)
    device = device_initializer()
    # 批处理大小
    batch_size = arg_list.batch_size
    num_workers = arg_list.num_workers
    # 激活函数
    act = arg_list.act
    input_shape = arg_list.input_shape
    # 数据集路径
    data_path = arg_list.data_path
    eval_type = arg_list.eval_type

    # 读取标签参数
    # 格式：文件名 x1,y1,x2,y2
    val_lines = get_val_lines(f"{data_path}/val.txt")
    # 种类数，种类
    class_names, num_classes = get_classes(f"{data_path}/classes.txt")

    # 评估数据集加载
    val_datasets = Datasets(annotation=val_lines, input_shape=input_shape, num_classes=num_classes, train=False,
                            path=data_path, mosaic=False)
    val_dataloader = DataLoader(dataset=val_datasets, batch_size=batch_size, shuffle=True, drop_last=False,
                                num_workers=num_workers, pin_memory=True, collate_fn=dataset_collate)
    # 初始化模型
    model = Net(depth=depth, width=width, num_classes=num_classes, act=act)
    # 所有结果文件夹的根目录
    result_path = arg_list.result_path
    # 模型所在结果文件夹的根目录
    load_model_dir = arg_list.load_model_dir
    dir_path = load_model_dir

    # 日志保存路径
    save_path = f"{result_path}/{dir_path}"
    logs_path = f"{save_path}/tensorboard"
    save_box_path = f"{save_path}/detection_results"
    save_gt_path = f"{save_path}/ground_truth"
    anno_path = f"{data_path}/Annotations"
    if not os.path.exists(save_box_path):
        os.makedirs(save_box_path)

    # 可视化训练过程
    tb_logger = SummaryWriter(log_dir=logs_path)
    # 生成真实框文件夹
    get_gt_dir(save_gt_path, val_lines, anno_path, class_names)
    check_detection_file(val_lines, save_box_path)

    if arg_list.mode == 0:
        # 模型名
        load_model_name = arg_list.load_model_name
        # 模型权重文件
        model_path = f"{save_path}/{load_model_name}.pth"
        logger.info(f"Model is using {model_path} weights.")
        model, _ = load_model_weights(model, model_path, device)
        model.to(device)
        evaluate(model, val_dataloader, input_shape, num_classes, device, save_box_path, class_names)
        # evaluate_one_img(model, val_lines, input_shape, num_classes, device, data_path, save_box_path, class_names)
        compute_mAP(eval_type=eval_type, class_names=class_names, save_path=save_path)
    elif arg_list.mode == 1:
        count = 1
        for weight in os.listdir(save_path):
            is_pth = weight.endswith((".pth", ".pt"))
            logger.info(f"Model is using {weight} weights.")
            if is_pth:
                load_model_name = weight
                file_name, file_extend = os.path.splitext(weight)
                # 模型权重文件
                model_path = f"{result_path}/{load_model_name}{file_extend}"
                model, _ = load_model_weights(model, model_path, device)
                model.to(device)
                evaluate(model, val_dataloader, input_shape, num_classes, device, save_box_path, class_names)
                mAP50, mAP5095 = compute_mAP(eval_type=eval_type, class_names=class_names, save_path=save_path)
                tb_logger.add_scalar("mAP50", mAP50, global_step=count)
                tb_logger.add_scalar("mAP5095", mAP5095, global_step=count)
                count += 1
    else:
        logger.warning("Please input the correct mode params. Use '--mode0' or '--mode 1'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型的类别s, m, l, x
    parser.add_argument("--model_type", type=str, default="s")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--conf_thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="NMS IoU threshold")
    # 输入尺寸大小
    parser.add_argument("--input_shape", type=list, default=[224, 224])
    # 数据集路径
    parser.add_argument("--data_path", type=str, default=f"{root}datasets/NEUDET")

    # 验证模式，0单独验证（需要写入模型名称和所在文件夹），1批量验证（需要写入模型文件夹）
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--result_path", type=str, default=f"{root}results")
    # 模型所在文件夹
    parser.add_argument("--load_model_dir", type=str, default="1663817017.9118543")
    # 模型名称
    parser.add_argument("--load_model_name", type=str, default="model_best")

    # 激活函数 relu, relu6, silu, lrelu
    parser.add_argument("--act", type=str, default="silu")
    # box, seg, mask, keypoint
    parser.add_argument("--eval_class", type=str, default="box")
    # voc, coco
    parser.add_argument("--eval_type", type=str, default="coco")
    args = parser.parse_args()
    logger.info(f"当前参数：{args}")
    main(args)
