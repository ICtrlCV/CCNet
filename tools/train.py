#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/4 17:45
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import time

import torch
import argparse
import logging
import coloredlogs
from tensorboardX import SummaryWriter

from tqdm import tqdm
from torch.utils.data import DataLoader

from network.net import Net
from network.loss import Loss
from eval import evaluate, get_gt_dir, compute_mAP
from utils.datasets import Datasets, dataset_collate
from utils.lr_scheduler import set_optimizer_lr
from utils.initialization import device_initializer, model_initializer
from utils.util import get_classes, get_train_lines, get_val_lines, load_model_weights

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def main(arg_list):
    model_type = arg_list.model_type
    depth, width = model_initializer(model_type)
    # 迭代次数
    epochs = arg_list.epochs
    # 批处理大小
    batch_size = arg_list.batch_size
    num_workers = arg_list.num_workers
    # 优化器类型
    optim = arg_list.optim
    # 激活函数
    act = arg_list.act
    # 评估模式
    eval_mode = arg_list.mode
    eval_interval = arg_list.eval_interval
    eval_type = arg_list.eval_type
    mosaic = arg_list.mosaic
    mosaic_epoch = arg_list.mosaic_epoch
    best_ap5095 = 0
    # 学习率函数
    lr_func = arg_list.lr_func
    # 检查我们使用的设备
    device = device_initializer()
    # sgd: 1e-2  adam, adamw: 1e-4
    if optim == "adam" or optim == "adamw":
        init_lr = 1e-4 / 8
        min_lr = init_lr * 0.01
    else:
        init_lr = 1e-2 / 8
        min_lr = init_lr * 0.01
    input_shape = arg_list.input_shape
    # 数据集路径
    data_path = arg_list.data_path

    # 读取标签参数
    # 格式：文件名 x1,y1,x2,y2
    train_lines = get_train_lines(f"{data_path}/train.txt")
    val_lines = get_val_lines(f"{data_path}/val.txt")
    # 种类数，种类
    class_names, num_classes = get_classes(f"{data_path}/classes.txt")
    loss_fn = Loss(num_classes=num_classes)
    # 初始化模型
    model = Net(depth=depth, width=width, num_classes=num_classes, act=act)
    logger.info("Successfully init model.")

    # 恢复训练
    if arg_list.resume:
        # 恢复训练迭代次数
        start_epoch = arg_list.start_epoch
        epochs_num = epochs - start_epoch + 1
        load_model_dir = arg_list.load_model_dir
        model_path = f"{arg_list.result_path}/{load_model_dir}/model_{str(start_epoch - 1).zfill(3)}.pth"
        dir_path = load_model_dir
        model, ap5095 = load_model_weights(model, model_path, device)
        best_ap5095 = ap5095
    else:
        start_epoch = 1
        epochs_num = epochs
        dir_path = str(time.time())
        # 如果存在预训练模型
        if arg_list.pretrain != "":
            model_path = arg_list.pretrain
            model = load_model_weights(model, model_path, device)
    # 将模型传输到设备
    model = model.to(device)

    # 日志保存路径
    logs_path = f"{arg_list.result_path}/{dir_path}/tensorboard"
    save_path = f"{arg_list.result_path}/{dir_path}"
    save_box_path = f"{save_path}/detection_results"
    save_gt_path = f"{save_path}/ground_truth"
    anno_path = f"{data_path}/Annotations"
    if not os.path.exists(save_box_path):
        os.makedirs(save_box_path)
    if eval_mode:
        get_gt_dir(save_gt_path, val_lines, anno_path, class_names)
    # 可视化训练过程
    tb_logger = SummaryWriter(log_dir=logs_path)
    # 训练器前加载模型
    analog_input = torch.rand(batch_size, 3, input_shape[0], input_shape[1]).to(device)
    tb_logger.add_graph(model.eval(), analog_input)

    # 优化器选择
    if optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, betas=(0.937, 0.999))
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, betas=(0.937, 0.999))
    elif optim == "sgd":
        # 动量不宜设置太高
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    else:
        raise

    # 预热或余弦退火学习率选择
    init_lr_func = True if lr_func == "warmcos" else False

    # 数据加载
    train_datasets = Datasets(annotation=train_lines, input_shape=input_shape, num_classes=num_classes, train=True,
                              path=data_path, mosaic=mosaic, epoch=epochs_num, mosaic_epoch=mosaic_epoch)
    val_datasets = Datasets(annotation=val_lines, input_shape=input_shape, num_classes=num_classes, train=False,
                            path=data_path, epoch=epochs_num, mosaic=False)
    # pin_memory锁页内存，适合大内存；shuffle打乱每个batch顺序；drop_last丢弃最后不满足的batch
    train_dataloader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers, pin_memory=True, collate_fn=dataset_collate)
    val_dataloader = DataLoader(dataset=val_datasets, batch_size=batch_size, shuffle=True, drop_last=False,
                                num_workers=num_workers, pin_memory=True, collate_fn=dataset_collate)

    train_max_iter = len(train_dataloader)

    # 训练开始时间
    t0 = time.time()
    # 训练模式
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        lr = set_optimizer_lr(optimizer=optimizer, current_epoch=epoch, max_epoch=epochs, lr_min=min_lr,
                              lr_max=init_lr, warmup=init_lr_func)
        loss_total = 0

        for current_iter, batch in enumerate(tqdm(train_dataloader)):
            image, label, _ = batch
            image = torch.Tensor(image).type(torch.FloatTensor).to(device)
            label = [torch.Tensor(lab).type(torch.FloatTensor).to(device) for lab in label]

            optimizer.zero_grad()

            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        mean_loss = loss_total / train_max_iter
        logger.info(f"Epoch: {epoch}/{epochs}, mean loss: {mean_loss},lr: {lr}")

        tb_logger.add_scalar("Loss", mean_loss, global_step=epoch)
        tb_logger.add_scalar("Lr", lr, global_step=epoch)

        # 如果是评估模式并且处于估间隔
        if eval_mode and (epoch % eval_interval == 0 or ((epochs - mosaic_epoch) < epoch <= epochs)):
            model.eval()
            evaluate(model, val_dataloader, val_lines, input_shape, num_classes, device, save_box_path, class_names)
            mAP50, mAP5095 = compute_mAP(eval_type=eval_type, class_names=class_names, save_path=save_path)
            tb_logger.add_scalar("mAP50", mAP50, global_step=epoch)
            tb_logger.add_scalar("mAP5095", mAP5095, global_step=epoch)
            logger.info(f"mAP@0.5: {mAP50}")
            logger.info(f"mAP@0.5:0.95: {mAP5095}")
        else:
            mAP50 = 0
            mAP5095 = 0
            logger.info("No evaluation, please use the eval method to evaluate after training.")
        # 保存模型
        save_model = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ap50": mAP50,
            "ap5095": mAP5095
        }
        torch.save(save_model, f"{save_path}/model_last.pth")
        if arg_list.save_model_interval:
            torch.save(save_model, f"{save_path}/model_{str(epoch).zfill(3)}.pth")
            logger.info(f"Successfully save the model_{str(epoch).zfill(3)}.")
        if best_ap5095 < mAP5095:
            best_ap5095 = mAP5095
            torch.save(save_model, f"{save_path}/model_best.pth")
            logger.info(f"Successfully save the model_best.")
        logger.info(f"Successfully save the model.")
    # 训练时间
    use_time = time.time() - t0
    logger.info(f"Training time is {use_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 初始化模型的类别并最终生成s, m, l, x
    parser.add_argument("--model_type", type=str, default="s")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    # 使用预训练权重
    parser.add_argument("--pretrain", type=str, default="")
    # cos, warmcos
    parser.add_argument("--lr_func", type=str, default="cos")
    # sgd, adam, adamw
    parser.add_argument("--optim", type=str, default="sgd")
    # 激活函数 relu, relu6, silu, lrelu
    parser.add_argument("--act", type=str, default="silu")
    # 输入尺寸大小
    parser.add_argument("--input_shape", type=list, default=[224, 224])
    # 数据集路径
    parser.add_argument("--data_path", type=str, default="../datasets/NEUDET")
    # 保存路径
    parser.add_argument("--result_path", type=str, default="../results")
    # 是否每次训练储存
    parser.add_argument("--save_model_interval", type=bool, default=True)
    # 打开评估模式
    parser.add_argument("--mode", type=bool, default=True)
    parser.add_argument("--eval_interval", type=int, default=1)
    # 打开Mosaic
    parser.add_argument("--mosaic", type=bool, default=True)
    parser.add_argument("--mosaic_epoch", type=int, default=15)
    # 采取评估方法voc, coco
    parser.add_argument("--eval_type", type=str, default="coco")
    # 训练异常中断：1.恢复训练将设置为“True” 2.设置异常中断的epoch编号 3.写入中断的epoch上一个加载模型的路径
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=-1)
    parser.add_argument("--load_model_dir", type=str, default="")

    args = parser.parse_args()
    logger.info(f"Input params: {args}")
    main(args)
