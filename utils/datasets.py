#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/6 10:23
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import numpy as np
from torch.utils.data import Dataset
from random import sample, shuffle
from PIL import Image
import cv2


class Datasets(Dataset):
    def __init__(self, annotation, input_shape, num_classes, train, path, epoch=300, mosaic=False, mosaic_epoch=15):
        self.annotation = annotation
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(self.annotation)
        self.path = path
        self.epoch = epoch
        self.epoch_now = -1
        self.mosaic = mosaic
        self.mosaic_epoch = mosaic_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.epoch_now += 1
        index = index % self.length

        # Mosaic模式
        if self.mosaic:
            if self.rand() < 0.5 and self.epoch_now < self.epoch - self.mosaic_epoch:
                annotation_lines = sample(self.annotation, 3)
                annotation_lines.append(self.annotation[index])
                shuffle(annotation_lines)
                image, label, image_info = self.get_random_data_with_Mosaic(annotation_lines, self.input_shape)
            else:
                image, label, image_info = self.get_random_data(self.annotation[index], self.input_shape,
                                                                random=self.train)
        # 一般模式
        else:
            image, label, image_info = self.get_random_data(self.annotation[index], self.input_shape, random=self.train)
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        label = np.array(label, dtype=np.float32)
        if len(label) != 0:
            label[:, 2:4] = label[:, 2:4] - label[:, 0:2]
            label[:, 0:2] = label[:, 0:2] + label[:, 2:4] / 2
        return image, label, image_info

    def rand(self, a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        line = annotation.split()

        # 读取图像并转换成RGB图像
        image = Image.open(f"{self.path}/JPEGImages/{line[0]}")
        # 获得图像的高宽与目标高宽
        iw, ih = image.size
        h, w = input_shape
        image_info = (line[0], iw, ih)

        # 获得预测框
        label = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 将图像多余的部分加上灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = np.array(new_image, np.float32)

            # 对真实框进行调整
            if len(label) > 0:
                np.random.shuffle(label)
                label[:, [0, 2]] = label[:, [0, 2]] * nw / iw + dx
                label[:, [1, 3]] = label[:, [1, 3]] * nh / ih + dy
                label[:, 0:2][label[:, 0:2] < 0] = 0
                label[:, 2][label[:, 2] > w] = w
                label[:, 3][label[:, 3] > h] = h
                label_w = label[:, 2] - label[:, 0]
                label_h = label[:, 3] - label[:, 1]
                # 丢弃无效框
                label = label[np.logical_and(label_w > 1, label_h > 1)]
            return image, label, image_info

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域扭曲
        image = np.array(image, np.uint8)
        # 对图像进行色域变换
        # 计算色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(label) > 0:
            np.random.shuffle(label)
            label[:, [0, 2]] = label[:, [0, 2]] * nw / iw + dx
            label[:, [1, 3]] = label[:, [1, 3]] * nh / ih + dy
            if flip:
                label[:, [0, 2]] = w - label[:, [2, 0]]
            label[:, 0:2][label[:, 0:2] < 0] = 0
            label[:, 2][label[:, 2] > w] = w
            label[:, 3][label[:, 3] > h] = h
            label_w = label[:, 2] - label[:, 0]
            label_h = label[:, 3] - label[:, 1]
            label = label[np.logical_and(label_w > 1, label_h > 1)]
        return image, label, image_info

    def get_random_data_with_Mosaic(self, annotation, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        label_datas = []
        image_info_datas = []
        index = 0
        for line in annotation:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(f"{self.path}/JPEGImages/{line_content[0]}")

            # 图片的大小
            iw, ih = image.size
            image_info = (line_content[0], iw, ih)
            # 保存框的位置
            label = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(label) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label[:, [0, 2]] = iw - label[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = 0
            dy = 0
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = np.array(new_image)

            index = index + 1
            label_data = []

            # 对label进行重新处理
            if len(label) > 0:
                np.random.shuffle(label)
                label[:, [0, 2]] = label[:, [0, 2]] * nw / iw + dx
                label[:, [1, 3]] = label[:, [1, 3]] * nh / ih + dy
                label[:, 0:2][label[:, 0:2] < 0] = 0
                label[:, 2][label[:, 2] > w] = w
                label[:, 3][label[:, 3] > h] = h
                label_w = label[:, 2] - label[:, 0]
                label_h = label[:, 3] - label[:, 1]
                label = label[np.logical_and(label_w > 1, label_h > 1)]
                label_data = np.zeros((len(label), 5))
                label_data[:len(label)] = label

            image_datas.append(image)
            label_datas.append(label_data)
            image_info_datas.append(image_info)

        # 将图片分割，放在一起
        cut_x = int(w * min_offset_x)
        cut_y = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cut_y, :cut_x, :] = image_datas[0][:cut_y, :cut_x, :]
        new_image[cut_y:, :cut_x, :] = image_datas[1][cut_y:, :cut_x, :]
        new_image[cut_y:, cut_x:, :] = image_datas[2][cut_y:, cut_x:, :]
        new_image[:cut_y, cut_x:, :] = image_datas[3][:cut_y, cut_x:, :]

        # 进行色域变换
        image = np.array(new_image, np.uint8)
        # 对图像进行色域变换
        # 计算色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # 对框进行进一步的处理
        merge_label = []
        for i in range(len(label_datas)):
            for label in label_datas[i]:
                tmp_label = []
                x1, y1, x2, y2 = label[0], label[1], label[2], label[3]

                if i == 0:
                    if y1 > cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x

                if i == 1:
                    if y2 < cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x

                if i == 2:
                    if y2 < cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x

                if i == 3:
                    if y1 > cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x
                tmp_label.append(x1)
                tmp_label.append(y1)
                tmp_label.append(x2)
                tmp_label.append(y2)
                tmp_label.append(label[-1])
                merge_label.append(tmp_label)
        label = merge_label
        image_info = image_info_datas

        return image, label, image_info


def dataset_collate(batch):
    images = []
    labels = []
    image_info = []
    for image, label, info in batch:
        images.append(image)
        labels.append(label)
        image_info.append(info)
    images = np.array(images)
    return images, labels, image_info
