#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2022/7/5 11:36
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, num_classes, strides=None, loss_type="siou"):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]
        self.num_classes = num_classes
        self.strides = strides

        self.bce_with_log_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOULoss(reduction="none", loss_type=loss_type)
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, inputs, labels=None):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):

        # 将特征图切分成bbox,obj,cls
        # [batch, n_anchors_all, 4]，每个特征点回归参数，调整后获得预测框
        bbox_preds = outputs[:, :, :4]

        # [batch, n_anchors_all, 1]，是否包含物体
        obj_preds = outputs[:, :, 4:5]

        # [batch, n_anchors_all, n_cls]，6位往后是物品所有类别
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]

        # 每个anchor的中心点相较于输出尺寸x坐标
        # [1, n_anchors_all]
        x_shifts = torch.cat(x_shifts, 1)

        # 每个anchor的中心点相较于输出尺寸y坐标
        # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)

        # 每个anchor相较于输入尺寸减小的strides
        # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0

        # 一张图片一张图片计算
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # [num_gt, num_classes]，每张图片的真实框数量和物体类别个数
                gt_bboxes_per_image = labels[batch_idx][..., :4]

                # [num_gt]
                gt_classes = labels[batch_idx][..., 4]

                # [n_anchors_all, 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                # [n_anchors_all, num_classes]
                cls_preds_per_image = cls_preds[batch_idx]

                # [n_anchors_all, 1]
                obj_preds_per_image = obj_preds[batch_idx]

                # 标签分配
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bce_with_log_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (self.bce_with_log_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                        cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):

        #   fg_mask                 [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        #   pair_wise_ious      [num_gt, fg_mask]
        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        # 源码
        # cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(
        #    num_gt, 1, 1).sigmoid_()
        # 进行优化后
        cls_preds_ = cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1) * obj_preds_.sigmoid_().unsqueeze(
            0).repeat(num_gt, 1, 1)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1,
                                                                                                               num_in_boxes_anchor,
                                                                                                               1)
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)

        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)

        #   center_deltas   [num_gt, n_anchors_all, 4]
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        matching_matrix = torch.zeros_like(cost)

        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        # 源码
        # for gt_idx in range(num_gt):
        #
        #     # 给每个真实框选取最小的动态k个点
        #     _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
        # 优化后
        ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # anchor_matching_gt  [fg_mask]
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # 当某一个特征点指向多个真实框的时候
            # 选取cost最小的真实框。
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        # fg_mask_inboxes  [fg_mask]
        # num_fg为正样本的特征点个数
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        # 对fg_mask进行更新
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # 获得特征点对应的物品种类
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


class IOULoss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou"):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        global loss
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "diou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2
            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2
            # 求diou
            diou = iou - d / c
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2
            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2
            # 求diou

            diou = iou - d / c

            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w = pred[:, 2]
            h = pred[:, 3]

            with torch.no_grad():
                arctan = torch.atan(w_gt / h_gt) - torch.atan(w / h)
                v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
                s = 1 - iou
                alpha = v / (s + v)

            ciou = diou - alpha * v
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "siou":
            """
                paper: SIoU Loss: More Powerful Learning for Bounding Box Regression 
                https://arxiv.org/pdf/2205.12740.pdf
            """
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]

            s_w_c = target[:, 0] - pred[:, 0]
            s_h_c = target[:, 1] - pred[:, 1]

            sigma = torch.pow(s_w_c ** 2 + s_h_c ** 2, 0.5)
            # sinα
            sin_alpha = torch.abs(s_h_c) / sigma
            # sinβ
            sin_beta = torch.abs(s_w_c) / sigma

            thres = torch.pow(torch.tensor(2.), 0.5) / 2
            sin_alpha = torch.where(sin_alpha < thres, sin_alpha, sin_beta)
            angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi / 4), 2)

            # get distance cost
            gamma = angle_cost - 2
            rho_x = (s_w_c / w_c) ** 2
            rho_y = (s_h_c / h_c) ** 2
            delta_x = 1 - torch.exp(gamma * rho_x)
            delta_y = 1 - torch.exp(gamma * rho_y)
            distance_cost = delta_x + delta_y

            # get shape cost
            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w_pred = pred[:, 2]
            h_pred = pred[:, 3]
            w_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
            w_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)
            theta = 4
            shape_cost = torch.pow((1 - torch.exp(-1 * w_w)), theta) + torch.pow((1 - torch.exp(-1 * w_h)), theta)
            siou = iou - (distance_cost + shape_cost) * 0.5
            loss = 1 - siou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
