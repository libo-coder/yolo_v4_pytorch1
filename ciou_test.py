# -*- coding: utf-8 -*-
""" CIOU:
IoU 是比值的概念，对目标物体的scale是不敏感的。然而常用的 BBox 的回归损失优化和 IoU 优化不是完全等价的，寻常的IoU无法直接优化没有重叠的部分。
于是有人提出直接使用 IOU 作为回归优化 loss，CIOU 是其中非常优秀的一种想法。
@author: libo
"""
import torch
import math
import numpy as np

def box_ciou(b1, b2):
    """
    :param b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :param b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :return ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / (union_area + 1e-7)

    # 计算中心的差距，分别代表了预测矿和真实框的中心点的欧式距离
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离，代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + 1e-7)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / b1_wh[..., 1]) - torch.atan(b2_wh[..., 0] / b2_wh[..., 1])), 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou


box1 = torch.from_numpy(np.array([[25, 25, 40, 40]])).type(torch.FloatTensor)
box2 = torch.from_numpy(np.array([[25, 25, 30, 40]])).type(torch.FloatTensor)
print(box_ciou(box1, box2))