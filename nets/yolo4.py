# -*- coding: utf-8 -*-
"""
yolo4: 在特征金字塔部分，YOLOV4 结合了两种改进：
1. 使用了 SPP 结构
2. 使用了 PANet 结构: 实例分割算法，具体结构主要为了反复提升特征的作用（特征的反复提取）
@author: libo
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from nets.CSPdarknet import darknet53

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    """ SPP 结构，利用不同大小的池化核进行池化 池化后堆叠 (最大池化)
    SPP 结构掺杂在对 CSPDarknet53 的最后一个特征层的卷积里，在对对CSPdarknet53的最后一个特征层进行三次DarknetConv2D_BN_Leaky卷积后，
    分别利用四个不同尺度的最大池化进行处理，最大池化的池化核大小分别为 13x13、9x9、5x5、1x1（1x1即无处理）
    作用：能够极大的增加感受野，分离出最显著的上下文特征
    """
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    """ 卷积 + 上采样 """
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


def make_three_conv(filters_list, in_filters):
    """ 三次卷积块 """
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    """ 五次卷积块 """
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    """ 最后获得 yolov4 的输出 """
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    """ yolo_body 此部分主要实现的就是 PANet，对于特征层的反复提取使用
    yolohead 利用获得到的特征进行预测
    """
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        """ YOLO Head 
        1. 在特征利用部分，YoloV4提取多特征层进行目标检测，一共提取三个特征层，分别位于中间层，中下层，底层，
           三个特征层的 shape 分别为(76,76,256)、(38,38,512)、(19,19,1024)
           
        2. 输出层的 shape 分别为(19,19,75)，(38,38,75)，(76,76,75)，最后一个维度为 75 是因为该图是基于voc数据集的，它的类为20种，
           YoloV4 只有针对每一个特征层存在 3 个先验框，所以最后维度为 3x25；
           如果使用的是coco训练集，类则为80种，最后的维度应该为255 = 3x85，三个特征层的shape为(19,19,255)，(38,38,255)，(76,76,255)
        """
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)

        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 5+num_classes = 4+1+num_classes
        final_out_filter1 = num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1], 256)

        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 = num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0], 512)

    def forward(self, x):
        """ P3, P4, P5 分别为 CSPDarkNet53 中的过程结果 """
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], dim=1)        # axis=1
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], dim=1)        # axis=1
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)      # axis=1
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)     # axis=1
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

