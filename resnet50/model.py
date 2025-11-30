#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(CBM, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)


    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = self.act(x2)
        return self.pooling(x3)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_stride=0, downsample_mode=0):
        super(ResBlock, self).__init__()
        mid_channels = out_channels//4
        self.conv1 = nn.Conv2d(in_channels=in_channels,  out_channels=mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.downsample = None
        if downsample_stride != 0:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=downsample_stride, padding=0, dilation=1, bias=True)
        if downsample_mode == 1:
            self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1, dilation=1, bias=True)
        return

    def forward(self, x):
        if self.downsample is not None:
            x1 = self.downsample(x)
        else:
            x1 = x
        x2 = self.conv1(x)
        x2 = F.leaky_relu(x2, 0.01)
        x2 = self.conv2(x2)
        x2 = F.leaky_relu(x2, 0.01)
        x2 = self.conv3(x2)
        x3 = x1 + x2
        x4 = F.leaky_relu(x3, 0.01)
        return x4

class ResNet50(nn.Module):
    def __init__(self, out_channels):
        super(ResNet50, self).__init__()
        self.layers = nn.Sequential(CBM(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, dilation=1, bias=True),
                                    # stage 1
                                    ResBlock(64,   256, 1),
                                    ResBlock(256,  256),
                                    ResBlock(256,  256),
                                    # stage 2
                                    ResBlock(256,  512, 2, 1),
                                    ResBlock(512,  512),
                                    ResBlock(512,  512),
                                    ResBlock(512,  512),
                                    # stage 3
                                    ResBlock(512,  1024, 2, 1),
                                    ResBlock(1024, 1024),
                                    ResBlock(1024, 1024),
                                    ResBlock(1024, 1024),
                                    ResBlock(1024, 1024),
                                    # stage 4
                                    ResBlock(1024, 2048, 2, 1),
                                    ResBlock(2048, 2048),
                                    ResBlock(2048, 2048),
                                    # stage 5
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(2048, out_channels, bias=True))
        return

    def forward(self, x):
        x = self.layers(x)
        return x

