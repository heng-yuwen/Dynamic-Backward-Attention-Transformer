###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Conv2d, AvgPool2d, ReLU, BatchNorm2d
from torch.nn.functional import interpolate
from torchtools.models.crfasrnn.crfrnn import CrfRnn


class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """

    def __init__(self, nclass, backbone: nn.Module, plus=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        # turn into needed shape
        self.backbone = backbone
        self.backbone.fc = nn.Conv2d(2048, nclass, kernel_size=1, bias=True)
        self.backbone.avgpool = nn.Identity()
        # out = self.backbone.forward(torch.randn(2,3,256,256))

        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.head = DeepLabV3Head(2048, 256, norm_layer, self._up_kwargs)
        self.plus = plus

        # v3 plus
        self.conv1 = Conv2d(256, 48, 1)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = Conv2d(304, 256, 3, padding=1)
        self.relu2 = ReLU(inplace=True)
        self.conv3 = Conv2d(256, 256, 3, padding=1)
        self.relu3 = ReLU(inplace=True)
        self.conv4 = Conv2d(256, nclass, 1, padding=0)

        self.bn2 = BatchNorm2d(256)
        self.bn3 = BatchNorm2d(256)
        # self.relu4 = ReLU(inplace=True)
        # self.crfrnn = CrfRnn(num_labels=nclass, num_iterations=10)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4, c5 = self.backbone.forward(x)
        out = self.head(c4)

        if self.plus:
            c1 = self.relu1(self.conv1(c1))
            _, _, h1, w1 = c1.size()
            out = interpolate(out, (h1, w1), **self._up_kwargs)
            plus_out = torch.cat((out, c1), dim=1)
            # plus_out = interpolate(plus_out, (h1*2, w1*2), **self._up_kwargs)
            plus_out = self.conv4(self.relu3(self.bn3(self.conv3(self.relu2(self.bn2(self.conv2(plus_out)))))))
            plus_out = interpolate(plus_out, (h, w), **self._up_kwargs)
            # plus_out = self.crfrnn(x, plus_out)
            return plus_out + interpolate(c5, (h, w), **self._up_kwargs)
        else:
            x = interpolate(x, (h, w), **self._up_kwargs)
            return x


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=[12, 24, 36], **kwargs):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
