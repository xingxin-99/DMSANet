#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""

import torch
import torch.nn as nn
from torchsummary import summary

import sys

current_module = sys.modules[__name__]

debug = False

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class lag_layer(nn.Module):


    def __init__(self, channel):
        super(lag_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class DMS_Conv(nn.Module):
    def __init__(self, channels=20, timekernel=25):
        super(DMS_Conv, self).__init__()
        self.ts1 = nn.Sequential(
            Conv2dWithConstraint(1, channels , (1, timekernel), padding='same', max_norm=2)
        )
        self.ts2 = nn.Sequential(
            Conv2dWithConstraint(1, channels, (1, timekernel), padding='same', max_norm=2),
            Conv2dWithConstraint(channels, channels, (1, timekernel), padding='same', max_norm=2,
                                 groups=channels),
        )
        self.ts3 = nn.Sequential(
            Conv2dWithConstraint(1, channels, (1, timekernel), padding='same', max_norm=2),
            Conv2dWithConstraint(channels, channels, (1, timekernel), padding='same', max_norm=2,
                                 groups=channels),
            Conv2dWithConstraint(channels, channels, (1, timekernel), padding='same', max_norm=2,
                                 groups=channels),
        )

    def forward(self, x):
        x1 = self.ts1(x)
        x2 = self.ts2(x)
        x3 = self.ts3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return x

class HS_Conv(nn.Module):
    def __init__(self, channels=20, timekernel=25):
        super(HS_Conv, self).__init__()
        self.ts1 = nn.Sequential(
            Conv2dWithConstraint(1, channels , (1, timekernel), padding='same', max_norm=2)
        )
        self.ts2 = nn.Sequential(
            Conv2dWithConstraint(1, channels, (1, timekernel*2), padding='same', max_norm=2),
        )
        self.ts3 = nn.Sequential(
            Conv2dWithConstraint(1, channels, (1, timekernel*3), padding='same', max_norm=2),
        )

    def forward(self, x):
        x1 = self.ts1(x)
        x2 = self.ts2(x)
        x3 = self.ts3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return x

class SS_Conv(nn.Module):
    def __init__(self, channels=20, timekernel=25):
        super(SS_Conv, self).__init__()
        self.ts = nn.Sequential(
            Conv2dWithConstraint(1, channels , (1, timekernel), padding='same', max_norm=2)
        )


    def forward(self, x):
        x = self.ts(x)
        return x


class DMSANet(nn.Module):
    def __init__(self, nChan=22, nTime=1000, nClass=4, dropoutP=0.25, *args, **kwargs):
        super(DMSANet, self).__init__()
        self.timekernel = 25
        self.poolkernel = 85
        self.poolstride = 15
        self.channels = 20
        self.totalChannels = self.channels*3
        self.D = 2

        self.DMS_Conv = DMS_Conv(self.channels,self.timekernel)
        self.SP = nn.Sequential(
            Conv2dWithConstraint(self.totalChannels, self.totalChannels*self.D, (nChan, 1), max_norm=2, groups=self.totalChannels),
            nn.BatchNorm2d(self.totalChannels*self.D),
            ActSquare(),
            nn.AvgPool2d((1, self.poolkernel), (1, self.poolstride)),
            ActLog(),
        )
        self.LAG = nn.Sequential(
            lag_layer(self.totalChannels*self.D),
            nn.Dropout(dropoutP)
        )
        self.FC = nn.Sequential(
            LinearWithConstraint(7440, nClass, max_norm=0.5),  # 3720->7440
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.DMS_Conv(x)
        x = self.SP(x)
        x = self.LAG(x)
        y = torch.flatten(x,start_dim=1)
        x = self.FC(y)

        return x,y

