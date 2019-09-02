# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import torch.functional as F
import numpy as np
import os

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN32s(nn.Module):
    def __init__(self,n_class=5):
        super(FCN32s,self).__init__()
        # conv1
        self.feature= models.vgg16().features
        # self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        # self.relu1_1 = nn.ReLU(inplace=True)
        # self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.relu1_2 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        #
        # # conv2
        # self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.relu2_1 = nn.ReLU(inplace=True)
        # self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.relu2_2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        #
        # # conv3
        # self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.relu3_1 = nn.ReLU(inplace=True)
        # self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.relu3_2 = nn.ReLU(inplace=True)
        # self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.relu3_3 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        #
        # # conv4
        # self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        # self.relu4_1 = nn.ReLU(inplace=True)
        # self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_2 = nn.ReLU(inplace=True)
        # self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_3 = nn.ReLU(inplace=True)
        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        #
        # # conv5
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_2 = nn.ReLU(inplace=True)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_3 = nn.ReLU(inplace=True)
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1) #1*1卷积层
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 32, stride=32,
                                          bias=False)

        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)# x.shape = (512,8,50)
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        x = self.score_fr(x)


        x = self.upscore(x)


        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)



