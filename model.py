import os
import numpy as np

import torch
import torch.nn as nn
import sys

from layer import *
from core.res_unet_plus import *

## Build Networks
# CycleGAN
# https://arxiv.org/pdf/1703.10593.pdf


class CycleGAN(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, norm='bnorm', nblk=6, learning_type='plain', network_block='unet', use_mask=False):
        super(CycleGAN, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nker = nker
        self.norm = norm
        self.nblk = nblk
        self.learning_type = learning_type
        self.network_block = network_block
        self.use_mask = use_mask

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        if use_mask:
            self.input_nc += 1

        self.enc1 = CBR2d(self.input_nc, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm,
                            relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm,
                            relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.output_nc, kernel_size=7, stride=1, padding=3, norm=None,
                          relu=None)
        res = []

        if self.nblk:
            if self.network_block == 'resnet':
                for i in range(self.nblk):
                    res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]
            elif self.network_block == 'unet':
                for i in range(self.nblk):
                    res += [UNet(self.input_nc, self.output_nc, nker=self.nker, learning_type=self.learning_type, norm=self.norm)]
            elif self.network_block == 'resunetplus':
                for i in range(self.nblk):
                    res += [ResUnetPlusPlus(self.input_nc)]
            elif self.network_block == 'resunetplus_v3':
                for i in range(self.nblk):
                    res += [ResUnetPlusPlusV3(self.input_nc, self.output_nc, norm=self.norm)]

        self.res = nn.Sequential(*res)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        if self.network_block == 'resnet':
            if self.use_mask:
                x = self.enc1(torch.cat((x, mask), 1))
            else:
                x = self.enc1(x)

            x = self.enc2(x)
            x = self.enc3(x)

            x = self.res(x)

            x = self.dec3(x)
            x = self.dec2(x)
            x = self.dec1(x)

        else:
            if self.use_mask:
                x = self.res(torch.cat((x, mask), 1))
            else:
                x = self.res(x)

        return self.tanh(x)


class Discriminator_cycle(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, norm="bnorm"):
        super(Discriminator_cycle, self).__init__()

        self.enc1 = CBR2d(1 * input_nc, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, output_nc, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

        self.fc_adv = nn.Sequential(
            LinearBlock(16 * nker * 8 * 8, 1024, 'none', 'relu'),
            LinearBlock(1024, 1, 'none', 'sigmoid')
        )

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        return torch.sigmoid(x)