import torch.nn as nn
import torch
import torch.nn.functional as F

from layer import *

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, norm='bnorm'):
        super(ResidualConv, self).__init__()

        norm_layer = []

        if not norm is None:
            if norm == "bnorm":
                norm_layer += [nn.BatchNorm2d(output_dim)]
            elif norm == "inorm":
                norm_layer += [nn.InstanceNorm2d(output_dim)]

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1),
            *norm_layer,
        )

        self.conv_block = nn.Sequential(BRC2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm),
                                        BRC2d(output_dim, output_dim, kernel_size=kernel_size, padding=padding,
                                              norm=norm)
                                        )

    def forward(self, x):
        return self.conv_skip(x) + self.conv_block(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# class ASPP_V3(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(ASPP_V3, self).__init__()
#
#         self.bn_conv_0 = nn.BatchNorm2d(in_channels)
#         # 1번 branch = 1x1 convolution → BatchNorm → ReLu
#         self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
#
#         # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
#         self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
#         self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
#
#         # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
#         self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
#         self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
#
#         # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
#         self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
#         self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)
#
#         # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
#
#         self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)  # (1280 = 5*256)
#         self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
#
#         self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size=1)
#
#     def forward(self, feature_map):
#         # feature map의 shape은 (batch_size, in_channels, height/output_stride, width/output_stride)
#
#         feature_map_h = feature_map.size()[2]  # (== h/16)
#         feature_map_w = feature_map.size()[3]  # (== w/16)
#
#         # 1번 branch = 1x1 convolution → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#
#         feature_map = F.relu(self.bn_conv_0(feature_map))
#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
#         # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
#         # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
#         # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
#
#         # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
#         # shape: (batch_size, in_channels, 1, 1)
#         out_img = self.avg_pool(feature_map)
#         # shape: (batch_size, out_channels, 1, 1)
#
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
#         # for batch_size = 1
#         # out_img = F.relu(self.conv_1x1_2(out_img))
#
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")
#
#         # shape: (batch_size, out_channels * 5, height/output_stride, width/output_stride)
#         out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
#         # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
#         out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
#         # shape: (batch_size, num_classes, height/output_stride, width/output_stride)
#         out = self.conv_1x1_4(out)
#
#         return out
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


class ASPP_V3(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, norm):
        super(ASPP_V3, self).__init__()

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        # self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_1x1_1 = BRC2d(in_channels, out_channels, kernel_size=1, padding=0, norm=norm)
        self.br_conv_1x1_1 = BR2d(out_channels, norm=norm)

        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        # self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv_3x3_1 = BRC2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, norm=norm)
        self.br_conv_3x3_1 = BR2d(out_channels, norm=norm)

        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        # self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_3x3_2 = BRC2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, norm=norm)
        self.br_conv_3x3_2 = BR2d(out_channels, norm=norm)

        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        # self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.conv_3x3_3 = BRC2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, norm=norm)
        self.br_conv_3x3_3 = BR2d(out_channels, norm=norm)

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = CBR2d(in_channels, out_channels, kernel_size=1, padding=0, norm='bnorm')

        # self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)  # (1280 = 5*256)
        self.conv_1x1_3 = CBR2d(out_channels * 5, out_channels, kernel_size=1, padding=0, norm=norm)

        self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # feature map의 shape은 (batch_size, in_channels, height/output_stride, width/output_stride)

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)

        # feature_map = F.relu(self.bn_conv_0(feature_map))

        out_1x1 = self.br_conv_1x1_1(self.conv_1x1_1(feature_map))
        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_1 = self.br_conv_3x3_1(self.conv_3x3_1(feature_map))
        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_2 = self.br_conv_3x3_2(self.conv_3x3_2(feature_map))
        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_3 = self.br_conv_3x3_3(self.conv_3x3_3(feature_map))

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, in_channels, 1, 1)
        out_img = self.avg_pool(feature_map)
        # shape: (batch_size, out_channels, 1, 1)

        out_img = self.conv_1x1_2(out_img)
        # for batch_size = 1
        # out_img = F.relu(self.conv_1x1_2(out_img))

        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        # shape: (batch_size, out_channels * 5, height/output_stride, width/output_stride)
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out = self.conv_1x1_3(out)
        # shape: (batch_size, num_classes, height/output_stride, width/output_stride)
        out = self.conv_1x1_4(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim, norm='bnorm'):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            BRC2d(input_encoder, output_dim, 3, padding=1, norm=norm),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            BRC2d(input_decoder, output_dim, 3, padding=1, norm=norm),
        )

        self.conv_attn = nn.Sequential(
            BRC2d(output_dim, 1, 1, padding=0, norm=norm),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2