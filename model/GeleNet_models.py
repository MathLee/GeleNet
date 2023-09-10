import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor

# out = channel_shuffle(out, 2)
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# SWSAM: Shuffle Weighted Spatial Attention Module
class SWSAM(nn.Module):
    def __init__(self, channel=32): # group=8, branch=4, group x branch = channel
        super(SWSAM, self).__init__()

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(BasicConv2d(1, 1, 3, padding=1),
                                       nn.Sigmoid()
                                       )

    def forward(self, x):
        x = channel_shuffle(x, 4)
        x1, x2, x3, x4 = torch.split(x, 8, dim = 1)
        s1 = self.SA1(x1)
        s2 = self.SA1(x2)
        s3 = self.SA1(x3)
        s4 = self.SA1(x4)
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1] + s3 * nor_weights[2] + s4 * nor_weights[3]
        x_out = self.sa_fusion(s_all) * x + x

        return x_out

class DirectionalConvUnit(nn.Module):
    def __init__(self, channel):
        super(DirectionalConvUnit, self).__init__()

        self.h_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))

    def forward(self, x):

        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        return x

    # Code from "CoANet- Connectivity Attention Network for Road Extraction From Satellite Imagery", and we modified the code
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

# Knowledge Transfer Module
class KTM(nn.Module):
    def __init__(self, channel=32):
        super(KTM, self).__init__()

        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        # following DANet
        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False),
                                      nn.Conv2d(channel, channel, 1)
                                      )


    def forward(self, x2, x3): # V
        x_sum = x2 + x3 # Q
        x_mul = x2 * x3 # K

        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x_sum.size()
        proj_query = self.query_conv(x_sum).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_mul).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value_2 = self.value_conv_2(x2).view(m_batchsize, -1, width * height)
        proj_value_3 = self.value_conv_3(x3).view(m_batchsize, -1, width * height)

        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1))
        out_2 = out_2.view(m_batchsize, C, height, width)
        out_2 = self.conv_2(self.gamma_2 * out_2 + x2)

        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1))
        out_3 = out_3.view(m_batchsize, C, height, width)
        out_3 = self.conv_3(self.gamma_3 * out_3 + x3)

        x_out = self.conv_out(out_2 + out_3)

        return x_out


class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3): # x1: 32x11x11, x2: 32x22x22, x3: 32x88x88,
        x1_1 = x1 # 32x11x11
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # 32x22x22
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) \
               * self.conv_upsample3(self.upsample(self.upsample(x2))) * x3 # 32x88x88

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1) # 32x22x22
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(self.upsample(x2_2)))), 1) # 32x88x88
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x) # 1x88x88

        return x


class GeleNet(nn.Module):
    def __init__(self, channel=32):
        super(GeleNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 2, 1) # 128x44x44->32x22x22
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        # SWSAM for x4_nor
        self.SWSAM_4 = SWSAM(channel)  # group x branch = channel

        # D-SWSAM for x1_nor
        self.dirConv = DirectionalConvUnit(channel)
        self.DSWSAM_1 = SWSAM(channel) # group x branch = channel

        # KTM for x2_nor and x3_nor
        self.KTM_23 = KTM(channel)

        self.PDecoder = PDecoder(channel)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2) # 32x22x22
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4) # 32x11x11

        # SWSAM for x4_nor
        x4_SWSAM_4 = self.SWSAM_4(x4_nor)  # 32x11x11

        # D-SWSAM for x1_nor
        x1_ori = self.dirConv(x1_nor)
        x1_DSWSAM_1 = self.DSWSAM_1(x1_ori) # 32x88x88

        # KTM for x2_nor and x3_nor
        x23_KTM = self.KTM_23(x2_nor, x3_nor) # 32x22x22


        prediction = self.upsample_4(self.PDecoder(x4_SWSAM_4, x23_KTM, x1_DSWSAM_1))


        return prediction, self.sigmoid(prediction)
