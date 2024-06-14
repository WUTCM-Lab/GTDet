# -*- coding: utf-8 -*-
# @Time    : 2019/5/29 15:30
# @Author  : ljf
from __future__ import absolute_import
import torch
from torch import nn
from mmcv.cnn import constant_init, kaiming_init
import math

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        # nn.init.constant(m[-1].weight,val=0)
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True
class ContextBlock2d(nn.Module):

    # def __init__(self, inplanes, planes, pool, fusions):
    def __init__(self, input, output, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        # self.inplanes = inplanes
        # self.planes = planes
        self.input = input
        self.output = output
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.conv_mask = nn.Conv2d(input, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                # nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.input, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input, self.output, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        # if 'channel_mul' in fusions:
            # self.channel_mul_conv = nn.Sequential(
            #     nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            #     nn.LayerNorm([self.planes, 1, 1]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            # )
        # else:
        #     self.channel_mul_conv = None
        self.channel_mul_conv = None 
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            # 汇集全文的信息 对应的像素点进行匹配，整个图像的像素点全部相加
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x, out):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # if self.channel_mul_conv is not None:
        #     # [N, C, 1, 1]
        #     channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        #     out = x * channel_mul_term
        # else:
        #     out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


if __name__ == "__main__":
    inputs = torch.randn(1,16,300,300)
    block = ContextBlock2d(16,16,"att",["channel_add"])
    out = block(inputs)
    print(out.size())
