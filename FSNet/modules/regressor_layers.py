"""
Some functions are borrowed from LoFTR: Detector-Free Local
Feature Matching with Transformers (https://github.com/zju3dv/LoFTR) and modified here.
If using this code, please consider citing LoFTR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) if bn else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class DeepResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn, padding_mode = 'zeros'):
        super().__init__()

        self.resblock1 = PreActBlock(in_channels, 64, stride=2, bn=bn, padding_mode=padding_mode)
        self.resblock2 = PreActBlock(64, 128, stride=2, bn=bn, padding_mode=padding_mode)
        self.resblock3 = PreActBlock(128, 256, stride=2, bn=bn, padding_mode=padding_mode)
        self.resblock4 = PreActBlock(256, out_channels, stride=2, bn=bn, padding_mode=padding_mode)


    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super().__init__()

        self.conv1 = conv_1x1(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.conv2 = conv_1x1(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        B, _, _, _ = x.shape

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


def conv_1x1(in_feats, out_feats, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_feats),
            nn.ReLU())
    else:
        return nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.ReLU())
