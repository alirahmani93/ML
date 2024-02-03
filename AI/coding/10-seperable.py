import torch
import torch.nn.functional as F
from torch import nn


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, *args, groups=in_channels, **kwargs)
        self.point_wise = nn.Conv2d(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x

model = SeparableConv2d()
