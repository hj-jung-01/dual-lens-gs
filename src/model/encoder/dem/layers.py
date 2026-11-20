from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert kernel % 2 == 1, f"only odd kernel is supported but kernel = {kernel}"
    layers = [nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)


class Basic2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=None,
        kernel_size=3,
        padding=1,
        act=nn.ReLU,
    ):
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=norm_layer is None,
        )
        self.conv = nn.Sequential(OrderedDict([("conv", conv)]))
        if norm_layer:
            self.conv.add_module("bn", norm_layer(out_channels))
        self.conv.add_module("relu", act())

    def forward(self, x):
        return self.conv(x)


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=norm_layer is None,
        )
        self.bn = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.relu = act()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x.contiguous())))


class UpCat(nn.Module):
    """Upsample the accumulated feature x2 and fuse it with a skip feature."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.upf = Basic2dTrans(in_channels, out_channels, norm_layer=norm_layer)
        self.conv = Basic2d(out_channels * 2, out_channels, norm_layer=norm_layer)

    def forward(self, feature, accumulated):
        return self.conv(torch.cat([self.upf(accumulated), feature], dim=1))


class LocalConv(Function):
    """Per-pixel k x k filtering with spatially varying kernels."""

    @staticmethod
    def forward(ctx, input, weight):
        k = int(weight.shape[1] ** 0.5)
        ctx.save_for_backward(input, weight)
        b, _, h, w = input.shape
        patches = F.unfold(input, kernel_size=k, padding=k // 2).view(b, k * k, h, w)
        return torch.sum(patches * weight, dim=1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        k = int(weight.shape[1] ** 0.5)
        grad_output = grad_output.contiguous()
        b, _, h, w = input.shape
        patches = F.unfold(input, kernel_size=k, padding=k // 2).view(b, k * k, h, w)
        grad_input = torch.sum(grad_output * weight, dim=1, keepdim=True)
        grad_weight = grad_output * patches
        return grad_input, grad_weight


local_conv = LocalConv.apply
