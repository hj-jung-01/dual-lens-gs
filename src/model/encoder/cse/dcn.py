import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DynamicAggregation(nn.Module):
    """Modulated deformable convolution whose sampling offsets are initialized
    with precomputed correspondences (C2-Matching, Jiang et al. CVPR 2021).

    Offsets/masks are predicted from a separate feature map, and the pre-offset
    from patch matching is added on top of the predicted residual offsets.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        k = self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * k,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, feat, pre_offset):
        """
        x: features to aggregate, [b, c, h, w]
        feat: features used to predict offsets and masks, [b, c, h, w]
        pre_offset: precomputed flow in (x, y) order, [b, 9, h, w, 2]
        """
        o1, o2, mask = torch.chunk(self.conv_offset_mask(feat), 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        # offsets are laid out as [y, x, y, x, ...] per group
        pre_offset = pre_offset.repeat([1, self.deformable_groups, 1, 1, 1])
        pre_offset_yx = torch.zeros_like(offset)
        pre_offset_yx[:, 0::2] = pre_offset[..., 1]
        pre_offset_yx[:, 1::2] = pre_offset[..., 0]
        offset = offset + pre_offset_yx

        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=torch.sigmoid(mask),
        )
