import functools

import torch
import torch.nn as nn

from .layers import Basic2d, local_conv


class GenKernel(nn.Module):
    """Predicts normalized k x k affinities. The center weight is set so that
    each kernel sums to one."""

    def __init__(self, in_channels, kernel_size, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, kernel_size**2 - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, x):
        weight = self.conv(x)
        weight = weight / (weight.abs().sum(dim=1, keepdim=True) + self.eps)
        weight_mid = 1 - weight.sum(dim=1, keepdim=True)
        half = weight.shape[1] // 2
        return torch.cat([weight[:, :half], weight_mid, weight[:, half:]], dim=1).contiguous()


class CSPNPlusPlus(nn.Module):
    """Spatial propagation with 3/5/7 kernels and learned fusion (CSPN++, Cheng et al.).

    Valid sparse entries are re-injected after every step through a learned
    confidence mask.
    """

    def __init__(self, in_channels, prop_time, norm_layer=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.prop_time = prop_time
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer, act)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer, act)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer, act)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    def forward(self, affinity_feature, init, sparse, mask):
        affinity_feature = affinity_feature.float()
        init, sparse, mask = init.float(), sparse.float(), mask.float()

        kernels = [self.weight3x3(affinity_feature),
                   self.weight5x5(affinity_feature),
                   self.weight7x7(affinity_feature)]
        masks = torch.split(self.convmask(affinity_feature) * mask, 1, dim=1)
        confs = torch.split(self.convck(affinity_feature), 1, dim=1)

        states = [init, init, init]
        snapshots = [init]
        for i in range(self.prop_time):
            states = [
                (1.0 - m) * local_conv(s, k) + m * sparse
                for s, k, m in zip(states, kernels, masks)
            ]
            if i == self.prop_time // 2 - 1:
                snapshots.append(sum(c * s for c, s in zip(confs, states)))
        snapshots.append(sum(c * s for c, s in zip(confs, states)))

        snapshots = torch.cat(snapshots, dim=1)
        weights = self.convct(torch.cat([affinity_feature, snapshots], dim=1))
        return torch.sum(weights * snapshots, dim=1, keepdim=True)
