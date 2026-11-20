import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .dcn import DynamicAggregation


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


def make_layer(n_blocks, nf):
    return nn.Sequential(*[ResidualBlockNoBN(nf) for _ in range(n_blocks)])


class ContentExtractor(nn.Module):
    def __init__(self, in_nc=3, nf=64, n_blocks=16):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = make_layer(n_blocks, nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.body(self.lrelu(self.conv_first(x)))


class ScaleBranch(nn.Module):
    """One level of the coarse-to-fine restoration: aggregate the reference
    feature with deformable conv, fuse it with the current feature and
    upsample."""

    def __init__(self, ngf, ref_channels, n_blocks, groups, last=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(ngf + ref_channels, ref_channels, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(ref_channels, ref_channels, 3, 1, 1)
        self.dyn_agg = DynamicAggregation(
            ref_channels, ref_channels, 3, padding=1, deformable_groups=groups
        )
        self.head = nn.Sequential(
            nn.Conv2d(ngf + ref_channels, ngf, 3, 1, 1), nn.LeakyReLU(0.1, True)
        )
        self.body = make_layer(n_blocks, ngf)
        if last:
            self.tail = nn.Sequential(
                nn.Conv2d(ngf, ngf // 2, 3, 1, 1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(ngf // 2, 3, 3, 1, 1),
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True)
            )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, ref_feat, pre_offset):
        offset_feat = self.lrelu(self.offset_conv1(torch.cat([x, ref_feat], 1)))
        offset_feat = self.lrelu(self.offset_conv2(offset_feat))
        swapped = self.lrelu(self.dyn_agg(ref_feat, offset_feat, pre_offset))
        h = self.head(torch.cat([x, swapped], 1))
        h = self.body(h) + x
        return self.tail(h)


class RestorationNet(nn.Module):
    """x4 reference-based restoration of the low-resolution view."""

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super().__init__()
        self.content_extractor = ContentExtractor(3, ngf, n_blocks)
        self.small = ScaleBranch(ngf, 256, n_blocks, groups)
        self.medium = ScaleBranch(ngf, 128, n_blocks, groups)
        self.large = ScaleBranch(ngf, 64, n_blocks, groups, last=True)
        self.init_weights()

    def init_weights(self, gain=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight.data, 0.0, gain)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        for branch in (self.small, self.medium, self.large):
            branch.dyn_agg.conv_offset_mask.weight.data.zero_()
            branch.dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, img_ref_feat):
        base = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        feat = self.content_extractor(x)
        feat = self.small(feat, img_ref_feat["relu3_1"], pre_offset["relu3_1"])
        feat = self.medium(feat, img_ref_feat["relu2_1"], pre_offset["relu2_1"])
        feat = self.large(feat, img_ref_feat["relu1_1"], pre_offset["relu1_1"])
        return feat + base
