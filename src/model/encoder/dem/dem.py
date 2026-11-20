import torch
import torch.nn as nn

from .cspn_plusplus import CSPNPlusPlus
from .layers import UpCat, conv_bn_relu
from .midas import MidasCore


class FoundationGuidance(nn.Module):
    """Builds the affinity feature A by accumulating multi-level features of a
    monocular depth model (MiDaS-small) and predicts the initial dense depth
    d^(0) from A, the image and the sparse depth."""

    def __init__(self, pretrained=True, img_size=(256, 256)):
        super().__init__()
        self.backbone = MidasCore.build("MiDaS_small", pretrained=pretrained, img_size=img_size)
        # only the biases of the foundation model are updated
        for name, param in self.backbone.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

        self.conv1_rgb = conv_bn_relu(3, 24, kernel=3, stride=1, padding=1, bn=False)
        self.conv1_depth = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1, bn=False)

        self.upconv1 = UpCat(512, 256)
        self.upconv2 = UpCat(256, 128)
        self.upconv3 = UpCat(128, 64)
        self.upconv4 = UpCat(64, 64)
        self.upconv5 = UpCat(64, 32)

        self.id_dec0 = conv_bn_relu(32 + 32, 1, kernel=3, stride=1, padding=1, bn=False)

    def forward(self, image, sparse):
        fe1 = torch.cat((self.conv1_rgb(image), self.conv1_depth(sparse)), dim=1)

        # [out_conv, l4_rn, r4, r3, r2, r1]
        _, feats = self.backbone(image)

        affinity = self.upconv1(feats[2], feats[1])
        affinity = self.upconv2(feats[3], affinity)
        affinity = self.upconv3(feats[4], affinity)
        affinity = self.upconv4(feats[5], affinity)
        affinity = self.upconv5(feats[0], affinity)

        init_depth = self.id_dec0(torch.cat((affinity, fe1), dim=1))
        return init_depth, affinity, [f.detach() for f in feats]


class DepthEnhancementModule(nn.Module):
    """Depth enhancement module (DEM).

    Upsamples the coarse disparity, scattered on the high-resolution grid as
    a sparse map, by spatial propagation guided by depth foundation features.
    """

    def __init__(self, prop_time=18, pretrained=True, img_size=(256, 256)):
        super().__init__()
        self.backbone = FoundationGuidance(pretrained=pretrained, img_size=img_size)
        self.cspn = CSPNPlusPlus(32, prop_time)

    def forward(self, image, sparse, mask):
        """Returns the dense disparity and the foundation features E^(l)."""
        init_depth, affinity, foundation_features = self.backbone(image, sparse)
        dense = self.cspn(affinity, init_depth, sparse, mask)
        return dense, foundation_features
