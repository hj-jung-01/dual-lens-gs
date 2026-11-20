import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correspondence import CorrespondenceGeneration
from .restoration import RestorationNet
from .vgg import ContrasExtractorSep


def convert_c2_matching_keys(state_dict):
    """Map parameter names of the official C2-Matching release to this module."""
    out = {}
    for k, v in state_dict.items():
        k = k.removeprefix("module.")
        k = re.sub(r"dyn_agg_restore\.(small|medium|large)_offset_conv", r"\1.offset_conv", k)
        k = re.sub(r"dyn_agg_restore\.(small|medium|large)_dyn_agg", r"\1.dyn_agg", k)
        k = re.sub(r"dyn_agg_restore\.(head|body|tail)_(small|medium|large)", r"\2.\1", k)
        out[k] = v
    return out


class CrossScaleEnhancement(nn.Module):
    """Cross-scale enhancement (CSE).

    Transfers high-frequency details of the high-resolution narrow-FoV view to
    the low-resolution wide-FoV view (x4) through dense correspondences and
    deformable aggregation. The architecture follows C2-Matching.
    """

    def __init__(
        self,
        extractor_weights: str | None = None,
        restoration_weights: str | None = None,
        pretrained: bool = False,
    ):
        super().__init__()
        self.net_extractor = ContrasExtractorSep(pretrained)
        self.net_map = CorrespondenceGeneration(patch_size=3, stride=1, pretrained=pretrained)
        self.net_g = RestorationNet(ngf=64, n_blocks=16, groups=8)

        if pretrained:
            self._load(self.net_extractor, extractor_weights)
            self._load(self.net_g, restoration_weights)

    @staticmethod
    def _load(net, path):
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(
                f"C2-Matching weights not found at {path}. See README for download links."
            )
        state_dict = torch.load(path, map_location="cpu")
        net.load_state_dict(convert_c2_matching_keys(state_dict))

    def forward(self, ref_high, target_low):
        """
        ref_high: high-resolution reference view, [b, 3, 4h, 4w]
        target_low: low-resolution view to enhance, [b, 3, h, w]
        """
        target_init = F.interpolate(
            target_low, scale_factor=4, mode="bicubic", align_corners=False
        ).clamp(0, 1)

        features = self.net_extractor(target_init, ref_high)
        pre_offset, ref_feat = self.net_map(features, ref_high)
        output = self.net_g(target_low, pre_offset, ref_feat)
        return output.clamp(0, 1)
