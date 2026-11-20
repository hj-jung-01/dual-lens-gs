# Multi-scale feature extraction from MiDaS, adapted from ZoeDepth
# (https://github.com/isl-org/ZoeDepth, MIT License).

import torch
import torch.nn as nn
import torch.nn.functional as F


class MidasCore(nn.Module):
    """Wraps a MiDaS model and collects intermediate decoder activations
    with forward hooks."""

    layer_names = ("out_conv", "l4_rn", "r4", "r3", "r2", "r1")

    def __init__(self, midas: nn.Module, img_size=(256, 256), trainable=True):
        super().__init__()
        self.core = midas
        self.img_size = tuple(img_size)
        self.trainable = trainable
        self.core_out = {}
        self.handles = []
        self.attach_hooks()

    @classmethod
    def build(cls, model_type="MiDaS_small", pretrained=True, **kwargs):
        midas = torch.hub.load(
            "intel-isl/MiDaS", model_type, pretrained=pretrained, trust_repo=True
        )
        return cls(midas, **kwargs)

    def attach_hooks(self):
        def save(name):
            def hook(module, inputs, output):
                self.core_out[name] = output

            return hook

        scratch = self.core.scratch
        modules = {
            "out_conv": list(scratch.output_conv.children())[3],
            "l4_rn": scratch.layer4_rn,
            "r4": scratch.refinenet4,
            "r3": scratch.refinenet3,
            "r2": scratch.refinenet2,
            "r1": scratch.refinenet1,
        }
        for name in self.layer_names:
            self.handles.append(modules[name].register_forward_hook(save(name)))

    def prepare(self, x):
        h, w = (int(round(s / 32) * 32) for s in self.img_size)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return (x - 0.5) / 0.5

    def forward(self, x):
        with torch.no_grad():
            x = self.prepare(x)
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            rel_depth = self.core(x)
        return rel_depth, [self.core_out[k] for k in self.layer_names]
