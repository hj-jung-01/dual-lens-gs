from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

VGG16_NAMES = [
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
]
VGG19_NAMES = [
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3",
    "conv3_4", "relu3_4", "pool3",
]


def _vgg_features(arch, pretrained):
    if arch == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vgg16(weights=weights).features
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    return models.vgg19(weights=weights).features


class ImageNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ContrasExtractorLayer(ImageNormalize):
    """VGG16 up to conv3_1, fine-tuned with contrastive correspondence learning."""

    def __init__(self, pretrained=False):
        super().__init__()
        last = VGG16_NAMES.index("conv3_1")
        features = _vgg_features("vgg16", pretrained)[: last + 1]
        self.model = nn.Sequential(OrderedDict(zip(VGG16_NAMES, features)))

    def forward(self, x):
        return self.model(super().forward(x))


class ContrasExtractorSep(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.feature_extraction_image1 = ContrasExtractorLayer(pretrained)
        self.feature_extraction_image2 = ContrasExtractorLayer(pretrained)

    def forward(self, image1, image2):
        return (
            self.feature_extraction_image1(image1),
            self.feature_extraction_image2(image2),
        )


class VGGFeatureExtractor(ImageNormalize):
    """Frozen VGG19 returning intermediate activations."""

    def __init__(self, layer_name_list, pretrained=False):
        super().__init__()
        self.layer_name_list = layer_name_list
        last = max(VGG19_NAMES.index(name) for name in layer_name_list)
        features = _vgg_features("vgg19", pretrained)[: last + 1]
        self.vgg_net = nn.Sequential(OrderedDict(zip(VGG19_NAMES, features)))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output
