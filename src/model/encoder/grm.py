import torch
import torch.nn as nn
import torchvision


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    layers = [nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    layers = [nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding, output_padding, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def crop_concat(fd, fe):
    """Concatenate a decoder feature with an encoder skip, cropping padding."""
    _, _, hd, wd = fd.shape
    _, _, he, we = fe.shape
    return torch.cat((fd[:, :, :he, :we], fe), dim=1)


class GaussianRefinementModule(nn.Module):
    """Gaussian feature refinement module (GRM).

    A ResNet-34 U-Net that takes the high-resolution image and the upsampled
    coarse Gaussian features. A second decoder injects the multi-level depth
    foundation features E^(l) into the main decoder through a zero-initialized
    1x1 convolution.
    """

    def __init__(self, feature_dim, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim

        # encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1, bn=False)
        self.conv1_dep = conv_bn_relu(feature_dim, 16, kernel=3, stride=1, padding=1, bn=False)

        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        self.conv2 = net.layer1  # 1/1
        self.conv3 = net.layer2  # 1/2
        self.conv4 = net.layer3  # 1/4
        self.conv5 = net.layer4  # 1/8
        del net
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)  # 1/16

        # decoder
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec4 = convt_bn_relu(256 + 512, 128, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec3 = convt_bn_relu(128 + 256, 64, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec2 = convt_bn_relu(64 + 128, 64, kernel=3, stride=2, padding=1, output_padding=1)
        self.id_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.id_dec0 = conv_bn_relu(64 + 64, feature_dim, kernel=3, stride=1, padding=1, bn=False, relu=True)

        # foundation feature branch
        self.channel_matching0 = conv_bn_relu(256, 512, kernel=1)
        self.channel_matching1 = conv_bn_relu(128, 256, kernel=1)
        self.channel_matching2 = conv_bn_relu(64, 128, kernel=1)
        self.channel_matching3 = conv_bn_relu(64, 64, kernel=1)
        self.channel_matching4 = conv_bn_relu(32, 64, kernel=1)

        self.shared_dec5 = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1)
        self.shared_dec4 = convt_bn_relu(256, 128, kernel=3, stride=2, padding=1, output_padding=1)
        self.shared_dec3 = convt_bn_relu(128, 64, kernel=3, stride=2, padding=1, output_padding=1)
        self.shared_dec2 = convt_bn_relu(64, 64, kernel=3, stride=2, padding=1, output_padding=1)
        self.shared_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1)

        self.zero_conv = zero_module(nn.Conv2d(64, 64, kernel_size=1))

    def forward(self, image, coarse_features, foundation_features):
        """
        image: [B, 3, H, W]
        coarse_features: [B, C, H, W] coarse Gaussian features upsampled to H x W
        foundation_features: E^(l) from coarse to fine resolution (1/16 ... 1/1)
        """
        fe1 = torch.cat((self.conv1_rgb(image), self.conv1_dep(coarse_features)), dim=1)
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        e0, e1, e2, e3, e4 = foundation_features

        sd5 = self.shared_dec5(self.channel_matching0(e0) + fe6)
        fd5 = self.dec5(fe6)
        sd4 = self.shared_dec4(self.channel_matching1(e1) + fd5 + sd5)
        fd4 = self.dec4(crop_concat(fd5, fe5))
        sd3 = self.shared_dec3(self.channel_matching2(e2) + fd4 + sd4)
        fd3 = self.dec3(crop_concat(fd4, fe4))
        sd2 = self.shared_dec2(self.channel_matching3(e3) + fd3 + sd3)
        fd2 = self.dec2(crop_concat(fd3, fe3))
        sd1 = self.shared_dec1(self.channel_matching4(e4) + fd2 + sd2)

        fd2 = fd2 + self.zero_conv(sd1)

        fd1 = self.id_dec1(crop_concat(fd2, fe2))
        return self.id_dec0(crop_concat(fd1, fe1))
