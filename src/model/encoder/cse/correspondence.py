import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import VGGFeatureExtractor


def sample_patches(inputs, patch_size=3, stride=1):
    """(c, h, w) -> (c, patch_size, patch_size, n_patches), row-major."""
    c, _, _ = inputs.shape
    return (
        inputs.unfold(1, patch_size, stride)
        .unfold(2, patch_size, stride)
        .reshape(c, -1, patch_size, patch_size)
        .permute(0, 2, 3, 1)
    )


def feature_match_index(feat_input, feat_ref, patch_size=3, stride=1):
    """Dense patch matching. Returns the index of the best reference patch for
    every input patch together with its normalized correlation."""
    patches_ref = sample_patches(feat_ref, patch_size, stride)

    _, h, w = feat_input.shape
    chunk = int(1024.0**2 * 512 / (h * w))
    n_patches = patches_ref.shape[-1]

    max_idx, max_val = None, None
    for idx in range(0, n_patches, chunk):
        batch = patches_ref[..., idx : idx + chunk]
        batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
        corr = F.conv2d(feat_input.unsqueeze(0), batch.permute(3, 0, 1, 2), stride=stride)
        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp
        else:
            better = max_val_tmp > max_val
            max_val[better] = max_val_tmp[better]
            max_idx[better] = max_idx_tmp[better] + idx

    patches_input = sample_patches(feat_input, patch_size, stride)
    norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
    norm = norm.view((h - patch_size) // stride + 1, (w - patch_size) // stride + 1)
    return max_idx, max_val / norm


def tensor_shift(x, shift):
    """Shift a [b, h, w, c] tensor towards the bottom-right, zero padded."""
    _, h, w, _ = x.size()
    shift_h, shift_w = shift
    out = torch.zeros_like(x)
    out[:, shift_h:, shift_w:, :] = x[:, : h - shift_h, : w - shift_w, :]
    return out


class CorrespondenceGeneration(nn.Module):
    """Turns patch matches between the two views into per-scale offsets for
    the deformable aggregation, and extracts reference VGG features."""

    def __init__(self, patch_size=3, stride=1, pretrained=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.vgg = VGGFeatureExtractor(["relu1_1", "relu2_1", "relu3_1"], pretrained)

    @staticmethod
    def index_to_flow(max_idx):
        device = max_idx.device
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float()
        flow = torch.stack((flow_w, flow_h), dim=2).unsqueeze(0).float()
        flow = flow - grid
        return F.pad(flow, (0, 0, 0, 2, 0, 2))

    @staticmethod
    def shifted_offsets(offset, scale):
        if scale > 1:
            offset = torch.repeat_interleave(offset, scale, 1)
            offset = torch.repeat_interleave(offset, scale, 2) * scale
        shifted = [
            tensor_shift(offset, (i * scale, j * scale)) for i in range(3) for j in range(3)
        ]
        return torch.cat(shifted, dim=0)

    def forward(self, features, img_ref):
        feat_in_all, feat_ref_all = features
        offsets = {"relu3_1": [], "relu2_1": [], "relu1_1": []}
        for feat_in, feat_ref in zip(feat_in_all, feat_ref_all):
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            max_idx, _ = feature_match_index(feat_in, feat_ref, self.patch_size, self.stride)
            offset = self.index_to_flow(max_idx)

            offsets["relu3_1"].append(self.shifted_offsets(offset, 1))
            offsets["relu2_1"].append(self.shifted_offsets(offset, 2))
            offsets["relu1_1"].append(self.shifted_offsets(offset, 4))

        # [b, 9, h, w, 2], last dim in (x, y)
        pre_offset = {k: torch.stack(v, dim=0) for k, v in offsets.items()}
        return pre_offset, self.vgg(img_ref)
