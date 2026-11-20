from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, BatchedViews, DataShim
from ...geometry.projection import sample_image_grid
from ...global_cfg import get_cfg
from ..types import Gaussians
from .backbone import BackboneMultiview
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .cse import CrossScaleEnhancement
from .dem import DepthEnhancementModule
from .encoder import Encoder
from .grm import GaussianRefinementModule


@dataclass
class CSEWeightsCfg:
    extractor: str | None
    restoration: str | None


@dataclass
class EncoderDualLensCfg:
    name: Literal["dual_lens"]
    d_feature: int
    num_depth_candidates: int
    gaussian_adapter: GaussianAdapterCfg
    unimatch_weights_path: str | None
    cse_weights: CSEWeightsCfg
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: list[int]
    costvolume_unet_attn_res: list[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: list[int]
    depth_unet_channel_mult: list[int]
    propagation_steps: int
    foundation_image_size: list[int]


class EncoderDualLens(Encoder[EncoderDualLensCfg]):
    """Coarse-to-fine feed-forward 3DGS for asymmetric dual-lens stereo.

    Context view 0 is the narrow-FoV (high-resolution) image and view 1 the
    cropped wide-FoV (low-resolution) image. ``context["image"]`` holds both
    views at the low resolution, ``context["upsized"]`` holds them at the high
    resolution (for view 1 it is only used as supervision).
    """

    def __init__(self, cfg: EncoderDualLensCfg) -> None:
        super().__init__(cfg)
        pretrained = get_cfg().mode == "train"
        num_views = get_cfg().dataset.view_sampler.num_context_views
        assert num_views == 2, "the dual-lens encoder expects exactly two context views"

        # coarse network
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )
        if pretrained and cfg.unimatch_weights_path is not None:
            print(f"Loading UniMatch backbone weights from {cfg.unimatch_weights_path}")
            weights = torch.load(cfg.unimatch_weights_path, map_location="cpu")["model"]
            own = self.backbone.state_dict()
            self.backbone.load_state_dict({k: v for k, v in weights.items() if k in own})

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=self.gaussian_adapter.d_in,
            num_views=num_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )

        # fine network
        self.cse = CrossScaleEnhancement(
            cfg.cse_weights.extractor, cfg.cse_weights.restoration, pretrained=pretrained
        )
        self.dem = DepthEnhancementModule(
            prop_time=cfg.propagation_steps,
            pretrained=pretrained,
            img_size=cfg.foundation_image_size,
        )
        # all coarse Gaussian channels except the disparity
        self.grm = GaussianRefinementModule(1 + self.gaussian_adapter.d_in, pretrained=pretrained)

    def coarse(self, context: BatchedViews) -> Tensor:
        """Coarse depth and Gaussian features on the low-resolution pair.

        Returns [b, v, h * w, c] with channels (disparity, density, xy offset,
        scale, rotation, SH)."""
        features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
        )
        disparities, densities, raw_gaussians = self.depth_predictor(
            features,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            images=rearrange(context["image"], "b v c h w -> (v b) c h w"),
            cnn_features=cnn_features,
        )
        return torch.cat((disparities[..., 0], densities[..., 0], raw_gaussians), dim=-1)

    @staticmethod
    def scatter_to_grid(coarse: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Map each coarse disparity to the high-resolution grid G, shifted by
        the predicted sub-pixel offset O. Returns the sparse disparity map and
        its validity mask."""
        bv, _, h_low, w_low = coarse.shape
        h, w = shape
        device = coarse.device

        sparse = F.interpolate(coarse[:, :1], size=(h, w), mode="nearest")

        offset = rearrange(coarse[:, 2:4].sigmoid(), "b xy h w -> b h w xy")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h_low, device=device) / h_low,
            torch.arange(w_low, device=device) / w_low,
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)[None]
        pixel_size = torch.tensor((1 / w_low, 1 / h_low), device=device)
        coords = grid + offset * pixel_size

        x = (coords[..., 0] * w).round().long().clamp(0, w - 1)
        y = (coords[..., 1] * h).round().long().clamp(0, h - 1)
        batch = torch.arange(bv, device=device).view(-1, 1, 1)

        mask = torch.zeros((bv, h, w), device=device)
        mask[batch, y, x] = 1
        mask = mask[:, None]
        return sparse * mask, mask

    def forward(self, context: BatchedViews, global_step: int) -> tuple[Gaussians, Tensor]:
        """Returns the Gaussians and the enhanced wide-FoV image I_w^h."""
        b, v, _, h_low, w_low = context["image"].shape
        _, _, _, h, w = context["upsized"].shape

        coarse = self.coarse(context)
        coarse = rearrange(coarse, "b v (h w) c -> (b v) c h w", h=h_low, w=w_low)

        sparse, mask = self.scatter_to_grid(coarse, (h, w))
        sparse = rearrange(sparse, "(b v) c h w -> b v c h w", b=b, v=v)
        mask = rearrange(mask, "(b v) c h w -> b v c h w", b=b, v=v)

        narrow = context["upsized"][:, 0]
        wide = self.cse(narrow, context["image"][:, 1])

        depths, foundation = [], []
        for i, image in enumerate((narrow, wide)):
            disp, feats = self.dem(image, sparse[:, i], mask[:, i])
            near = rearrange(context["near"][:, i], "b -> b () () ()")
            far = rearrange(context["far"][:, i], "b -> b () () ()")
            depths.append(1.0 / disp.clamp(1.0 / far, 1.0 / near))
            foundation.append(feats)
        depths = rearrange(torch.stack(depths, dim=1), "b v c h w -> (b v) c h w")
        images = rearrange(torch.stack((narrow, wide), dim=1), "b v c h w -> (b v) c h w")

        # MiDaS features ordered from coarse to fine: [r4, r3, r2, r1, out_conv]
        foundation = [
            rearrange(
                torch.stack((foundation[0][i], foundation[1][i]), dim=1),
                "b v c h w -> (b v) c h w",
            )
            for i in (2, 3, 4, 5, 0)
        ]

        coarse_up = F.interpolate(
            coarse[:, 1:], size=(h, w), mode="bilinear", align_corners=False
        )
        fine = self.grm(images, coarse_up, foundation)

        features = torch.cat((depths, fine), dim=1)
        features = rearrange(features, "(b v) c h w -> b v (h w) () c", b=b, v=v)
        depths = features[..., 0]
        opacities = features[..., 1].sigmoid()
        raw_gaussians = features[..., 2:]

        xy_ray, _ = sample_image_grid((h, w), features.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=features.device)
        xy_ray = xy_ray + (raw_gaussians[..., :2].sigmoid() - 0.5) * pixel_size

        gaussians = self.gaussian_adapter(
            rearrange(context["extrinsics"], "b v i j -> b v () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () i j"),
            xy_ray,
            depths,
            opacities,
            raw_gaussians[..., 2:],
            (h, w),
        )

        return (
            Gaussians(
                rearrange(gaussians.means, "b v r srf xyz -> b (v r srf) xyz"),
                rearrange(gaussians.rotations, "b v r srf xyzw -> b (v r srf) xyzw"),
                rearrange(gaussians.scales, "b v r srf xyz -> b (v r srf) xyz"),
                rearrange(gaussians.harmonics, "b v r srf c d_sh -> b (v r srf) c d_sh"),
                rearrange(gaussians.opacities, "b v r srf -> b (v r srf)"),
            ),
            rearrange(wide, "(b v) c h w -> b v c h w", v=1),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            return apply_patch_shim(
                batch, patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor
            )

        return data_shim
