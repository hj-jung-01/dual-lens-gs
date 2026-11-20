from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDualLensCfg(DatasetCfgCommon):
    name: Literal["dual_lens"]
    roots: list[Path]
    downscale_factor: int


class DatasetDualLens(IterableDataset):
    """Pre-processed dual-lens captures, one ``.pth`` file per scene.

    Each file stores a dict with ``context``/``target`` views (extrinsics as
    OpenCV-style camera-to-world matrices, intrinsics normalized by the image
    size, float images in [0, 1], near and far) and a ``scene`` name. Context
    view 0 is the narrow-FoV (tele/wide) camera and view 1 the wide-FoV
    (ultra-wide) camera cropped to the overlapping region.
    """

    def __init__(self, cfg: DatasetDualLensCfg, stage: Stage, view_sampler: ViewSampler) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.scenes = sorted(p for root in cfg.roots for p in Path(root).glob("*.pth"))

    def __iter__(self):
        scenes = self.scenes
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            scenes = scenes[worker_info.id :: worker_info.num_workers]

        for path in scenes:
            example = torch.load(path, weights_only=False)
            example.setdefault("scene", path.stem)
            yield apply_crop_shim(example, tuple(self.cfg.image_shape), self.cfg.downscale_factor)

    def __len__(self) -> int:
        return len(self.scenes)
