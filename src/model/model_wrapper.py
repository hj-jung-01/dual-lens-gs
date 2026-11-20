import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from .decoder.decoder import Decoder
from .encoder import Encoder


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    enhancement_loss_weight: float
    print_log_every_n_steps: int


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: Encoder
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.benchmarker = Benchmarker()
        self.test_step_outputs = {}
        self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def render(self, batch: BatchedExample, views: slice = slice(None)):
        gaussians, enhanced = self.encoder(batch["context"], self.global_step)
        target = batch["target"]
        output = self.decoder.forward(
            gaussians,
            target["extrinsics"][:, views],
            target["intrinsics"][:, views],
            target["near"][:, views],
            target["far"][:, views],
            tuple(target["upsized"].shape[-2:]),
        )
        return output, enhanced

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        output, enhanced = self.render(batch)
        target_gt = batch["target"]["upsized"]
        enhanced_gt = batch["context"]["upsized"][:, 1:]

        psnr = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr.mean())

        # reconstruction loss L_r and enhancement loss L_e
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output.color, target_gt, self.global_step)
            self.log(f"loss/render_{loss_fn.name}", loss)
            total_loss = total_loss + loss
        for loss_fn in self.losses:
            loss = loss_fn.forward(enhanced, enhanced_gt, self.global_step)
            loss = self.train_cfg.enhancement_loss_weight * loss
            self.log(f"loss/enhance_{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # for the checkpoint monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, _, _ = batch["target"]["upsized"].shape
        assert b == 1

        with self.benchmarker.time("encoder"):
            gaussians, _ = self.encoder(batch["context"], self.global_step)
        with self.benchmarker.time("decoder", num_calls=v):
            target = batch["target"]
            output = self.decoder.forward(
                gaussians,
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                tuple(target["upsized"].shape[-2:]),
            )

        (scene,) = batch["scene"]
        rgb = output.color[0]
        rgb_gt = batch["target"]["upsized"][0]

        path = self.test_cfg.output_path / get_cfg()["wandb"]["name"]
        if self.test_cfg.save_image:
            for index, color, color_gt in zip(batch["target"]["index"][0], rgb, rgb_gt):
                save_image(color, path / scene / f"color/{index:0>6}.png")
                save_image(color_gt, path / scene / f"color_gt/{index:0>6}.png")

        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            scores = {
                "psnr": compute_psnr(rgb_gt, rgb).mean().item(),
                "ssim": compute_ssim(rgb_gt, rgb).mean().item(),
                "lpips": compute_lpips(rgb_gt, rgb).mean().item(),
            }
            for name, value in scores.items():
                self.test_step_outputs.setdefault(name, []).append(value)
            self.test_step_outputs.setdefault("scene", []).append(scene)

    def on_test_end(self) -> None:
        out_dir = self.test_cfg.output_path / get_cfg()["wandb"]["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarker.dump(out_dir / "benchmark.json")

        if not self.test_cfg.compute_scores:
            self.benchmarker.summarize()
            return

        summary = {}
        scenes = self.test_step_outputs.pop("scene")
        for name, values in self.test_step_outputs.items():
            summary[name] = float(np.mean(values))
            print(f"{name}: {summary[name]:.4f}")
        with (out_dir / "scores_per_scene.json").open("w") as f:
            json.dump(
                {s: {k: v[i] for k, v in self.test_step_outputs.items()} for i, s in enumerate(scenes)},
                f,
                indent=1,
            )

        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(self.time_skip_steps_dict[tag]):]
            summary[f"time_{tag}"] = float(np.mean(times))
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times):.4f} seconds per call")
        with (out_dir / "scores.json").open("w") as f:
            json.dump(summary, f, indent=2)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        assert batch["target"]["upsized"].shape[0] == 1

        output, _ = self.render(batch, views=slice(0, 3))
        rgb = output.color[0]
        rgb_gt = batch["target"]["upsized"][0][:3]

        self.log("val/psnr", compute_psnr(rgb_gt, rgb).mean())
        self.log("val/lpips", compute_lpips(rgb_gt, rgb).mean())
        self.log("val/ssim", compute_ssim(rgb_gt, rgb).mean())

        comparison = hcat(
            add_label(vcat(*batch["context"]["upsized"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb), "Target (Rendered)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 1 / warm_up_steps, 1, total_iters=warm_up_steps
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
