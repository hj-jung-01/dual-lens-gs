# Feed-Forward Gaussian Splatting from Asymmetric Dual-lens Stereo

Official implementation of **Feed-Forward Gaussian Splatting from Asymmetric Dual-lens Stereo** (IJCV 2026).

Hyunjun Jung\*, Hojoon Lee\*, Changyeon Won, Jin-Hwi Park, Hae-Gon Jeon (\* equal contribution)

[[Paper](https://link.springer.com/article/10.1007/s11263-026-03117-2)]

Given a narrow-FoV high-resolution image and a wide-FoV low-resolution image from a dual-lens camera, our model predicts pixel-aligned 3D Gaussians in a single forward pass.

- **Coarse network**: a multi-view cost volume (MVSplat-style) on the low-resolution pair gives coarse depth and Gaussian features.
- **Cross-scale enhancement (CSE)**: high-frequency details of the narrow view are transferred to the wide view (x4) through dense correspondences.
- **Depth enhancement module (DEM)**: the coarse depth is scattered on the high-resolution grid with learned sub-pixel offsets and densified by spatial propagation (CSPN++) guided by MiDaS features.
- **Gaussian feature refinement module (GRM)**: a U-Net refines the coarse Gaussian features using the high-resolution images and the depth foundation features.

## Installation

Tested with Python 3.11, PyTorch 2.4.1 and CUDA 12.1.

```bash
conda create -n dualgs python=3.11 -y
conda activate dualgs
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The deformable convolution in CSE uses `torchvision.ops.deform_conv2d`, so no extra CUDA extension is needed besides the Gaussian rasterizer.
MiDaS-small is loaded through `torch.hub` (`intel-isl/MiDaS`), which requires internet access the first time it is built.

## Checkpoints

| Model | Training data | Download |
| --- | --- | --- |
| Ours | RealEstate10K (64x64 + 256x256) | [re10k.ckpt](https://drive.google.com/file/d/1FQcIAODgqRUh3Vkymmg9Mhv1xDVMtafX/view?usp=sharing) |

Put the checkpoint under `checkpoints/`.

To train from scratch, the following pretrained weights are also required:

- UniMatch depth model (initializes the multi-view backbone)
  ```bash
  wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth -P checkpoints
  ```
- [C2-Matching](https://github.com/yumingj/C2-Matching) feature extractor and restoration network (initializes CSE). Download them from the C2-Matching repository and save them as `checkpoints/feature_extraction.pth` and `checkpoints/c2_matching_restoration.pth`.

ImageNet weights for VGG/ResNet and the MiDaS-small weights are downloaded automatically.

## Datasets

### RealEstate10K / ACID

We use the same preprocessed `.torch` chunks as [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat). Place them as follows (or change `dataset.roots`):

```
datasets
├── re10k
│   ├── train/{000000.torch, ..., index.json}
│   └── test/{000000.torch, ..., index.json}
└── acid
    ├── train
    └── test
```

All images are center-cropped and resized to 256x256. Context view 0 is kept at 256x256 (narrow FoV) and context view 1 is downsampled to 64x64 (wide FoV). The rendered target views are evaluated at 256x256.

### Dual-lens captures

Real-world dual-lens scenes are stored as one `.pth` file per scene, containing a dict:

```python
{
    "context": {  # view 0: narrow FoV (wide lens), view 1: wide FoV (ultra-wide lens), cropped to the overlap
        "extrinsics": Tensor[2, 4, 4],   # camera-to-world, OpenCV convention
        "intrinsics": Tensor[2, 3, 3],   # normalized by image width / height
        "image": Tensor[2, 3, H, W],     # float in [0, 1]
        "near": Tensor[2], "far": Tensor[2],
        "index": list[str],
    },
    "target": {...},                     # same keys, N views
    "scene": str,
}
```

## Evaluation

```bash
# RealEstate10K
bash scripts/test_re10k.sh checkpoints/re10k.ckpt
bash scripts/test_re10k_narrow.sh checkpoints/re10k.ckpt   # 10 ~ 30 frame intervals

# cross-dataset generalization to ACID
bash scripts/test_acid.sh checkpoints/re10k.ckpt
bash scripts/test_acid_narrow.sh checkpoints/re10k.ckpt

# dual-lens captures (saves renderings)
bash scripts/test_dual_lens.sh checkpoints/re10k.ckpt /path/to/dual_lens_pths
```

Scores are written to `outputs/test/<name>/scores.json` and `scores_per_scene.json`. Set `test.save_image=true` to save rendered images.

## Training

```bash
bash scripts/train_re10k.sh
bash scripts/train_acid.sh
```

The paper model was trained on 8 RTX 3090 GPUs with a batch size of 2 per GPU for 400K iterations (Adam, lr 2e-4 with cosine decay).
The loss is `L = L_r + 0.1 * L_e`, where both terms are `MSE + 0.05 * LPIPS`, on the rendered target views and on the CSE output, respectively.
Logging to wandb can be enabled with `wandb.mode=online`.

## Acknowledgements

This code is built upon [MVSplat](https://github.com/donydchen/mvsplat) and [pixelSplat](https://github.com/dcharatan/pixelsplat).
We also use components from [UniMatch](https://github.com/autonomousvision/unimatch), [C2-Matching](https://github.com/yumingj/C2-Matching), [ZoeDepth](https://github.com/isl-org/ZoeDepth) / [MiDaS](https://github.com/isl-org/MiDaS) and CSPN++. We thank the authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE). Code adapted from third-party projects (listed above) remains under the license of the original project.

## Citation

```bibtex
@article{jung2026dualgs,
  title   = {Feed-Forward Gaussian Splatting from Asymmetric Dual-lens Stereo},
  author  = {Jung, Hyunjun and Lee, Hojoon and Won, Changyeon and Park, Jin-Hwi and Jeon, Hae-Gon},
  journal = {International Journal of Computer Vision},
  year    = {2026},
  doi     = {10.1007/s11263-026-03117-2}
}
```
