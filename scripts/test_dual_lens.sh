#!/usr/bin/env bash
# Usage: bash scripts/test_dual_lens.sh [checkpoint] [data root]
CKPT=${1:-checkpoints/re10k.ckpt}
ROOT=${2:-datasets/dual_lens}

python -m src.main +experiment=dual_lens \
    checkpointing.load=$CKPT \
    "dataset.roots=[$ROOT]"
