#!/usr/bin/env bash
# Usage: bash scripts/test_acid_narrow.sh [checkpoint]
CKPT=${1:-checkpoints/acid.ckpt}

python -m src.main +experiment=acid \
    mode=test \
    checkpointing.load=$CKPT \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/evaluation_index_acid_narrow_interpolation.json \
    wandb.name=acid_narrow
