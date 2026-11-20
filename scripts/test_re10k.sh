#!/usr/bin/env bash
# Usage: bash scripts/test_re10k.sh [checkpoint]
CKPT=${1:-checkpoints/re10k.ckpt}

python -m src.main +experiment=re10k \
    mode=test \
    checkpointing.load=$CKPT \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
    wandb.name=re10k
