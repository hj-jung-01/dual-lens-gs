#!/usr/bin/env bash
# Trained on 8 x RTX 3090 (batch size 2 per GPU) for 400k iterations.

python -m src.main +experiment=acid \
    data_loader.train.batch_size=2 \
    trainer.max_steps=400_001 \
    checkpointing.every_n_train_steps=20000
