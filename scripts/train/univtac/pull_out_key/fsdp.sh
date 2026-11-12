#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/lerobot

exec .venv/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  .venv/bin/lerobot-train \
  --config_path=configs/univtac/pull_out_key/20261110_001.yaml
