#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/lerobot
export HTTP_PROXY="http://10.19.125.92:7890"
export HTTPS_PROXY="http://10.19.125.92:7890"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
unset ALL_PROXY all_proxy

DATASET_ROOT=/root/datasets/univtac/pull_out_key_v30
OUTPUT_DIR=/root/workspace/lerobot/outputs/train/univtac_smolvla_panda9_b32_s8000_20261109

.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch

source = json.loads(Path('/root/datasets/univtac/pull_out_key/meta/info.json').read_text())
staged = json.loads(Path('/root/datasets/univtac/pull_out_key_v30/meta/info.json').read_text())
assert source['codebase_version'] == 'v2.1', source['codebase_version']
assert staged['codebase_version'] == 'v3.0', staged['codebase_version']
assert staged['features']['observation.state']['shape'] == [9]
assert staged['features']['action']['shape'] == [9]
assert torch.cuda.is_available(), 'CUDA unavailable; pslurm selected an incompatible GPU node.'
print('[train] original v2.1 retained; using staged v3.0 data, task=Empty')
print('[train] GPU:', torch.cuda.get_device_name(0))
PY

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[train] refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

echo "[train] output_dir=$OUTPUT_DIR"
echo "[train] steps=8000 batch_size=32 num_workers=16 state_dim=9 action_dim=9"

.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.state_dim=9 \
  --policy.action_dim=9 \
  --policy.adapt_pretrained_dimensions=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=univtac/pull_out_key \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size=32 \
  --num_workers=16 \
  --steps=8000 \
  --log_freq=100 \
  --save_freq=2000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=univtac_smolvla_panda9_b32_s8000
