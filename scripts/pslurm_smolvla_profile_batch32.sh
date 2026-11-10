#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/lerobot
export HTTP_PROXY="http://10.19.125.92:7890"
export HTTPS_PROXY="http://10.19.125.92:7890"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
unset ALL_PROXY all_proxy

DATASET_ROOT=/root/datasets/univtac/pull_out_key_v30
OUTPUT_DIR="/tmp/lerobot_smolvla_batch32_profile_${PSLURM_JOB_ID:-manual}"
TRAIN_LOG="$OUTPUT_DIR/train.log"
MEMORY_LOG="$OUTPUT_DIR/nvidia-smi-memory.csv"

.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch

original = json.loads(Path('/root/datasets/univtac/pull_out_key/meta/info.json').read_text())
staged = json.loads(Path('/root/datasets/univtac/pull_out_key_v30/meta/info.json').read_text())
assert original['codebase_version'] == 'v2.1', original['codebase_version']
assert staged['codebase_version'] == 'v3.0', staged['codebase_version']
assert staged['features']['observation.state']['shape'] == [9]
assert staged['features']['action']['shape'] == [9]
assert torch.cuda.is_available(), 'CUDA unavailable; pslurm selected a GPU with an incompatible driver.'
print('[profile] preserving original v2.1; using staged v3.0 dataset')
print('[profile] task labels are unchanged: Empty')
print('[profile] GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$OUTPUT_DIR"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
(
  printf 'timestamp,memory_used_mib,memory_total_mib\n'
  while true; do
    values=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    printf '%s,%s\n' "$(date -Is)" "$values"
    sleep 0.2
  done
) > "$MEMORY_LOG" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  awk -F, 'NR > 1 && $2 + 0 > max { max=$2 + 0; row=$0 } END { if (NR > 1) print "[profile] max nvidia-smi memory sample:", row; else print "[profile] no nvidia-smi samples captured" }' "$MEMORY_LOG"
  echo "[profile] detailed nvidia-smi samples: $MEMORY_LOG"
  echo "[profile] training log: $TRAIN_LOG"
}
trap cleanup EXIT

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
  --steps=1 \
  --log_freq=1 \
  --save_checkpoint=false \
  --output_dir="$OUTPUT_DIR/run" \
  --job_name=smolvla_batch32_memory_profile 2>&1 | tee "$TRAIN_LOG"

echo "[profile] trainer-reported peak allocated memory is in the mem_gb metric above."
