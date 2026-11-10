#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/lerobot
export HTTP_PROXY="http://10.19.125.92:7890"
export HTTPS_PROXY="http://10.19.125.92:7890"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
unset ALL_PROXY all_proxy
.venv/bin/python -m pytest -q \
  tests/policies/smolvla/test_modeling_smolvla.py \
  tests/processor/test_smolvla_processor.py \
  tests/policies/smolvla/test_smolvla_rtc.py \
  tests/scripts/test_convert_dataset_v21_to_v30.py

.venv/bin/python - <<'PY'
from lerobot.datasets import LeRobotDataset
root = '/root/datasets/univtac/pull_out_key_v30'
dataset = LeRobotDataset('univtac/pull_out_key', root=root, download_videos=False)
print('dataset length:', len(dataset))
print('features:', {k: tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__ for k, v in dataset[0].items()})
assert len(dataset) == 10114
assert dataset[0]['observation.state'].shape[-1] == 9
assert dataset[0]['action'].shape[-1] == 9
PY
