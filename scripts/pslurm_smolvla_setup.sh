#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/lerobot

echo "[setup] host=$(hostname)"
echo "[setup] python=$(python --version 2>&1 || true)"

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c \
  'import accelerate, datasets, jsonlines, pandas, pyarrow, transformers' >/dev/null 2>&1; then
  echo "[setup] creating uv environment"
  uv sync --locked --extra smolvla --extra training --extra test
else
  echo "[setup] existing .venv found"
fi

echo "[setup] python=$(.venv/bin/python --version)"
echo "[setup] torch=$(.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')"
.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable on this node; select a node with a compatible NVIDIA driver.')
PY

ORIGINAL_DATASET_ROOT=/root/datasets/univtac/pull_out_key
DATASET_ROOT=/root/datasets/univtac/pull_out_key_v30
if [[ "$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/root/datasets/univtac/pull_out_key_v30/meta/info.json')
if not p.exists():
    print('missing')
else:
    print(json.loads(p.read_text()).get('codebase_version', ''))
PY
)" != "v3.0" ]]; then
  echo "[setup] no verified v3.0 staging dataset at $DATASET_ROOT"
  echo "[setup] original dataset remains at $ORIGINAL_DATASET_ROOT; stage conversion separately before training"
  exit 2
fi

.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
root = Path('/root/datasets/univtac/pull_out_key_v30')
original = Path('/root/datasets/univtac/pull_out_key')
info = json.loads((root / 'meta/info.json').read_text())
original_info = json.loads((original / 'meta/info.json').read_text())
print('[setup] staged dataset version:', info.get('codebase_version'))
print('[setup] original dataset version:', original_info.get('codebase_version'))
print('[setup] state/action:', info['features']['observation.state']['shape'], info['features']['action']['shape'])
if original_info.get('codebase_version') != 'v2.1':
    raise SystemExit('Expected the original dataset path to remain v2.1.')
print('[setup] gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY

