cd /root/workspace/lerobot

.venv/bin/lerobot-train \
  --config_path=outputs/train/univtac/lift_bottle/20261125_002/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=30000 \
  --save_freq=10000