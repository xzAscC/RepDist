#!/usr/bin/env bash
# Train layer 1 on the full 50k-prompt store. Safe to re-run: the trainer
# resumes from checkpoints/layer1-50k/layer_01/latest.pt automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/layer1-50k
exec uv run repdist --config configs/layer1-50k.yaml train
