#!/usr/bin/env bash
# Train all five probe layers on the full 50k-prompt store. Safe to re-run:
# each layer resumes from checkpoints/layers-50k/layer_XX/latest.pt, and
# finished layers are skipped instantly.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/layers-50k
exec uv run repdist --config configs/layers-50k.yaml train
