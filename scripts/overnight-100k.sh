#!/usr/bin/env bash
# Unattended 50k -> 100k continuation, run as a single gpu-queue job:
#   train (5 layers in parallel) -> per-layer val-SWD checkpoint selection
#   -> test eval on winners -> cross-layer comparison -> notebook inputs
#   -> re-execute notebooks -> morning summary.
# Safe to re-run: training resumes from latest.pt and selection re-evaluates
# whatever step checkpoints exist.
set -euo pipefail
cd "$(dirname "$0")/.."

LAYERS=(1 8 16 24 32)
RUN_LOG=logs/layers-50k
mkdir -p "$RUN_LOG"
echo "=== overnight-100k start $(date -Is) ==="

# Keep the 50k-step candidate for layers whose runs never wrote step files.
for l in 01 08; do
    d="checkpoints/layers-50k/layer_$l"
    if [ -f "$d/latest.pt" ] && [ ! -f "$d/step_0050000.pt" ]; then
        cp -n "$d/latest.pt" "$d/step_0050000.pt"
        echo "snapshotted layer_$l 50k-step candidate"
    fi
done

echo "--- train to 100k (parallel per layer) $(date -Is)"
pids=()
for l in "${LAYERS[@]}"; do
    uv run repdist --config configs/layers-50k.yaml --layer "$l" train \
        > "$RUN_LOG/train-100k-layer$l.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "TRAINING FAILED; see $RUN_LOG/train-100k-layer*.log" >&2
    exit 1
fi

echo "--- select best checkpoint by val SWD $(date -Is)"
for l in "${LAYERS[@]}"; do
    uv run repdist --config configs/layers-50k.yaml --layer "$l" select-swd --apply
done

echo "--- cross-layer comparison $(date -Is)"
uv run repdist --config configs/layers-50k.yaml compare-layers

echo "--- regenerate notebook inputs $(date -Is)"
uv run repdist --config configs/layers-50k.yaml report

echo "--- re-execute notebooks $(date -Is)"
export MPLBACKEND=Agg
for nb in loss swd effective_rank eigenspace; do
    uv run jupyter nbconvert --to notebook --execute --inplace "notebooks/$nb.ipynb"
done

uv run python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("logs/layers-50k")
lines = [f"# layers-50k -> 100k overnight summary", "",
         f"generated {datetime.now(timezone.utc).isoformat()}", ""]
lines += ["| layer | best step | val SWD | val SWD random | beats random |",
          "| ---: | ---: | ---: | ---: | --- |"]
for layer in (1, 8, 16, 24, 32):
    sel = json.loads((root / f"layer_{layer:02d}" / "swd_selection.json").read_text())
    lines.append(
        f"| {layer} | {sel['best_step']} | {sel['best_swd_val']:.4f} "
        f"| {sel['swd_val_random']:.4f} | {'yes' if sel['beats_random'] else 'no'} |"
    )
comparison = json.loads((root / "comparison" / "summary.json").read_text())
lines += ["", "## test-space comparison (selected checkpoints vs N(0,I))", ""]
for layer, metrics in sorted(comparison["metrics"].items()):
    beats = comparison["beats_random_swd"][layer]
    lines.append(
        f"- layer {layer}: SWD diff {metrics['swd_real_diffusion']:.4f} vs random "
        f"{metrics['swd_real_random']:.4f} — {'beats' if beats else 'loses to'} random"
    )
out = root / "overnight_summary.md"
out.write_text("\n".join(lines) + "\n")
print(f"summary -> {out}")
PY

echo "=== overnight-100k done $(date -Is) ==="
