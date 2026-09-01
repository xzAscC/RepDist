#!/usr/bin/env bash
# Finish the layers-50k 100k pipeline after training/selection/eval are done:
# regenerate notebook inputs, re-execute notebooks, write the morning summary.
# Complements scripts/overnight-100k.sh, which already produced swd_selection
# files, winner evals, and the cross-layer comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

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

echo "=== finish-100k-report done $(date -Is) ==="
