# AGENTS.md

This repository trains a DDPM on last-prompt-token hidden states from `allenai/Olmo-3-7B-Think` (layers 1/8/16/24/32) using `allenai/Dolci-Think-SFT-7B`.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run repdist --config configs/smoke.yaml all
uv run repdist --config configs/default.yaml extract
uv run repdist --config configs/default.yaml train
uv run repdist --config configs/default.yaml eval
uv run repdist --config configs/default.yaml --layer 16 train
```

`extract` writes aligned stores for all `extract.layers`. `train` / `eval` loop those layers unless `--layer` is passed. Training resumes from each layer's `checkpoints/layer_XX/latest.pt` unless `--no-resume` is passed.

## Layout

| Path | Role |
| --- | --- |
| `src/repdist/` | Library: schedule, MLP, extract, train, eval |
| `configs/` | YAML experiment configs |
| `tests/` | CPU unit tests; no 7B model required |
| `data/hidden_states/layer_XX/` | Cached val/test tensors and train shards per layer |
| `checkpoints/layer_XX/` | `latest.pt`, `best.pt`, `step_XXXXXXX.pt` |
| `logs/layer_XX/train.jsonl` | Train/val diffusion loss records |
| `outputs/layer_XX/figures/` | Spectrum, PCA, loss plots |
| `outputs/layer_XX/metrics/` | `eval.json` and spectra |
| `notes/` | Experiment protocol |
| `runs/rankfix-pilot/` | Isolated corrected pilot normalizer, checkpoints, logs, and outputs |
| `runs/rankfix/` | Isolated corrected full-run normalizer, checkpoints, logs, and outputs |
| `results/` | Optional cross-layer comparison tables and figures |

## Conventions

- Match the memo: cosine DDPM (`T=1000`, `s=0.008`, `β≤0.999`), two-layer SiLU MLP (8196-d hidden), global scalar normalization, DDPM reverse with no noise at `t=1`.
- Corrected runs use the full-rank skip with zero-initialized residual, checkpoint v2, and a frozen normalizer. The standard-normal `N(0,I)` baseline is independent and is evaluated in normalized space; PCA rank is 32.
- Probe layers are `hidden_states[k]` for `k in {1, 8, 16, 24, 32}` (embeddings at 0, after block k). Extract them in one forward pass so prompts stay aligned; train an independent ambient DDPM per layer.
- Val/test are extracted first and never overwritten. Extra train shards are appended only on plateau.
- Use `configs/rankfix-pilot.yaml` for the 2500-step no-streaming go/no-go run; proceed only with finite diagnostics and normalized reverse RMS ≤3, and stop on hard RMS failures twice. If stable but high-timestep MSE is poor, retry once at half LR. Validation is for tuning only.
- Old checkpoints are incompatible with corrected runs; never resume the legacy 50k run.
- Do not commit tensors, checkpoints, or logs. Do not add the nested slides directory.
- Keep new code CPU-testable. Real OLMo extraction is gated behind `configs/default.yaml`.

## Checks before claiming done

1. `uv run pytest`
2. Smoke: `uv run repdist --config configs/smoke.yaml all`
3. If training: each trained layer's `checkpoints/layer_XX/latest.pt` and `logs/layer_XX/train.jsonl` exist and grow
