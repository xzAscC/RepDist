# AGENTS.md

This repository trains a DDPM on last-prompt-token hidden states from `allenai/Olmo-3-7B-Think` (layer 16) using `allenai/Dolci-Think-SFT-7B`.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run repdist --config configs/smoke.yaml all
uv run repdist --config configs/default.yaml extract
uv run repdist --config configs/default.yaml train
uv run repdist --config configs/default.yaml eval
```

Training resumes from `checkpoints/latest.pt` unless `--no-resume` is passed.

## Layout

| Path | Role |
| --- | --- |
| `src/repdist/` | Library: schedule, MLP, extract, train, eval |
| `configs/` | YAML experiment configs |
| `tests/` | CPU unit tests; no 7B model required |
| `data/hidden_states/` | Cached val/test tensors and train shards |
| `checkpoints/` | `latest.pt`, `best.pt`, `step_XXXXXXX.pt` |
| `logs/train.jsonl` | Train/val diffusion loss records |
| `outputs/figures/` | Spectrum, PCA, loss plots |
| `outputs/metrics/` | `eval.json` and spectra |
| `notes/` | Experiment protocol |

## Conventions

- Match the memo: cosine DDPM (`T=1000`, `s=0.008`, `β≤0.999`), two-layer SiLU MLP (1024-d hidden), global scalar normalization, DDPM reverse with no noise at `t=1`.
- Layer index `16` is `output.hidden_states[16]` (embeddings at 0, after block 16).
- Val/test are extracted first and never overwritten. Extra train shards are appended only on plateau.
- Do not commit tensors, checkpoints, or logs. Do not add the nested slides directory.
- Keep new code CPU-testable. Real OLMo extraction is gated behind `configs/default.yaml`.

## Checks before claiming done

1. `uv run pytest`
2. Smoke: `uv run repdist --config configs/smoke.yaml all`
3. If training: `checkpoints/latest.pt` and `logs/train.jsonl` exist and grow
