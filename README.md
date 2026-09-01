# RepDist

DDPM modeling of Olmo-3 hidden-state distributions.

Given last-prompt-token hidden states \(\mathbf{h}\) at transformer layers 1, 8, 16, 24, and 32 of `allenai/Olmo-3-7B-Think`, this repo asks (1) whether \(p_H\) has nontrivial low-dimensional geometry (anisotropy, spectral concentration, effective rank), and (2) whether a diffusion model \(p_\theta\) recovers it on held-out states beyond a Gaussian baseline.

Requires Python 3.13+, a CUDA GPU for the 7B extraction (an RTX 4090 in bf16 is enough), and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Files Architecture

```text
src/: source code
tests/: test code
notebooks/: jupyter notebooks
scripts/: sh files to run the code
logs/: training logs and eval metrics
figs/: figures for experiments
data/: hidden-state stores and other data
configs/: yaml files for default configs
checkpoints/: model checkpoints and fitted normalizers
README.md: this file
AGENTS.md: agent rules
pyproject.toml: project metadata and build config
.python-version: Python version pin
.gitignore: git ignore rules
.ignore: ! un-ignore gitignored paths so agents can find them
LICENSE: MIT license text
```

Artifacts live under per-run subdirectories: `data/hidden_states/layer_XX/` stores, `checkpoints/<run>/layer_XX/latest.pt` and `normalizer.pt`, `logs/<run>/layer_XX/train.jsonl` and `eval.json`, `figs/<run>/layer_XX/*.pdf`. The default run writes directly under the top-level folders.

## Data

| Item | Value |
| --- | --- |
| Model | `allenai/Olmo-3-7B-Think` (bf16) |
| Dataset | `allenai/Dolci-Think-SFT-7B` (streaming) |
| Layers | `hidden_states[k]` for `k` in {1, 8, 16, 24, 32}, aligned in one forward pass |
| Token | last prompt token; chat template with `add_generation_prompt=True`, left padding, 1024 tokens |
| Splits | 5k val, 5k test (frozen), 10k train initially, streamed up to 50k |

User/assistant traces are truncated to the user prompt; assistant `<think>` traces are never encoded.

## Method

- Normalize: center by the training mean, divide by a global scalar RMS \(s_{\mathrm{train}}\); the normalizer is fitted once and frozen.
- Forward: cosine DDPM (\(T=1000\), \(s=0.008\), \(\beta_t \le 0.999\)), \(\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}\).
- Predictor: two-layer SiLU MLP (8196 hidden) with a sinusoidal time embedding and a full-rank skip whose residual branch is zero-initialized.
- Reverse: standard DDPM mean and posterior variance, no extra noise at \(t=1\); samples map back via \(\tilde{\mathbf{h}} = s_{\mathrm{train}}\tilde{\mathbf{x}}_0 + \boldsymbol{\mu}_{\mathrm{train}}\).
- Eval: covariance spectra, effective rank \(d_{\mathrm{eff}}\), PCA (rank 32, fitted on train), and sliced Wasserstein-2 against both the diffusion samples and an independent \(\mathcal{N}(0,I)\) baseline in normalized space. Smaller SWD and closer spectra mean better recovery.

## Commands

```bash
# CPU smoke (synthetic hidden states)
uv run repdist --config configs/smoke.yaml all

# Full experiment (extract all five layers, then train/eval each)
uv run repdist --config configs/default.yaml extract
uv run repdist --config configs/default.yaml train
uv run repdist --config configs/default.yaml eval

# One layer only
uv run repdist --config configs/default.yaml --layer 16 train

# Memo figures from saved eval tensors
uv run repdist --config configs/default.yaml --layer 16 viz

# Cross-layer comparison over logs/
uv run repdist --config configs/default.yaml compare-layers
```

`train`/`eval`/`viz` loop `extract.layers` unless `--layer` is set. Training resumes from `checkpoints/.../latest.pt` unless `--no-resume` is passed.

## Corrected-run protocol

`configs/rankfix-pilot.yaml` is a 2,500-step go/no-go run (no streaming, 250-step eval/checkpoint/diagnostic cadence, 16 diagnostic samples). Proceed to `configs/rankfix.yaml` (50,000 steps, streaming, 500-step cadence) only if diagnostics stay finite and normalized reverse RMS stays at or below 3; hard-abort at RMS 25. Use validation only for tuning: if stable but high-timestep MSE stays poor, retry once at half the learning rate; if RMS hard-fails twice, stop. Corrected runs reuse the frozen per-layer stores but isolate checkpoints, logs, figures, and normalizers under `checkpoints/rankfix*/`, `logs/rankfix*/`, and `figs/rankfix*/`. Legacy (pre-v2) checkpoints are incompatible and must never be resumed.

## License

MIT. See [LICENSE](LICENSE).
