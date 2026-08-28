# RepDist

DDPM modeling of hidden-state representations.

Given last-prompt-token hidden states \(\mathbf{h}\) at a fixed transformer layer, this repo asks:

1. Does \(p_H\) have nontrivial low-dimensional geometry (anisotropy, spectral concentration, effective rank)?
2. Can a diffusion model \(p_\theta\) recover that distribution on held-out hidden states, beyond a Gaussian baseline?

The implementation follows the experimental memo in the slides: cosine DDPM, MLP noise predictor, covariance spectra, effective rank, PCA, and sliced Wasserstein-2.

## Setup

Python 3.13+, a CUDA GPU for the 7B extraction (RTX 4090 is enough in bf16), and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev
uv run pytest
```

## Data

| Item | Value |
| --- | --- |
| Model | `allenai/Olmo-3-7B-Think` |
| Dataset | `allenai/Dolci-Think-SFT-7B` (streaming) |
| Layer | 16 (`hidden_states[16]`) |
| Token | last prompt token after chat template (`add_generation_prompt=True`, left padding) |
| Splits | 5k val, 5k test (frozen), 10k train initially |
| Streaming | if val diffusion loss plateaus, append more train hidden states |

User/assistant traces are truncated to the user prompt. Assistant `<think>` traces are never encoded.

## Method

Hidden states are centered with the training mean and divided by a global RMS scale \(s_{\mathrm{train}}\).

Forward process (cosine schedule, \(T=1000\), \(s=0.008\), \(\beta_t \le 0.999\)):

\[
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}
\]

The noise predictor is a two-layer MLP with a 1024-dimensional SiLU hidden layer and a sinusoidal timestep embedding. Training minimizes \(\|\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)\|_2^2\).

Reverse sampling uses the standard DDPM mean and posterior variance \(\tilde\beta_t\), with no extra noise at \(t=1\). Samples are mapped back by \(\tilde{\mathbf{h}} = s_{\mathrm{train}}\tilde{\mathbf{x}}_0 + \boldsymbol{\mu}_{\mathrm{train}}\).

## Commands

```bash
# CPU smoke (synthetic hidden states)
uv run repdist --config configs/smoke.yaml all

# Full experiment
uv run repdist --config configs/default.yaml extract
uv run repdist --config configs/default.yaml train
uv run repdist --config configs/default.yaml eval
```

`train` resumes from `checkpoints/latest.pt` by default. Use `--no-resume` to start over.

## Artifacts

| Directory | Contents |
| --- | --- |
| `data/hidden_states/` | `val.pt`, `test.pt`, `train_*.pt`, `manifest.json`, `normalizer.pt` |
| `checkpoints/` | `latest.pt`, `best.pt`, periodic `step_*.pt` |
| `logs/train.jsonl` | train/val MSE vs step |
| `outputs/figures/` | `spectrum.png`, `pca.png`, `loss.png` |
| `outputs/metrics/eval.json` | \(d_{\mathrm{eff}}\) and SWD vs Gaussian baseline |

## Evaluation

Held-out test hidden states are compared to an equal number of diffusion samples and to \(\mathcal N(\hat\mu_{\mathrm{train}}, \hat\Sigma_{\mathrm{train}})\):

- covariance eigenvalue spectra and effective rank \(d_{\mathrm{eff}}=(\sum\lambda_i)^2 / \sum\lambda_i^2\)
- PCA scatter in a basis fitted on the training set
- sliced Wasserstein-2 over random 1-D projections

Smaller SWD and closer spectra indicate better recovery of \(p_H\).
