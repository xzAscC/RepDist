# Rank-fix run summary

Corrected full-rank-skip DDPM (checkpoint `best.pt`, step 40500), 50k steps, 50k streamed train
hidden states, frozen train normalizer, cosine schedule with `beta_max=0.99` (still `β ≤ 0.999`
per memo), lr `1e-4`.

| Metric | Real | Diffusion | Random `N(0,I)` | Real–real split |
| --- | ---: | ---: | ---: | ---: |
| `d_eff` (full, 4096-d, normalized) | 12.01 | 14.6 | 2250.9 | — |
| `d_eff` (PCA-32 subspace) | 3.54 | 2.16 | 31.80 | 3.62 |
| SWD-2 (full space) | — | 1.01 | 0.46 | 0.087 |
| SWD-2 (PCA subspace) | — | 11.10 | 7.86 | 0.61 |
| PCA covariance rel. error | — | 0.66 | 0.998 | 0.065 |

- Validation diffusion loss: 0.808 (failed legacy run) → **0.371**.
- Reverse sampling: exploding (RMS ≈ 17,600 normalized) → **bounded ≈ 1.6–1.7**.
- Diffusion beats the random baseline on every geometry metric (effective rank,
  PCA covariance error 0.66 vs 0.998) but still loses on SWD due to systematic
  over-dispersion (generated RMS ≈ 1.7 vs 1.0). Known next step: variance-calibrated
  sampling or low-t loss weighting.

Artifacts: figures `*.png`, full metrics `eval.json`. Raw tensors/checkpoints/logs stay under
`runs/rankfix/` (untracked).
