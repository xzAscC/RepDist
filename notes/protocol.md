# Experiment protocol

1. Stream `allenai/Dolci-Think-SFT-7B`.
2. Keep messages through the last user turn. Drop assistant traces.
3. Apply the Olmo chat template with `add_generation_prompt=True`.
4. Truncate to 1024 tokens. Left-pad so the last prompt token is at index `-1`.
5. Run `allenai/Olmo-3-7B-Think` in bf16. Save `hidden_states[16][:, -1, :]`.
6. Write 5k val, then 5k test, then 10k train shards. Never rewrite val/test.
7. Fit \(\mu_{\mathrm{train}}\) and scalar \(s_{\mathrm{train}}\) on train only.
8. Train the cosine DDPM MLP. Log train/val MSE. Save `latest` / `best` / step checkpoints.
9. If val loss plateaus, extract another train chunk and continue (resume-safe).
10. Generate as many samples as test points. Compare real / diffusion / Gaussian via spectrum, \(d_{\mathrm{eff}}\), PCA, SWD.
