# Experiment protocol

1. Stream `allenai/Dolci-Think-SFT-7B`.
2. Keep messages through the last user turn. Drop assistant traces.
3. Apply the Olmo chat template with `add_generation_prompt=True`.
4. Truncate to 1024 tokens. Left-pad so the last prompt token is at index `-1`.
5. Run `allenai/Olmo-3-7B-Think` in bf16. In one forward pass save last-prompt-token states at layers `[1, 8, 16, 24, 32]` (`hidden_states[k][:, -1, :]`).
6. Write 5k val, then 5k test, then 10k train shards. Never rewrite val/test.
7. Fit \(\mu_{\mathrm{train}}\) and scalar \(s_{\mathrm{train}}\) on train only.
8. Train an independent cosine DDPM MLP on each layer. Log train/val MSE. Save `latest` / `best` / step checkpoints per layer.
9. If val loss plateaus, extract another train chunk and continue (resume-safe).
10. Freeze the fitted normalizer for the entire run. Generate as many samples as test points in normalized space.
11. Compare real / diffusion / independent standard-normal \(\mathcal{N}(0,I)\) samples via spectrum, \(d_{\mathrm{eff}}\), PCA rank 32, and SWD. Do not fit the baseline covariance.
12. Use validation only for tuning. The pilot is 2,500 steps with no streaming and 250-step eval/checkpoint/diagnostic cadence; go forward only when diagnostics are finite and final/max normalized reverse RMS is at most 3. Hard-abort at RMS 25.
13. For the full corrected run use 50,000 steps, the original cosine schedule and 1024-wide SiLU MLP, 500-step cadence, and allow streaming. If stable but high-timestep MSE is poor, retry once at half the learning rate; if RMS hard-fails twice, stop.
14. Keep corrected artifacts isolated under `runs/rankfix-pilot/` or `runs/rankfix/`. Old checkpoints are incompatible and must never be resumed.
15. The main run extracts probe layers `[1, 8, 16, 24, 32]` in one forward pass so samples stay aligned, then trains the same 8196-wide ambient SiLU MLP independently on each layer.
