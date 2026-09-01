# AGENTS

1. Use TDD: write tests before writing source code.
2. Start work on a new branch. When the work is finished, commit all changes and open a PR. Keep the PR concise and clear; do not add fluff.
3. Use only the folders already listed in the README. Put each file in its corresponding folder. Do not create new folders unless I am unavailable and you cannot ask me.
4. Use uv to manage virtual environments.
5. If you are unsure, ask me first. If you cannot ask me, search the website or documentation. Do not guess.
6. Save figures as PDF first. Do not save as PNG.
7. Keep code clean and concise.
8. `uv run pytest` and `uv run repdist --config configs/smoke.yaml all` must pass before claiming done; tests and the smoke config stay CPU-only. Real OLMo extraction runs only with `configs/default.yaml`.
9. Never resume legacy checkpoints with corrected configs; old (pre-v2) checkpoints are incompatible. Val/test hidden-state stores are frozen and never rewritten.
10. Do not commit tensors, checkpoints, logs, or figures. Do not add the nested slides directory.
