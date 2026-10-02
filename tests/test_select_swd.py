import json
import math
from pathlib import Path

from repdist.config import ExperimentConfig
from repdist.select import candidate_steps, select_best_swd
from repdist.train import train


def _smoke_cfg(tmp_path: Path, max_steps: int) -> ExperimentConfig:
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "data")
    cfg.paths.checkpoints = str(tmp_path / "checkpoints")
    cfg.paths.logs = str(tmp_path / "logs")
    cfg.paths.figs = str(tmp_path / "figs")
    cfg.paths.normalizer = str(tmp_path / "normalizer.pt")
    cfg.device = "cpu"
    cfg.diffusion.max_steps = max_steps
    cfg.diffusion.eval_every = max(max_steps // 2, 1)
    cfg.diffusion.ckpt_every = max(max_steps // 2, 1)
    cfg.diffusion.log_every = 1
    cfg.diffusion.warmup_steps = 1
    cfg.diffusion.diagnostic_sample_count = 2
    cfg.diffusion.diagnostic_every = 50
    return cfg


def test_candidate_steps_discovers_and_sorts_step_checkpoints(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=6)
    cfg.diffusion.ckpt_every = 3
    cfg.diffusion.step_ckpt_every = 3
    train(cfg, resume=False)
    assert candidate_steps(cfg.paths.checkpoints) == [3, 6]


def test_select_best_swd_ranks_candidates_and_writes_json(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=6)
    cfg.diffusion.ckpt_every = 3
    cfg.diffusion.step_ckpt_every = 3
    train(cfg, resume=False)

    result = select_best_swd(cfg)

    steps = [row["step"] for row in result["candidates"]]
    swds = [row["swd_val"] for row in result["candidates"]]
    assert sorted(steps) == [3, 6]
    assert swds == sorted(swds)
    assert result["best_step"] in (3, 6)
    assert result["best_swd_val"] == swds[0]
    assert math.isfinite(result["swd_val_random"])
    assert isinstance(result["beats_random"], bool)
    saved = json.loads((Path(cfg.paths.logs) / "swd_selection.json").read_text())
    assert saved["best_step"] == result["best_step"]
    assert saved["split"] == "val"
    assert saved["n_candidates"] == 2


def test_select_best_swd_honors_explicit_steps(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=6)
    cfg.diffusion.ckpt_every = 3
    cfg.diffusion.step_ckpt_every = 3
    train(cfg, resume=False)

    result = select_best_swd(cfg, steps=[6])

    assert [row["step"] for row in result["candidates"]] == [6]
    assert result["best_step"] == 6


def test_select_best_swd_without_candidates_raises(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=2)
    train(cfg, resume=False)
    for path in Path(cfg.paths.checkpoints).glob("step_*.pt"):
        path.unlink()
    try:
        select_best_swd(cfg)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("selection must fail when no step checkpoints exist")
