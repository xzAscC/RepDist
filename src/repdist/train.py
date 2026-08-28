from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from repdist.checkpoint import (
    best_path,
    capture_rng,
    latest_path,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    step_path,
    validate_checkpoint_compatibility,
)
from repdist.config import ExperimentConfig
from repdist.diffusion import (
    diffusion_loss,
    epsilon_diagnostics,
    sample_with_diagnostics,
)
from repdist.extract import ensure_train
from repdist.model import NoisePredictor
from repdist.normalize import Normalizer
from repdist.schedule import CosineSchedule
from repdist.store import HiddenStateStore


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_at(step: int, base_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return (
        base_lr
        * 0.5
        * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
    )


@torch.no_grad()
def eval_loss(
    model: NoisePredictor,
    data: torch.Tensor,
    schedule: CosineSchedule,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size].to(device)
        loss = diffusion_loss(model, batch, schedule)
        total += float(loss.item()) * batch.shape[0]
        count += batch.shape[0]
    return total / max(count, 1)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _valid_diagnostic_timesteps(
    configured: list[int], schedule: CosineSchedule
) -> list[int]:
    valid = [t for t in configured if 0 <= t < schedule.timesteps]
    return valid or [schedule.timesteps - 1]


def train(cfg: ExperimentConfig, resume: bool = True) -> dict:
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    store = HiddenStateStore(cfg.paths.data)
    train_hidden = ensure_train(cfg, store, device)
    val_hidden = store.load_split("val")

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.paths.logs) / "train.jsonl"
    Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)

    normalizer_path = Path(cfg.paths.normalizer)
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    resuming = resume and latest_path(ckpt_dir).exists()
    payload = None
    if resuming:
        payload = load_checkpoint(latest_path(ckpt_dir), map_location=device)
        normalizer = Normalizer.from_state_dict(payload["normalizer"])
        torch.save(normalizer.state_dict(), normalizer_path)
    elif normalizer_path.exists():
        normalizer = Normalizer.from_state_dict(
            torch.load(normalizer_path, map_location="cpu", weights_only=False)
        )
    else:
        normalizer = Normalizer.fit(train_hidden)
        torch.save(normalizer.state_dict(), normalizer_path)

    x_train = normalizer.encode(train_hidden)
    x_val = normalizer.encode(val_hidden)
    data_dim = x_train.shape[1]

    model = NoisePredictor(
        data_dim,
        hidden_dim=cfg.diffusion.hidden_dim,
        time_embed_dim=cfg.diffusion.time_embed_dim,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.diffusion.lr,
        weight_decay=cfg.diffusion.weight_decay,
    )
    schedule = CosineSchedule(
        cfg.diffusion.timesteps,
        cosine_s=cfg.diffusion.cosine_s,
        beta_max=cfg.diffusion.beta_max,
        device=device,
    )

    model_spec = {
        "architecture": "NoisePredictor",
        "data_dim": data_dim,
        "hidden_dim": cfg.diffusion.hidden_dim,
        "time_embed_dim": cfg.diffusion.time_embed_dim,
        "full_rank_skip": True,
    }
    schedule_spec = {
        "timesteps": cfg.diffusion.timesteps,
        "cosine_s": cfg.diffusion.cosine_s,
        "beta_max": cfg.diffusion.beta_max,
    }

    step = 0
    best_val = float("inf")
    patience_left = cfg.diffusion.patience
    if resuming:
        assert payload is not None
        validate_checkpoint_compatibility(
            payload,
            model_spec=model_spec,
            schedule_spec=schedule_spec,
            normalizer_fingerprint=normalizer.fingerprint(),
        )
        x_train = normalizer.encode(train_hidden)
        x_val = normalizer.encode(val_hidden)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        best_val = float(payload["best_val"])
        patience_left = int(payload["patience_left"])
        if payload.get("rng"):
            restore_rng(payload["rng"])
        print(f"resumed from step {step}")

    def make_loader(data: torch.Tensor) -> DataLoader:
        batch_size = min(cfg.diffusion.batch_size, max(int(data.shape[0]), 1))
        return DataLoader(
            TensorDataset(data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=data.shape[0] >= 2 * batch_size,
        )

    loader = make_loader(x_train)
    iterator = iter(loader)
    model.train()
    pbar = tqdm(initial=step, total=cfg.diffusion.max_steps, desc="train")
    last_val = None

    while step < cfg.diffusion.max_steps:
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (batch,) = next(iterator)
        batch = batch.to(device)
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(
                step,
                cfg.diffusion.lr,
                cfg.diffusion.warmup_steps,
                cfg.diffusion.max_steps,
            )
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, batch, schedule)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.diffusion.grad_clip)
        optimizer.step()
        step += 1
        pbar.update(1)

        if step % cfg.diffusion.log_every == 0:
            record = {
                "step": step,
                "split": "train",
                "loss": float(loss.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "n_train": int(x_train.shape[0]),
            }
            _append_jsonl(log_path, record)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % cfg.diffusion.eval_every == 0 or step == cfg.diffusion.max_steps:
            val = eval_loss(model, x_val, schedule, cfg.diffusion.batch_size, device)
            last_val = val
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "split": "val",
                    "loss": val,
                    "n_train": int(x_train.shape[0]),
                },
            )
            diagnostic = None
            diagnostics_enabled = cfg.diffusion.diagnostic_every > 0 and (
                step % cfg.diffusion.diagnostic_every == 0
                or step == cfg.diffusion.max_steps
            )
            if diagnostics_enabled:
                diagnostic_timesteps = _valid_diagnostic_timesteps(
                    cfg.diffusion.diagnostic_timesteps, schedule
                )
                diagnostic_x = x_val[: cfg.diffusion.diagnostic_sample_count].to(device)
                epsilon = epsilon_diagnostics(
                    model,
                    diagnostic_x,
                    schedule,
                    diagnostic_timesteps,
                    seed=cfg.seed,
                )
                reverse = sample_with_diagnostics(
                    model,
                    n=cfg.diffusion.diagnostic_sample_count,
                    dim=data_dim,
                    schedule=schedule,
                    device=device,
                    batch_size=cfg.diffusion.batch_size,
                    generator=torch.Generator(device=device).manual_seed(cfg.seed),
                    selected_steps=sorted(set(diagnostic_timesteps + [0])),
                )
                diagnostic = {
                    "step": step,
                    "split": "diagnostic",
                    "epsilon_mse": epsilon["mse"],
                    "zero_predictor_mse": epsilon["zero_predictor_mse"],
                    "reverse_rms": {str(k): v for k, v in reverse.reverse_rms.items()},
                    "final_rms": reverse.final_rms,
                    "max_rms": reverse.max_rms,
                    "finite": reverse.finite,
                    "n_samples": cfg.diffusion.diagnostic_sample_count,
                }
                _append_jsonl(log_path, diagnostic)
                if (
                    not reverse.finite
                    or not torch.isfinite(
                        torch.tensor([reverse.final_rms, reverse.max_rms])
                    ).all()
                ):
                    raise RuntimeError(f"nonfinite reverse trajectory at step {step}")
                if reverse.max_rms >= cfg.diffusion.hard_abort_rms:
                    raise RuntimeError(
                        f"reverse trajectory RMS {reverse.max_rms:.4g} reached hard-abort "
                        f"threshold {cfg.diffusion.hard_abort_rms:.4g} at step {step}"
                    )
            healthy = diagnostic is None or (
                diagnostic["finite"]
                and diagnostic["final_rms"] <= cfg.diffusion.max_healthy_final_rms
                and diagnostic["max_rms"] <= cfg.diffusion.max_healthy_final_rms
            )
            improved = healthy and val < best_val - cfg.diffusion.min_delta
            if improved:
                best_val = val
                patience_left = cfg.diffusion.patience
                save_checkpoint(
                    best_path(ckpt_dir),
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler_state=None,
                    normalizer=normalizer,
                    best_val=best_val,
                    patience_left=patience_left,
                    model_spec=model_spec,
                    schedule_spec=schedule_spec,
                )
            else:
                patience_left -= 1
            model.train()

            if (
                patience_left <= 0
                and cfg.diffusion.stream_on_plateau
                and store.n_train() < cfg.extract.max_train_samples
            ):
                target = min(
                    store.n_train() + cfg.extract.n_train_stream_chunk,
                    cfg.extract.max_train_samples,
                )
                train_hidden = ensure_train(cfg, store, device, n_target=target)
                x_train = normalizer.encode(train_hidden)
                x_val = normalizer.encode(store.load_split("val"))
                loader = make_loader(x_train)
                iterator = iter(loader)
                patience_left = cfg.diffusion.patience
                _append_jsonl(
                    log_path,
                    {
                        "step": step,
                        "event": "stream_more_data",
                        "n_train": int(x_train.shape[0]),
                    },
                )

        if step % cfg.diffusion.ckpt_every == 0 or step == cfg.diffusion.max_steps:
            rng = capture_rng()
            save_checkpoint(
                latest_path(ckpt_dir),
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler_state=None,
                normalizer=normalizer,
                best_val=best_val,
                patience_left=patience_left,
                rng_state=rng,
                model_spec=model_spec,
                schedule_spec=schedule_spec,
            )
            save_checkpoint(
                step_path(ckpt_dir, step),
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler_state=None,
                normalizer=normalizer,
                best_val=best_val,
                patience_left=patience_left,
                rng_state=rng,
                model_spec=model_spec,
                schedule_spec=schedule_spec,
            )

    pbar.close()
    return {"step": step, "best_val": best_val, "last_val": last_val}
