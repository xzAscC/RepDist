from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from tqdm import tqdm

from repdist.config import ExperimentConfig
from repdist.store import HiddenStateStore


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def prompt_messages(messages: list[dict]) -> list[dict]:
    """Keep messages through the last user turn; drop assistant traces."""
    prompt: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            break
        prompt.append({"role": role, "content": message.get("content", "")})
    if not prompt:
        raise ValueError("example has no user prompt")
    return prompt


def synthetic_hidden_states(
    n: int,
    dim: int,
    rank: int,
    seed: int,
) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    factors = torch.randn(dim, rank, generator=g)
    z = torch.randn(n, rank, generator=g)
    noise = 0.05 * torch.randn(n, dim, generator=g)
    return z @ factors.T + noise


def _iter_dataset(cfg: ExperimentConfig, skip: int) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        cfg.extract.dataset_name,
        split=cfg.extract.dataset_split,
        streaming=True,
    )
    for i, row in enumerate(ds):
        if i < skip:
            continue
        yield row


def _load_lm(cfg: ExperimentConfig, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.extract.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.extract.model_name,
        torch_dtype=_dtype(cfg.extract.dtype),
        device_map="auto" if device == "cuda" else None,
        attn_implementation=cfg.extract.attn_implementation,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _encode_batch(
    tokenizer,
    prompts: list[list[dict]],
    max_length: int,
    device: torch.device | str,
):
    texts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        for messages in prompts
    ]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.inference_mode()
def _extract_batch(model, batch: dict, layer: int) -> Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = out.hidden_states[layer]
    return hidden[:, -1, :].float().cpu()


def extract_count(
    cfg: ExperimentConfig,
    store: HiddenStateStore,
    n: int,
    *,
    split: str,
    device: str,
    seed: int,
    tokenizer=None,
    model=None,
) -> Tensor:
    if cfg.extract.model_name == "synthetic":
        start = store.manifest.get("cursor", 0)
        hidden = synthetic_hidden_states(
            n,
            cfg.extract.synthetic_dim,
            cfg.extract.synthetic_rank,
            seed + start,
        )
        cursor = start + n
        if split in {"val", "test"}:
            store.save_split(split, hidden, cursor)
        else:
            store.append_train(hidden, cursor)
        return hidden

    owns_model = model is None
    if tokenizer is None or model is None:
        tokenizer, model = _load_lm(cfg, device)
    model_device = next(model.parameters()).device
    skip = int(store.manifest.get("cursor", 0))
    batch_prompts: list[list[dict]] = []
    collected: list[Tensor] = []
    seen = 0
    cursor = skip
    iterator = _iter_dataset(cfg, skip)
    pbar = tqdm(total=n, desc=f"extract-{split}")
    for row in iterator:
        cursor += 1
        messages = row.get("messages")
        if not messages:
            continue
        try:
            batch_prompts.append(prompt_messages(list(messages)))
        except ValueError:
            continue
        if len(batch_prompts) < cfg.extract.batch_size:
            continue
        enc = _encode_batch(
            tokenizer, batch_prompts, cfg.extract.max_prompt_tokens, model_device
        )
        hidden = _extract_batch(model, enc, cfg.extract.layer)
        collected.append(hidden)
        seen += hidden.shape[0]
        pbar.update(hidden.shape[0])
        batch_prompts = []
        if seen >= n:
            break
    if batch_prompts and seen < n:
        enc = _encode_batch(
            tokenizer, batch_prompts, cfg.extract.max_prompt_tokens, model_device
        )
        hidden = _extract_batch(model, enc, cfg.extract.layer)
        collected.append(hidden)
        seen += hidden.shape[0]
        pbar.update(hidden.shape[0])
    pbar.close()
    if not collected:
        raise RuntimeError("extracted zero hidden states")
    hidden = torch.cat(collected, dim=0)[:n]
    if split in {"val", "test"}:
        store.save_split(split, hidden, cursor)
    else:
        store.append_train(hidden, cursor)
    if owns_model:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return hidden


def ensure_heldout(
    cfg: ExperimentConfig,
    store: HiddenStateStore,
    device: str,
    tokenizer=None,
    model=None,
) -> None:
    if store.has_val_test():
        return
    if not store.val_path().exists():
        extract_count(
            cfg,
            store,
            cfg.extract.n_val,
            split="val",
            device=device,
            seed=cfg.seed,
            tokenizer=tokenizer,
            model=model,
        )
    if not store.test_path().exists():
        extract_count(
            cfg,
            store,
            cfg.extract.n_test,
            split="test",
            device=device,
            seed=cfg.seed + 1,
            tokenizer=tokenizer,
            model=model,
        )


def ensure_train(
    cfg: ExperimentConfig,
    store: HiddenStateStore,
    device: str,
    n_target: int | None = None,
) -> Tensor:
    tokenizer = model = None
    owns_model = False
    if cfg.extract.model_name != "synthetic":
        need_model = (not store.has_val_test()) or (
            store.n_train()
            < min(
                n_target or cfg.extract.n_train_initial, cfg.extract.max_train_samples
            )
        )
        if need_model:
            tokenizer, model = _load_lm(cfg, device)
            owns_model = True
    try:
        ensure_heldout(cfg, store, device, tokenizer=tokenizer, model=model)
        target = n_target or cfg.extract.n_train_initial
        target = min(target, cfg.extract.max_train_samples)
        have = store.n_train()
        while have < target:
            chunk = min(cfg.extract.n_train_stream_chunk, target - have)
            extract_count(
                cfg,
                store,
                chunk,
                split="train",
                device=device,
                seed=cfg.seed + 2 + have,
                tokenizer=tokenizer,
                model=model,
            )
            have = store.n_train()
    finally:
        if owns_model:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    return store.load_train()
