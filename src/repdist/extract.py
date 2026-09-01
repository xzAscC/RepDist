from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from tqdm import tqdm

from repdist.config import ExperimentConfig, layer_store_root, resolved_layers
from repdist.data import HiddenStateStore


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


def layer_stores(
    cfg: ExperimentConfig, store: HiddenStateStore | None = None
) -> dict[int, HiddenStateStore]:
    layers = resolved_layers(cfg)
    if len(layers) == 1:
        if store is None:
            store = HiddenStateStore(cfg.paths.data)
        return {layers[0]: store}
    return {
        layer: HiddenStateStore(layer_store_root(cfg.paths.data, layer, len(layers)))
        for layer in layers
    }


def _shared_cursor(stores: dict[int, HiddenStateStore]) -> int:
    cursors = [int(item.manifest.get("cursor", 0)) for item in stores.values()]
    if any(cursor != cursors[0] for cursor in cursors):
        raise RuntimeError("layer stores have diverged dataset cursors")
    return cursors[0]


def _shared_n_train(stores: dict[int, HiddenStateStore]) -> int:
    counts = [item.n_train() for item in stores.values()]
    if any(count != counts[0] for count in counts):
        raise RuntimeError("layer stores have diverged train counts")
    return counts[0]


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
def _extract_batch(model, batch: dict, layers: list[int]) -> dict[int, Tensor]:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = out.hidden_states
    n_hidden = len(hidden_states)
    extracted: dict[int, Tensor] = {}
    for layer in layers:
        if layer < 0 or layer >= n_hidden:
            raise ValueError(
                f"layer {layer} is outside hidden_states length {n_hidden}"
            )
        extracted[layer] = hidden_states[layer][:, -1, :].float().cpu()
    return extracted


def _write_split(
    stores: dict[int, HiddenStateStore],
    hidden: dict[int, Tensor],
    split: str,
    cursor: int,
) -> None:
    for layer, tensor in hidden.items():
        if split in {"val", "test"}:
            stores[layer].save_split(split, tensor, cursor)
        else:
            stores[layer].append_train(tensor, cursor)


def extract_count(
    cfg: ExperimentConfig,
    store: HiddenStateStore | None,
    n: int,
    *,
    split: str,
    device: str,
    seed: int,
    tokenizer=None,
    model=None,
) -> Tensor:
    stores = layer_stores(cfg, store)
    layers = list(stores)
    primary = cfg.extract.layer if cfg.extract.layer in stores else layers[0]
    if cfg.extract.model_name == "synthetic":
        start = _shared_cursor(stores)
        hidden = {
            layer: synthetic_hidden_states(
                n,
                cfg.extract.synthetic_dim,
                cfg.extract.synthetic_rank,
                seed + start + 17 * layer,
            )
            for layer in layers
        }
        cursor = start + n
        _write_split(stores, hidden, split, cursor)
        return hidden[primary]

    owns_model = model is None
    if tokenizer is None or model is None:
        tokenizer, model = _load_lm(cfg, device)
    model_device = next(model.parameters()).device
    skip = _shared_cursor(stores)
    batch_prompts: list[list[dict]] = []
    collected: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    seen = 0
    cursor = skip
    iterator = _iter_dataset(cfg, skip)
    pbar = tqdm(total=n, desc=f"extract-{split}-L{','.join(str(x) for x in layers)}")
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
        batch_hidden = _extract_batch(model, enc, layers)
        for layer, tensor in batch_hidden.items():
            collected[layer].append(tensor)
        seen += next(iter(batch_hidden.values())).shape[0]
        pbar.update(next(iter(batch_hidden.values())).shape[0])
        batch_prompts = []
        if seen >= n:
            break
    if batch_prompts and seen < n:
        enc = _encode_batch(
            tokenizer, batch_prompts, cfg.extract.max_prompt_tokens, model_device
        )
        batch_hidden = _extract_batch(model, enc, layers)
        for layer, tensor in batch_hidden.items():
            collected[layer].append(tensor)
        seen += next(iter(batch_hidden.values())).shape[0]
        pbar.update(next(iter(batch_hidden.values())).shape[0])
    pbar.close()
    if not collected[primary]:
        raise RuntimeError("extracted zero hidden states")
    hidden = {layer: torch.cat(parts, dim=0)[:n] for layer, parts in collected.items()}
    _write_split(stores, hidden, split, cursor)
    if owns_model:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return hidden[primary]


def ensure_heldout(
    cfg: ExperimentConfig,
    store: HiddenStateStore | None,
    device: str,
    tokenizer=None,
    model=None,
) -> None:
    stores = layer_stores(cfg, store)
    have = [item.has_val_test() for item in stores.values()]
    if all(have):
        return
    if any(have) and not all(have):
        raise RuntimeError("layer stores have partial val/test caches")
    if not all(path.val_path().exists() for path in stores.values()):
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
    if not all(path.test_path().exists() for path in stores.values()):
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
    store: HiddenStateStore | None,
    device: str,
    n_target: int | None = None,
) -> Tensor:
    stores = layer_stores(cfg, store)
    primary = cfg.extract.layer if cfg.extract.layer in stores else next(iter(stores))
    tokenizer = model = None
    owns_model = False
    if cfg.extract.model_name != "synthetic":
        need_model = (not all(item.has_val_test() for item in stores.values())) or (
            _shared_n_train(stores)
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
        have = _shared_n_train(stores)
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
            have = _shared_n_train(stores)
    finally:
        if owns_model:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    return stores[primary].load_train()
