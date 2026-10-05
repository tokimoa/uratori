"""学習と予測のループ。backbone を替えても同じコードで回し、比較の条件を揃える。"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from uratori.data.schema import Record
from uratori.models.scorer import OptionScorer, resolve_marker
from uratori.serialize import OPTION_MARKER, build_input
from uratori.train.losses import combine_by_type, hard_label_loss, ordinal_loss, soft_label_loss

TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}


@dataclass
class TrainConfig:
    model_id: str
    max_tokens: int = 1024
    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 2e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    soft_weight: float = 0.0  # 教師分布への KL の重み。0 なら確定ラベルだけで学習
    ordinal_weight: float = 0.0  # score 型に足す順序損失の重み
    grad_ckpt: bool = False
    device: str = "mps"
    seed: int = 0
    # 学習済みの checkpoint から始める場合のパス（NLI で先に学習した重みなど）
    init_from: str | None = None
    # 0 より大きければ LoRA（全ての線形層）で学習する。追加した候補マーカーの埋め込み行も学習する
    lora_r: int = 0
    # backbone を読み込むときの dtype。GPU で LoRA を使うときは bf16 にする
    model_dtype: str = "fp32"
    # autocast を使うか（CUDA の bf16 向け）
    autocast: bool = False


@dataclass
class Encoded:
    record: Record
    input_ids: list[int]
    n_options: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build_model(
    model_id: str, device: str, grad_ckpt: bool = False, lora_r: int = 0, model_dtype: str = "fp32"
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    marker, marker_id, added = resolve_marker(tokenizer)
    backbone = AutoModel.from_pretrained(model_id, dtype=_DTYPES[model_dtype])
    if added:
        backbone.resize_token_embeddings(len(tokenizer))
    if grad_ckpt:
        backbone.gradient_checkpointing_enable()
        if lora_r:
            backbone.enable_input_require_grads()
    if lora_r:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules="all-linear",
            # 追加した候補マーカーの埋め込み行だけを学習対象にする
            trainable_token_indices={"embed_tokens": [marker_id]} if added else None,
        )
        backbone = get_peft_model(backbone, config)
    hidden_size = (
        getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    )
    model = OptionScorer(backbone, hidden_size, marker_id).to(device)
    # head は常に fp32 で持つ（backbone が bf16 でも）
    model.head.float()
    return model, tokenizer, marker


def encode_records(
    records: list[Record], tokenizer, marker: str, marker_id: int, max_tokens: int
) -> tuple[list[Encoded], list[str]]:
    """トークン化する。上限を超えた件は切らずに外し、id を返す（呼び出し側が件数を報告する）。"""
    kept, too_long = [], []
    for record in records:
        serialized = build_input(record.state, record.to_question())
        ids = tokenizer(serialized.text.replace(OPTION_MARKER, marker), truncation=False)[
            "input_ids"
        ]
        if len(ids) > max_tokens:
            too_long.append(record.id)
            continue
        if ids.count(marker_id) != serialized.n_options:
            raise ValueError(f"{record.id}: マーカーの数が候補数と合わない")
        kept.append(Encoded(record, ids, serialized.n_options))
    return kept, too_long


def batches(
    items: list[Encoded], batch_size: int, rng: random.Random | None
) -> list[list[Encoded]]:
    """長さの近い件をまとめて、padding の無駄を減らす。rng を渡すと順番を混ぜる。"""
    if rng is None:
        order = sorted(items, key=lambda e: len(e.input_ids))
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    shuffled = items[:]
    rng.shuffle(shuffled)
    chunk = batch_size * 50
    out = []
    for start in range(0, len(shuffled), chunk):
        block = sorted(shuffled[start : start + chunk], key=lambda e: len(e.input_ids))
        out += [block[i : i + batch_size] for i in range(0, len(block), batch_size)]
    rng.shuffle(out)
    return out


def collate(batch: list[Encoded], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    width = max(len(e.input_ids) for e in batch)
    k_max = max(e.n_options for e in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    target = torch.full((len(batch),), -1, dtype=torch.long)
    soft = torch.zeros((len(batch), k_max))
    has_soft = torch.zeros(len(batch), dtype=torch.bool)
    for i, e in enumerate(batch):
        input_ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attention[i, : len(e.input_ids)] = 1
        if e.record.target_index is not None:
            target[i] = e.record.target_index
        if e.record.target_probs is not None:
            soft[i, : e.n_options] = torch.tensor(e.record.target_probs)
            has_soft[i] = True
    type_ids = torch.tensor([TYPE_IDS[e.record.type.value] for e in batch])
    out = {"input_ids": input_ids, "attention_mask": attention, "target": target,
           "soft": soft, "has_soft": has_soft, "type_ids": type_ids}  # fmt: skip
    return {k: v.to(device) for k, v in out.items()}


def compute_loss(logits, option_mask, batch, cfg: TrainConfig) -> torch.Tensor:
    has_hard = batch["target"] >= 0
    per_example = torch.zeros(logits.shape[0], device=logits.device)
    if has_hard.any():
        target = batch["target"].clamp_min(0)
        per_example = per_example + has_hard * hard_label_loss(logits, target)
        if cfg.ordinal_weight > 0:
            is_score = (batch["type_ids"] == TYPE_IDS["score"]) & has_hard
            per_example = per_example + cfg.ordinal_weight * is_score * ordinal_loss(
                logits, option_mask, target
            )
    # 確定ラベルのない件は教師分布だけが頼りなので、soft_weight が 0 でも KL を使う
    use_soft = batch["has_soft"] & ((cfg.soft_weight > 0) | ~has_hard)
    if use_soft.any():
        weight = torch.where(has_hard, cfg.soft_weight, 1.0)
        per_example = per_example + use_soft * weight * soft_label_loss(
            logits, option_mask, batch["soft"]
        )
    return combine_by_type(per_example, batch["type_ids"])


@torch.inference_mode()
def predict(model, items: list[Encoded], pad_id: int, device: str, batch_size: int = 8) -> dict:
    model.eval()
    out: dict[str, list[float]] = {}
    for batch in batches(items, batch_size, None):
        tensors = collate(batch, pad_id, device)
        logits, _ = model(tensors["input_ids"], tensors["attention_mask"])
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for e, row in zip(batch, probs, strict=True):
            out[e.record.id] = row[: e.n_options].tolist()
    return out


ADAPTER_FILE = "adapter_and_head.safetensors"


def _is_trained_key(key: str) -> bool:
    """LoRA で学習する重み（LoRA の行列、head、追加した候補マーカーの埋め込み）かどうか。"""
    return "lora_" in key or key.startswith("head.") or "trainable_tokens" in key


def load_weights(model: OptionScorer, path: Path) -> None:
    """checkpoint の重みを読む。LoRA の checkpoint は学習した分だけなので、土台は読み込み済みの値を使う。"""
    adapter = path / ADAPTER_FILE
    if not adapter.exists():
        model.load_state_dict(load_file(str(path / "model.safetensors")))
        return
    result = model.load_state_dict(load_file(str(adapter)), strict=False)
    lost = [k for k in result.missing_keys if _is_trained_key(k)]
    if result.unexpected_keys or lost:
        raise ValueError(f"重みが合わない（余分 {result.unexpected_keys[:3]}、不足 {lost[:3]}）")


def save_checkpoint(
    model: OptionScorer, tokenizer, marker: str, cfg: TrainConfig, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if cfg.lora_r > 0:
        # LoRA のときは学習した分だけを保存する。土台は model_id から読み直せる
        state = {k: v for k, v in state.items() if _is_trained_key(k)}
        save_file(state, str(out / ADAPTER_FILE))
    else:
        save_file(state, str(out / "model.safetensors"))
    tokenizer.save_pretrained(out / "tokenizer")
    meta = {"marker": marker, "config": asdict(cfg)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, device: str):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    model, tokenizer, marker = build_model(
        cfg["model_id"],
        device,
        lora_r=cfg.get("lora_r", 0),
        model_dtype=cfg.get("model_dtype", "fp32"),
    )
    load_weights(model, path)
    return model.to(device), tokenizer, marker, meta


def write_predictions(preds: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rid, probs in preds.items():
            f.write(json.dumps({"id": rid, "probs": probs}) + "\n")


def train(
    cfg: TrainConfig, train_records: list[Record], dev_records: list[Record], out: Path
) -> dict:
    set_seed(cfg.seed)
    model, tokenizer, marker = build_model(
        cfg.model_id, cfg.device, cfg.grad_ckpt, cfg.lora_r, cfg.model_dtype
    )
    if cfg.init_from:
        load_weights(model, Path(cfg.init_from))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_items, long_train = encode_records(
        train_records, tokenizer, marker, model.marker_id, cfg.max_tokens
    )
    dev_items, long_dev = encode_records(
        dev_records, tokenizer, marker, model.marker_id, cfg.max_tokens
    )

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr},
            {"params": model.head.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_items) / cfg.batch_size)
    total_updates = max(1, math.ceil(steps_per_epoch * cfg.epochs / cfg.grad_accum))
    warmup = max(1, int(total_updates * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda u: (
            (u + 1) / warmup
            if u < warmup
            else max(0.0, (total_updates - u) / max(1, total_updates - warmup))
        ),
    )

    rng = random.Random(cfg.seed)
    log: list[dict] = []
    step, updates, started = 0, 0, time.perf_counter()
    total_steps = math.ceil(steps_per_epoch * cfg.epochs)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        for batch in batches(train_items, cfg.batch_size, rng):
            if step >= total_steps:
                break
            tensors = collate(batch, pad_id, cfg.device)
            with torch.autocast(
                cfg.device.split(":")[0], dtype=torch.bfloat16, enabled=cfg.autocast
            ):
                logits, option_mask = model(tensors["input_ids"], tensors["attention_mask"])
            loss = compute_loss(logits, option_mask, tensors, cfg)
            (loss / cfg.grad_accum).backward()
            step += 1
            if step % cfg.grad_accum == 0 or step == total_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if step % 20 == 0 or step == total_steps:
                log.append(
                    {"step": step, "loss": loss.item(), "sec": time.perf_counter() - started}
                )

    preds = predict(model, dev_items, pad_id, cfg.device, cfg.batch_size)
    save_checkpoint(model, tokenizer, marker, cfg, out)
    write_predictions(preds, out / "dev_pred.jsonl")
    summary = {
        "train_examples": len(train_items),
        "skipped_too_long": len(long_train),
        # 上限を超えた件は切らずに外し、件数を残す
        "dev_skipped_too_long": len(long_dev),
        "steps": step,
        "updates": updates,
        "train_seconds": round(time.perf_counter() - started, 1),
        "final_loss": log[-1]["loss"] if log else None,
    }
    (out / "train_log.json").write_text(
        json.dumps({"summary": summary, "log": log}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
