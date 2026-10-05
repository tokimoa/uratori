"""1 構成ぶんの学習または推論の負荷を測り、結果を JSON 1 行で標準出力に書く。

ピークメモリを構成ごとに分けて測るため、1 回の起動で 1 構成だけ測る。

  uv run --group train python scripts/measure_load.py \
      --model sbintuitions/modernbert-ja-310m --mode full --seq-len 1024 --batch 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings

import torch
from transformers import AutoModel, AutoTokenizer

from uratori.models.scorer import OptionScorer, resolve_marker

N_OPTIONS = 3
WARMUP = 3


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def mps_memory_gb() -> float:
    return torch.mps.driver_allocated_memory() / 1e9


def make_batch(vocab: int, marker_id: int, batch: int, seq_len: int, device: str):
    # 特殊トークンを避けて語彙の中ほどから引く。末尾付近に候補マーカーを置く
    ids = torch.randint(1000, min(vocab, 30000), (batch, seq_len))
    ids[ids == marker_id] = 1000
    for k in range(N_OPTIONS):
        ids[:, seq_len - 2 - 8 * k] = marker_id
    labels = torch.randint(0, N_OPTIONS, (batch,))
    return ids.to(device), torch.ones_like(ids).to(device), labels.to(device)


def build(model_id: str, mode: str, device: str, attn: str) -> tuple[OptionScorer, int, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    marker, marker_id, added = resolve_marker(tokenizer)
    extra = {} if attn == "default" else {"attn_implementation": attn}
    backbone = AutoModel.from_pretrained(model_id, dtype=torch.float32, **extra)
    if added:
        backbone.resize_token_embeddings(len(tokenizer))
    info = {
        "backbone_class": type(backbone).__name__,
        "marker": marker,
        "attn_used": getattr(backbone.config, "_attn_implementation", None),
        "params_m": round(sum(p.numel() for p in backbone.parameters()) / 1e6, 1),
    }
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
        )
    model = OptionScorer(backbone, backbone.config.hidden_size, marker_id).to(device)
    info["trainable_m"] = round(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1
    )
    return model, len(tokenizer), info


def measure_train(model, vocab, args) -> dict:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    times, peak = [], 0.0
    for step in range(WARMUP + args.steps):
        ids, mask, labels = make_batch(
            vocab, model.marker_id, args.batch, args.seq_len, args.device
        )
        sync(args.device)
        t0 = time.perf_counter()
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            logits, _ = model(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(args.device)
        if step >= WARMUP:
            times.append(time.perf_counter() - t0)
        if args.device == "mps":
            peak = max(peak, mps_memory_gb())
    sec = statistics.median(times)
    return {
        "sec_per_step": round(sec, 4),
        "examples_per_sec": round(args.batch / sec, 2),
        "tokens_per_sec": round(args.batch * args.seq_len / sec),
        "peak_mem_gb": round(peak, 2) if args.device == "mps" else None,
        "final_loss": round(float(loss), 4),
    }


def measure_infer(model, vocab, args) -> dict:
    model.eval()
    times, peak = [], 0.0
    with torch.inference_mode():
        for step in range(WARMUP + args.steps):
            ids, mask, _ = make_batch(vocab, model.marker_id, args.batch, args.seq_len, args.device)
            sync(args.device)
            t0 = time.perf_counter()
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
                model(ids, mask)
            sync(args.device)
            if step >= WARMUP:
                times.append(time.perf_counter() - t0)
            if args.device == "mps":
                peak = max(peak, mps_memory_gb())
    times.sort()
    p50 = statistics.median(times)
    return {
        "p50_ms": round(p50 * 1000, 1),
        "p95_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))] * 1000, 1),
        "examples_per_sec": round(args.batch / p50, 2),
        "peak_mem_gb": round(peak, 2) if args.device == "mps" else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["full", "lora", "infer"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--attn", choices=["default", "eager", "sdpa"], default="default")
    p.add_argument("--grad-ckpt", action="store_true")
    args = p.parse_args()

    result = vars(args).copy()
    result["torch"] = torch.__version__
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, vocab, info = build(args.model, args.mode, args.device, args.attn)
            result.update(info)
            if args.grad_ckpt:
                model.backbone.gradient_checkpointing_enable()
            fn = measure_infer if args.mode == "infer" else measure_train
            result.update(fn(model, vocab, args))
        fallbacks = sorted({str(w.message)[:160] for w in caught if "fall back" in str(w.message)})
        result["mps_fallback_warnings"] = fallbacks
        result["status"] = "ok"
    except Exception as e:  # 構成ごとの失敗（メモリ不足、未対応の演算）も結果として残す
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
