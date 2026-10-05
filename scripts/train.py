"""学習して dev の予測とレポートを出す。

  uv run --group train python scripts/train.py --model sbintuitions/modernbert-ja-310m \
      --train data/train.jsonl --dev data/dev.jsonl --out runs/mbj310
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# MPS の割り当てに上限をかける。上限なしだとメモリを使い切ってスワップに入る
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

from uratori.data.augment import augment, soften
from uratori.data.rubrics import load_rubrics
from uratori.data.validate import load_records
from uratori.eval.report import build_report, load_predictions, render_markdown
from uratori.train.loop import TrainConfig, train

ROOT = Path(__file__).parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", nargs="*", type=Path, default=[])
    p.add_argument("--dev", nargs="*", type=Path, default=[])
    p.add_argument("--overfit-rubric-examples", action="store_true",
                   help="動作確認用。rubric の例を学習にも dev にも使う")  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--soft-weight", type=float, default=0.0)
    p.add_argument("--ordinal-weight", type=float, default=0.0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", help="ここにある checkpoint の重みから始める")
    p.add_argument("--lora", type=int, default=0, help="LoRA の rank。0 なら全層学習")
    p.add_argument("--model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--autocast", action="store_true")
    p.add_argument(
        "--teacher-preds", type=Path, nargs="*", help="教師モデルの予測。target_probs に入れる"
    )
    p.add_argument("--aug-noul-drop", type=float, default=0.0, help="noul の criteria を外す割合")
    p.add_argument(
        "--aug-choice-shuffle", type=float, default=0.0, help="choice の候補を並べ替える割合"
    )
    args = p.parse_args()

    train_records = [r for path in args.train for r in load_records(path)]
    dev_records = [r for path in args.dev for r in load_records(path)]
    if args.teacher_preds:
        teacher = {}
        for path in args.teacher_preds:
            teacher |= {k: v.tolist() for k, v in load_predictions(path).items()}
        train_records, used = soften(train_records, teacher)
        print(f"教師の確率を入れた件 {used} / {len(train_records)}")
    if args.aug_noul_drop or args.aug_choice_shuffle:
        train_records = augment(
            train_records, args.seed, args.aug_noul_drop, args.aug_choice_shuffle
        )
    if args.overfit_rubric_examples:
        for rubric in load_rubrics(ROOT / "rubrics").values():
            examples = [rubric.example_record(i) for i in range(len(rubric.examples))]
            train_records += examples
            dev_records += examples

    cfg = TrainConfig(
        model_id=args.model, max_tokens=args.max_tokens, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, head_lr=args.head_lr,
        soft_weight=args.soft_weight, ordinal_weight=args.ordinal_weight, grad_ckpt=args.grad_ckpt,
        device=args.device, seed=args.seed, init_from=args.init_from, lora_r=args.lora,
        model_dtype=args.model_dtype, autocast=args.autocast,
    )  # fmt: skip
    summary = train(cfg, train_records, dev_records, args.out)
    preds = load_predictions(args.out / "dev_pred.jsonl")
    report = build_report([r for r in dev_records if r.id in preds], preds, n_resamples=200)
    (args.out / "dev_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "dev_report.md").write_text(render_markdown(report), encoding="utf-8")
    o = report["overall"]
    print(json.dumps(summary, ensure_ascii=False))
    print(
        f"dev {o['n']} 件: accuracy {o['accuracy']:.3f}、macro F1 {o['macro_f1']:.3f}、NLL {o['nll']:.3f}"
    )


if __name__ == "__main__":
    main()
