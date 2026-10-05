"""1 つの checkpoint を複数の評価セットで測り、要約を eval.json に書く。

  uv run --group train python scripts/eval_ckpt.py --ckpt runs/mbj310 \
      --set dev=data/dev.jsonl --set test=data/test.jsonl

各セットの予測は <ckpt>/pred__<名前>.jsonl、レポートは <ckpt>/report__<名前>.md に置く。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

from uratori.data.validate import load_records  # noqa: E402
from uratori.eval.report import build_report, load_predictions, render_markdown  # noqa: E402
from uratori.train.loop import (  # noqa: E402
    encode_records,
    load_checkpoint,
    predict,
    write_predictions,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--set", action="append", required=True, help="名前=パス")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--resamples", type=int, default=500)
    args = p.parse_args()

    model, tokenizer, marker, _ = load_checkpoint(args.ckpt, args.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # 前に測ったセットの結果を消さないよう、既存の eval.json に足していく
    eval_path = args.ckpt / "eval.json"
    summary = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    for spec in args.set:
        name, path = spec.split("=", 1)
        records = [r for r in load_records(Path(path)) if r.target_hard is not None]
        items, too_long = encode_records(
            records, tokenizer, marker, model.marker_id, args.max_tokens
        )
        started = time.perf_counter()
        preds = predict(model, items, pad_id, args.device, args.batch_size)
        elapsed = time.perf_counter() - started
        write_predictions(preds, args.ckpt / f"pred__{name}.jsonl")
        kept = [r for r in records if r.id in preds]
        report = build_report(
            kept, load_predictions(args.ckpt / f"pred__{name}.jsonl"), n_resamples=args.resamples
        )
        (args.ckpt / f"report__{name}.md").write_text(
            render_markdown(report, f"{args.ckpt.name} / {name}"), encoding="utf-8"
        )
        o = report["overall"]
        summary[name] = {
            "n": o["n"],
            "skipped_too_long": len(too_long),
            "accuracy": round(o["accuracy"], 4),
            "accuracy_ci": [round(o["accuracy_ci"]["lo"], 4), round(o["accuracy_ci"]["hi"], 4)],
            "macro_f1": round(o["macro_f1"], 4),
            "macro_f1_ci": [round(o["macro_f1_ci"]["lo"], 4), round(o["macro_f1_ci"]["hi"], 4)],
            "nll": round(o["nll"], 4),
            "brier": round(o["brier"], 4),
            "ece": round(o["ece"], 4),
            "aurc": round(o["aurc"], 4),
            "selective_70": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in report["selective"].items()
            },
            "pairs": report["minimal_pairs"],
            "by_type": {k: round(v["accuracy"], 4) for k, v in report["slices"]["type"].items()},
            "by_domain": {
                k: round(v["accuracy"], 4) for k, v in report["slices"]["domain"].items()
            },
            "by_difficulty": {
                k: round(v["accuracy"], 4) for k, v in report["slices"]["difficulty"].items()
            },
            "ms_per_item": round(1000 * elapsed / max(1, len(preds)), 1),
        }
        print(
            name,
            {k: summary[name][k] for k in ("n", "accuracy", "macro_f1", "nll", "ece")},
            flush=True,
        )
    (args.ckpt / "eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
