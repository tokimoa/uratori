"""保存した checkpoint でデータを判定し、uratori-eval が読める予測ファイルを書く。

uv run --group train python scripts/predict.py --ckpt runs/x --data d.jsonl --out runs/x/pred.jsonl
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

from uratori.data.validate import load_records  # noqa: E402
from uratori.train.loop import (  # noqa: E402
    encode_records,
    load_checkpoint,
    predict,
    write_predictions,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    records = [r for path in args.data for r in load_records(path)]
    model, tokenizer, marker, _ = load_checkpoint(args.ckpt, args.device)
    items, too_long = encode_records(records, tokenizer, marker, model.marker_id, args.max_tokens)
    started = time.perf_counter()
    preds = predict(model, items, tokenizer.pad_token_id or 0, args.device, args.batch_size)
    write_predictions(preds, args.out)
    print(
        f"{len(preds)} 件を {time.perf_counter() - started:.1f} 秒で判定。長すぎて外した件 {len(too_long)}: {too_long[:5]}"
    )


if __name__ == "__main__":
    main()
