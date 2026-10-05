"""noul の criteria（はいといいえの条件）を外しても答えられるかを調べる。

uv run --group train python scripts/noul_probe.py --ckpt runs/mbj310 --data data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np  # noqa: E402

from uratori.data.validate import load_records  # noqa: E402
from uratori.train.loop import encode_records, load_checkpoint, predict  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    with_c = [
        r for r in load_records(args.data) if r.type.value == "noul" and r.criteria is not None
    ]
    if not with_c:
        raise SystemExit("criteria つきの noul がデータにない")
    without = [r.model_copy(update={"id": r.id + "#nc", "criteria": None}) for r in with_c]
    model, tokenizer, marker, _ = load_checkpoint(args.ckpt, args.device)
    items, _ = encode_records(with_c + without, tokenizer, marker, model.marker_id, 2048)
    preds = predict(model, items, tokenizer.pad_token_id or 0, args.device, 4)

    def stats(rows):
        p_true = np.array([preds[r.id][1] for r in rows])
        gold = np.array([r.target_hard == "true" for r in rows])
        order = np.argsort(p_true)
        ranks = np.empty(len(rows))
        ranks[order] = np.arange(1, len(rows) + 1)
        auroc = (ranks[gold].sum() - gold.sum() * (gold.sum() + 1) / 2) / (
            gold.sum() * (~gold).sum()
        )
        return {"accuracy": round(float(((p_true >= 0.5) == gold).mean()), 4), "auroc": round(float(auroc), 4),
                "mean_p_true": round(float(p_true.mean()), 3), "gold_true_rate": round(float(gold.mean()), 3)}  # fmt: skip

    out = {"n": len(with_c), "with_criteria": stats(with_c), "without_criteria": stats(without)}
    (args.ckpt / f"noul_probe__{args.data.stem}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
