"""候補の順番を逆にしても答えが変わらないかを調べる。

候補マーカーを採点する方式のモデルは、中身ではなく「候補の位置」に反応することがある
（最初の候補をほとんど選ばない、など）。choice は候補の並びを逆にし、score は段階の説明を逆順にする
（正解の段階は n-1-i に移る）。noul は候補が固定なので対象外。

  uv run --group train python scripts/position_probe.py --ckpt runs/mbj310 --data data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

from uratori.data.schema import Record  # noqa: E402
from uratori.data.validate import load_records  # noqa: E402
from uratori.train.loop import encode_records, load_checkpoint, predict  # noqa: E402


def reversed_record(r: Record) -> Record:
    if r.type.value == "choice":
        criteria = dict(reversed(list(r.criteria.items())))
        target = r.target_hard
    else:
        criteria = list(reversed(r.criteria))
        target = str(len(criteria) - 1 - int(r.target_hard))
    return r.model_copy(update={"id": r.id + "#rev", "criteria": criteria, "target_hard": target})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    originals = [
        r for r in load_records(args.data) if r.type.value != "noul" and r.target_hard is not None
    ]
    flipped = [reversed_record(r) for r in originals]
    model, tokenizer, marker, _ = load_checkpoint(args.ckpt, args.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    items, _ = encode_records(
        originals + flipped, tokenizer, marker, model.marker_id, args.max_tokens
    )
    preds = predict(model, items, pad_id, args.device, args.batch_size)

    out = {}
    for kind in ("choice", "score"):
        rows = [
            (a, b)
            for a, b in zip(originals, flipped, strict=True)
            if a.type.value == kind and a.id in preds and b.id in preds
        ]
        n = len(rows)
        pick_a = [int(max(range(len(preds[a.id])), key=preds[a.id].__getitem__)) for a, _ in rows]
        pick_b = [int(max(range(len(preds[b.id])), key=preds[b.id].__getitem__)) for _, b in rows]
        acc_a = sum(i == a.target_index for (a, _), i in zip(rows, pick_a, strict=True)) / n
        acc_b = sum(i == b.target_index for (_, b), i in zip(rows, pick_b, strict=True)) / n
        # 同じ中身の候補を選んでいれば、逆順での位置は k-1-i になる
        same = (
            sum(
                j == len(a.labels) - 1 - i
                for (a, _), i, j in zip(rows, pick_a, pick_b, strict=True)
            )
            / n
        )
        first_gold = sum(a.target_index == 0 for a, _ in rows) / n
        last_gold = sum(a.target_index == len(a.labels) - 1 for a, _ in rows) / n
        out[kind] = {
            "n": n,
            "accuracy_original": round(acc_a, 4),
            "accuracy_reversed": round(acc_b, 4),
            "same_content_chosen": round(same, 4),
            "picked_first_original": round(sum(i == 0 for i in pick_a) / n, 4),
            "picked_first_reversed": round(sum(j == 0 for j in pick_b) / n, 4),
            "gold_first_original": round(first_gold, 4),
            "gold_first_reversed": round(last_gold, 4),
            "positions_original": dict(sorted(Counter(pick_a).items())),
            "positions_reversed": dict(sorted(Counter(pick_b).items())),
        }
    (args.ckpt / f"position_probe__{args.data.stem}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
