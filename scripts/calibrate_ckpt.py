"""checkpoint の予測に temperature scaling を当て、校正前後の指標を比べる。

  uv run python scripts/calibrate_ckpt.py --ckpt runs/mbj310 \
      --fit cal=data/calibration.jsonl --eval test=data/test.jsonl

予測は eval_ckpt.py が書いた <ckpt>/pred__<名前>.jsonl を使う。温度は <ckpt>/calibration.json に書く。
--split-eval を付けると、評価セットを family 単位で 3 等分し、1/3 で温度を合わせて残りで測る
（運用に近い分布が校正用にない場合の代わり）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from uratori.data.validate import load_records
from uratori.eval import metrics as M
from uratori.eval.calibrate import apply_by_group, fit_by_group
from uratori.eval.report import load_predictions


def summarize(records, preds) -> dict:
    kept = [r for r in records if r.id in preds]
    probs = [preds[r.id] for r in kept]
    y = np.array([r.target_index for r in kept])
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(p.argmax()) == t for p, t in zip(probs, y, strict=True)])
    sel = M.selective_at_coverage(conf, correct, 0.7)
    return {
        "n": len(kept),
        "accuracy": round(float(correct.mean()), 4),
        "nll": round(M.nll(probs, y), 4),
        "brier": round(M.brier(probs, y), 4),
        "ece": round(M.ece(conf, correct)["ece"], 4),
        "aurc": round(M.aurc(conf, correct), 4),
        "error_at_70": round(sel["error_rate"], 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--fit", action="append", default=[], help="名前=パス。温度を合わせるデータ")
    p.add_argument("--eval", action="append", default=[], help="名前=パス。校正前後を測るデータ")
    p.add_argument("--split-eval", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    def load(spec):
        name, path = spec.split("=", 1)
        return name, load_records(Path(path)), load_predictions(args.ckpt / f"pred__{name}.jsonl")

    out = {}
    if args.fit:
        fit_records, fit_preds = [], {}
        for spec in args.fit:
            _, records, preds = load(spec)
            fit_records += records
            fit_preds |= preds
        temps = fit_by_group(fit_records, fit_preds)
        (args.ckpt / "calibration.json").write_text(json.dumps(temps, indent=2), encoding="utf-8")
        out["temperatures"] = {k: round(v, 3) for k, v in temps.items()}
        for spec in args.eval:
            name, records, preds = load(spec)
            out[name] = {
                "before": summarize(records, preds),
                "after": summarize(records, apply_by_group(records, preds, temps)),
            }

    if args.split_eval:
        for spec in args.eval:
            name, records, preds = load(spec)
            families = sorted({r.family_id for r in records})
            random.Random(args.seed).shuffle(families)
            folds = [set(families[i::3]) for i in range(3)]
            before, after = [], []
            for k in range(3):
                cal = [r for r in records if r.family_id in folds[k]]
                rest = [r for r in records if r.family_id not in folds[k]]
                temps = fit_by_group(cal, preds)
                before.append(summarize(rest, preds))
                after.append(summarize(rest, apply_by_group(rest, preds, temps)))
            avg = lambda rows: {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}  # noqa: E731
            out[f"{name}_3fold"] = {"before": avg(before), "after": avg(after)}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
