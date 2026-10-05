"""公開している評価セット（tokimoa/uratori-ja-eval）でモデルを測る。

  uv run --group train python scripts/eval_hub.py --model tokimoa/uratori-ja-310m --device cpu
  uv run --group train python scripts/eval_hub.py --ckpt runs/mbj310 --split test --device cuda

評価セットは Hub から取ってくる。split ごとに、Record の形に直したデータを <out>/<split>.jsonl、
予測を <out>/pred__<split>.jsonl、レポートを <out>/report__<split>.md に書く。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from uratori.data.public import EVAL_REPO, EVAL_SPLITS, load_eval_split
from uratori.eval.report import build_report, load_predictions, render_markdown
from uratori.serialize import InputTooLongError
from uratori.serve.predictor import HubPredictor, UratoriPredictor
from uratori.train.loop import write_predictions


def main() -> None:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hub の repo id か、同じ形のローカルのフォルダ")
    source.add_argument("--ckpt", type=Path, help="scripts/train.py が書いた checkpoint のフォルダ")
    p.add_argument("--split", nargs="+", default=["test", "challenge"], choices=EVAL_SPLITS)
    p.add_argument("--dataset", default=EVAL_REPO)
    p.add_argument("--device", default="cpu", help="cpu、cuda、mps のどれか")
    p.add_argument("--max-tokens", type=int, default=2048, help="--ckpt のときの入力の上限")
    p.add_argument("--resamples", type=int, default=500)
    p.add_argument("--out", type=Path, help="出力先。省略すると runs/eval_hub/<モデル名>")
    args = p.parse_args()

    if args.model:
        predictor = HubPredictor.from_pretrained(args.model, args.device)
    else:
        predictor = UratoriPredictor(args.ckpt, args.device, args.max_tokens)
    out = args.out or Path("runs") / "eval_hub" / predictor.name.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        records = load_eval_split(split, args.dataset)
        (out / f"{split}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
        )
        preds, too_long = {}, []
        started = time.perf_counter()
        for record in records:
            try:
                probs, _ = predictor.predict(record.state, {"q": record.to_question()})
            except InputTooLongError:
                # 上限を超えた件は切らずに外し、件数を残す
                too_long.append(record.id)
                continue
            preds[record.id] = probs["q"]
        elapsed = time.perf_counter() - started
        write_predictions(preds, out / f"pred__{split}.jsonl")
        report = build_report(
            [r for r in records if r.id in preds],
            load_predictions(out / f"pred__{split}.jsonl"),
            n_resamples=args.resamples,
        )
        (out / f"report__{split}.md").write_text(
            render_markdown(report, f"{predictor.name} / {split}"), encoding="utf-8"
        )
        o = report["overall"]
        line = {
            "split": split,
            "n": o["n"],
            "skipped_too_long": len(too_long),
            "accuracy": round(o["accuracy"], 4),
            "accuracy_ci": [round(o["accuracy_ci"]["lo"], 4), round(o["accuracy_ci"]["hi"], 4)],
            "macro_f1": round(o["macro_f1"], 4),
            "ece": round(o["ece"], 4),
            "ms_per_item": round(1000 * elapsed / max(1, len(preds)), 1),
        }
        print(json.dumps(line, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
