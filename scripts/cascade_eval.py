"""3 層構成の 4 方式を比べる。

  uv run python scripts/cascade_eval.py --ckpt runs/mbj310 --data data/test.jsonl --pred-name test \
      --judge runs/judge/deepseek-flash__test.jsonl

--pred-name は eval_ckpt.py の --set に付けた名前で、<ckpt>/pred__<名前>.jsonl を読む。校正の温度は
<ckpt>/calibration.json（calibrate_ckpt.py が書く）、--judge は judge_with_llm.py の出力を使う。

方式: uratori 単独、強い LLM 単独、uratori＋低確信を LLM へ、ルール＋uratori＋低確信を LLM へ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uratori.cascade import layer1_check, route
from uratori.data.validate import load_records
from uratori.eval.calibrate import apply_by_group
from uratori.eval.report import load_predictions
from uratori.gen.plan import SOURCE_FIELD


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pred-name", required=True)
    p.add_argument("--judge", type=Path, required=True)
    args = p.parse_args()

    records = load_records(args.data)
    temps = json.loads((args.ckpt / "calibration.json").read_text())
    model = apply_by_group(
        records, load_predictions(args.ckpt / f"pred__{args.pred_name}.jsonl"), temps
    )
    judge = {
        x["id"]: np.array(x["probs"])
        for x in map(json.loads, args.judge.read_text().splitlines())
        if "probs" in x
    }
    records = [r for r in records if r.id in judge]

    def source_keys(r):
        key = SOURCE_FIELD.get(r.tags.rubric_id) if r.tags.rubric_id else None
        return [key] if key else [next(iter(r.state))]

    m_ok = np.array([int(model[r.id].argmax()) == r.target_index for r in records])
    j_ok = np.array([int(judge[r.id].argmax()) == r.target_index for r in records])
    flags = [layer1_check(r.state, source_keys(r)) for r in records]
    flagged = np.array([f.flagged for f in flags])
    print(
        f"件数 {len(records)}。第 1 層が合図を出した件 {flagged.sum()}（{flagged.mean():.1%}）。合図のある件での uratori の誤り率 {1 - m_ok[flagged].mean():.3f}、ない件 {1 - m_ok[~flagged].mean():.3f}"
    )
    print(f"uratori 単独 {m_ok.mean():.3f}、LLM 単独 {j_ok.mean():.3f}")
    print("| 閾値 | 方式 | LLM 呼び出し | Accuracy |")
    for threshold in (0.6, 0.7, 0.8, 0.9):
        for use_rules in (False, True):
            decisions = [
                route(model[r.id], f, threshold, use_rules)
                for r, f in zip(records, flags, strict=True)
            ]
            esc = np.array([d == "escalate" for d in decisions])
            acc = np.where(esc, j_ok, m_ok).mean()
            name = "ルール＋uratori＋LLM" if use_rules else "uratori＋LLM"
            print(f"| {threshold} | {name} | {esc.mean():.1%} | {acc:.3f} |")


if __name__ == "__main__":
    main()
