"""checkpoint を ONNX に書き出す。CPU だけの環境や、PyTorch を入れたくない環境で動かすため。

ONNX にするのは「トークンごとのスコア」を出すところまで。候補マーカーの位置のスコアを
集めて softmax する部分は、実行する側で numpy で行う（onnx_scores_to_probs）。

  uv run --group train --group export python -m uratori.serve.onnx_export --ckpt runs/x --out runs/x/onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def onnx_scores_to_probs(
    scores: np.ndarray, input_ids: np.ndarray, marker_id: int
) -> list[np.ndarray]:
    """[batch, seq] のスコアから、件ごとの候補の確率を作る。"""
    out = []
    for row_scores, row_ids in zip(scores, input_ids, strict=True):
        logits = row_scores[row_ids == marker_id].astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        out.append(probs / probs.sum())
    return out


def export(ckpt: Path, out: Path) -> Path:
    import torch

    from uratori.train.loop import load_checkpoint

    model, tokenizer, marker, meta = load_checkpoint(ckpt, "cpu")
    model.eval()

    class TokenScores(torch.nn.Module):
        def __init__(self, scorer):
            super().__init__()
            self.scorer = scorer

        def forward(self, input_ids, attention_mask):
            hidden = self.scorer.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return self.scorer.head(hidden).squeeze(-1)

    out.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(
        ["状態:\nテスト\n\n質問: 正しいか\n候補:\nいいえ " + marker + "\nはい " + marker],
        return_tensors="pt",
    )
    path = out / "model.onnx"
    torch.onnx.export(
        TokenScores(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "scores": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )
    tokenizer.save_pretrained(out / "tokenizer")
    info = {"marker": marker, "marker_id": model.marker_id, "source": meta["config"]["model_id"]}
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if (ckpt / "calibration.json").exists():
        shutil.copy(ckpt / "calibration.json", out / "calibration.json")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(export(args.ckpt, args.out))


if __name__ == "__main__":
    main()
