"""公開している評価セット（tokimoa/uratori-ja-eval）を Record の形に直す。

公開の形は、state と候補を型の揃った list にしてある。ここで学習と評価のコードが使う
Record に戻すので、変換後のファイルは uratori-eval や scripts/eval_ckpt.py にそのまま渡せる。
"""

from __future__ import annotations

import json
from pathlib import Path

from uratori.data.schema import LabelSource, Provenance, Record, Tags, Verification

EVAL_REPO = "tokimoa/uratori-ja-eval"
EVAL_SPLITS = ("test", "challenge", "calibration", "dev")


def record_from_public_row(row: dict, split: str, source: str = EVAL_REPO) -> Record:
    labels = [option["label"] for option in row["options"]]
    descriptions = [option["description"] for option in row["options"]]
    pairs = zip(labels, descriptions, strict=True)
    if row["type"] == "noul":
        # 説明のない noul は criteria なしとして扱う
        criteria = {label: desc for label, desc in pairs if desc} or None
    elif row["type"] == "choice":
        criteria = {label: desc or None for label, desc in pairs}
    else:
        criteria = descriptions
    return Record(
        id=row["id"],
        family_id=row["family_id"],
        split=split,
        state={part["name"]: part["text"] for part in row["state"]},
        type=row["type"],
        question=row["question"],
        criteria=criteria,
        target_hard=row["answer"],
        # 正解は複数の LLM の判定で決めたもの。人が付けたラベルではない
        label_source=LabelSource.TEACHER,
        verification=Verification(
            status="verified" if row.get("agreement") == "unanimous" else "disputed"
        ),
        provenance=Provenance(
            source=source, source_license="CC-BY-4.0", contrast_of=row.get("pair_of")
        ),
        tags=Tags(
            lang="ja",
            domain=row["domain"],
            difficulty=row["difficulty"],
            phenomena=row.get("phenomena") or [],
            rubric_id=row.get("template"),
        ),
    )


def load_eval_split(split: str, repo: str = EVAL_REPO, revision: str | None = None) -> list[Record]:
    """評価セットの 1 つの split を Hub から取ってきて Record にする。"""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, f"{split}.jsonl", repo_type="dataset", revision=revision)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [record_from_public_row(json.loads(line), split, repo) for line in lines if line.strip()]
