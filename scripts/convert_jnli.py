"""JNLI を変換して data/external/jnli/ に書く。

元データは JGLUE の GitHub（https://github.com/yahoojapan/JGLUE）から取り、JNLI の train と valid を
data/external/jnli/ に train.jsonl、valid.jsonl の名前で置いておく。変換後も CC BY-SA 4.0 を引き継ぐ。

uv run python scripts/convert_jnli.py
"""

from __future__ import annotations

import json
from pathlib import Path

from uratori.data.convert import jnli_to_mixed_records, jnli_to_records
from uratori.data.rubrics import load_rubrics
from uratori.data.schema import Split
from uratori.data.validate import validate_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "external" / "jnli"


def dedupe(rows: list[dict]) -> tuple[list[dict], int, int]:
    """同じ文の組が複数回出る。ラベルが一致すれば 1 件にまとめ、食い違えば全て捨てる。"""
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_pair.setdefault((row["sentence1"], row["sentence2"]), []).append(row)
    kept = [group[0] for group in by_pair.values() if len({r["label"] for r in group}) == 1]
    conflict = sum(len(g) for g in by_pair.values() if len({r["label"] for r in g}) > 1)
    return kept, len(rows) - conflict - len(kept), conflict


def main() -> None:
    rubrics = load_rubrics(ROOT / "rubrics")
    rubric = rubrics["gr-support-3"]
    train = [json.loads(line) for line in (SRC / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (SRC / "valid.jsonl").read_text().splitlines()]

    # valid に出てくる文を含む行は train から外す（同じ説明文が別の組で使い回されている）
    valid_sentences = {row[k] for row in valid for k in ("sentence1", "sentence2")}
    no_leak = [
        row
        for row in train
        if row["sentence1"] not in valid_sentences and row["sentence2"] not in valid_sentences
    ]
    kept, n_dup, n_conflict = dedupe(no_leak)
    valid_kept, v_dup, v_conflict = dedupe(valid)

    outputs = {
        "jnli_train.jsonl": jnli_to_records(kept, Split.TRAIN, rubric, "jnli-train"),
        "jnli_valid.jsonl": jnli_to_records(valid_kept, Split.DEV, rubric, "jnli-valid"),
        # 先に NLI で学習する段階で使う。選択と真偽の形式を混ぜたもの
        "jnli_train_mixed.jsonl": jnli_to_mixed_records(kept, Split.TRAIN, rubrics, "jnli-mix"),
    }
    for name, records in outputs.items():
        with (SRC / name).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
    report = validate_files([SRC / name for name in outputs])
    print(
        f"train {len(kept)}/{len(train)} 件。除いたのは valid と文が重なる {len(train) - len(no_leak)} 件、"
        f"重複 {n_dup} 件、ラベルが食い違う重複 {n_conflict} 件"
    )
    print(
        f"valid {len(valid_kept)}/{len(valid)} 件。重複 {v_dup} 件、ラベルが食い違う重複 {v_conflict} 件"
    )
    print("validator:", "OK" if report.ok else report.errors[:3], f"警告 {len(report.warnings)} 件")


if __name__ == "__main__":
    main()
