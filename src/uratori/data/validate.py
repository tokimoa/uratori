"""JSONL の validator。1 件ずつの schema 検査に加え、ファイルをまたぐ検査をする。

uv run uratori-validate examples/sample_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from uratori.data.schema import Record, Split

# family がまたいではいけない分割の組。train に入った原文の派生が評価側に漏れるのを防ぐ。
# pool は train に昇格するので train と同じ側に置く。
_TRAIN_SIDE = {Split.TRAIN, Split.POOL}


@dataclass
class Report:
    n_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                yield lineno, json.loads(line)


def load_records(path: Path) -> list[Record]:
    return [Record.model_validate(row) for _, row in read_jsonl(path)]


def validate_files(paths: Iterable[Path]) -> Report:
    report = Report()
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    family_splits: dict[str, set[Split]] = defaultdict(set)
    known_ids: set[str] = set()
    contrast_refs: list[tuple[str, str, str]] = []

    for path in paths:
        try:
            rows = list(read_jsonl(path))
        except json.JSONDecodeError as e:
            report.errors.append(f"{path}: JSON として読めない行がある（{e}）")
            continue
        for lineno, row in rows:
            where = f"{path}:{lineno}"
            try:
                rec = Record.model_validate(row)
            except ValidationError as e:
                first = e.errors()[0]
                loc = ".".join(str(x) for x in first["loc"]) or "record"
                report.errors.append(f"{where}: {loc}: {first['msg']}")
                continue
            report.n_records += 1
            report.counts[(rec.split.value, rec.tags.domain, rec.type.value)] += 1
            known_ids.add(rec.id)

            if rec.id in seen_ids:
                report.errors.append(f"{where}: id {rec.id!r} が {seen_ids[rec.id]} と重複")
            seen_ids[rec.id] = where

            h = rec.content_hash()
            if h in seen_hashes:
                report.errors.append(f"{where}: 内容が {seen_hashes[h]} と完全に重複")
            seen_hashes[h] = where

            family_splits[rec.family_id].add(rec.split)
            if rec.soft_hard_mismatch():
                report.warnings.append(f"{where}: target_probs の最頻ラベルが target_hard と違う")
            if rec.provenance.contrast_of:
                contrast_refs.append((where, rec.id, rec.provenance.contrast_of))

    for family, splits in sorted(family_splits.items()):
        if splits & _TRAIN_SIDE and splits - _TRAIN_SIDE:
            names = ", ".join(sorted(s.value for s in splits))
            report.errors.append(f"family {family!r} が学習側と評価側にまたがっている（{names}）")
        elif len(splits - _TRAIN_SIDE) > 1:
            names = ", ".join(sorted(s.value for s in splits))
            report.warnings.append(f"family {family!r} が複数の評価分割にある（{names}）")

    for where, rec_id, ref in contrast_refs:
        if ref not in known_ids:
            report.warnings.append(f"{where}: {rec_id!r} の contrast_of {ref!r} が見つからない")

    return report


def format_report(report: Report) -> str:
    lines = [f"{report.n_records} 件を読んだ"]
    if report.counts:
        lines.append("split / domain / type ごとの件数")
        for (split, domain, qtype), n in sorted(report.counts.items()):
            lines.append(f"  {split:<12}{domain:<11}{qtype:<7}{n}")
    for w in report.warnings:
        lines.append(f"警告 {w}")
    for e in report.errors:
        lines.append(f"エラー {e}")
    lines.append("OK" if report.ok else f"エラー {len(report.errors)} 件")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="uratori のデータ JSONL を検査する")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    report = validate_files(args.paths)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
