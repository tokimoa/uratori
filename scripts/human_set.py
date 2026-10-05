"""評価セットを作るときの補助。下書き（ラベルの案が入った Record の JSONL）からラベルを確定させる。

  uv run python scripts/human_set.py sheet --drafts data/drafts/*.jsonl --dir data/myset
  uv run python scripts/human_set.py consensus --drafts data/drafts/*.jsonl --dir data/myset \
      --judge runs/judge/a__myset.jsonl runs/judge/b__myset.jsonl

  sheet      読むためのシート（下書きのラベルなし）を作る
  compare    オーナーのラベル（label_cli.py の出力）を下書きや LLM の判定と突き合わせる
  finalize   裁定を反映して、確定したデータを書く
  consensus  複数の判定者のラベルから評価用のラベルを決め、割れた件の見直しシートを作る

作業用のファイル（sheet.md、labels.csv、compare.md、adjudication.csv、review.md）は --dir に置く。
確定したデータは --out（省略すると <dir>/final.jsonl）に書く。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from uratori.data.human import (
    compare,
    consensus,
    finalize,
    needs_review,
    read_labels,
    render_item,
    shuffled,
)
from uratori.data.validate import load_records, read_jsonl, validate_files


def load_drafts(paths: list[Path]):
    return [r for path in sorted(paths) for r in load_records(path)]


def cmd_sheet(args) -> None:
    records = shuffled(load_drafts(args.drafts))
    out = args.dir / "sheet.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n---\n\n".join(render_item(r, i, len(records)) for i, r in enumerate(records, 1))
    out.write_text(f"# {args.dir.name} ラベル付けシート\n\n{body}\n", encoding="utf-8")
    print(f"{len(records)} 件を {out} に書いた")


def cmd_compare(args) -> None:
    records = load_drafts(args.drafts)
    base = args.dir
    owner = read_labels(base / "labels.csv")
    draft = {r.id: r.target_hard for r in records}
    judges = {"下書き": draft}
    reasons: dict[str, dict[str, str]] = {}
    for path in args.judge or []:
        rows = [row for _, row in read_jsonl(path) if "label" in row]
        judges[path.stem] = {row["id"]: row["label"] for row in rows}
        reasons[path.stem] = {row["id"]: row.get("reason", "") for row in rows}

    lines = [
        f"# {base.name} 突き合わせ",
        "",
        f"オーナーのラベル {len(owner)}/{len(records)} 件。",
        "",
    ]
    lines += [
        "| 相手 | 件数 | 一致 | noul の κ | choice の κ | score の κ |",
        "|---|---|---|---|---|---|",
    ]
    for name, other in judges.items():
        c = compare(records, owner, other)
        kappa = [f"{c[t]['kappa']:.2f}" if t in c else "" for t in ("noul", "choice", "score")]
        lines.append(f"| {name} | {c['n']} | {c['agreement']:.1%} | {' | '.join(kappa)} |")

    review = [r for r in records if r.id in owner and needs_review(r, owner[r.id])]
    lines += ["", f"## 見直す件（{len(review)} 件）", "",
              "オーナーのラベルが下書きと違う件と、迷いの印を付けた件。最終ラベルを adjudication.csv に書く。", ""]  # fmt: skip
    for r in review:
        mine = owner[r.id]
        lines += [render_item(r, records.index(r) + 1, len(records)), ""]
        lines.append(
            f"- オーナー: {mine.label}"
            + ("（迷い）" if mine.unsure else "")
            + (f" メモ: {mine.memo}" if mine.memo else "")
        )
        lines.append(f"- 下書き: {r.target_hard}。{r.verification.note or ''}")
        for name, other in judges.items():
            if name in reasons and r.id in other:
                lines.append(f"- {name}: {other[r.id]}。{reasons[name].get(r.id, '')}")
        lines += ["", "---", ""]
    base.mkdir(parents=True, exist_ok=True)
    (base / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    adj = base / "adjudication.csv"
    if not adj.exists():
        with adj.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "final", "memo"])
            writer.writerows([r.id, "", ""] for r in review)
    print(f"{base / 'compare.md'} を書いた。見直す件は {len(review)} 件。裁定は {adj} に書く")


def _read_adjudication(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row["final"] for row in csv.DictReader(f) if row["final"]}


def cmd_finalize(args) -> None:
    records = load_drafts(args.drafts)
    base = args.dir
    owner = read_labels(base / "labels.csv")
    done, pending = finalize(records, owner, _read_adjudication(base / "adjudication.csv"))
    out = args.out or base / "final.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in done:
            f.write(r.model_dump_json() + "\n")
    report = validate_files([out])
    print(
        f"{len(done)} 件を {out} に書いた。未確定 {len(pending)} 件。validator: {'OK' if report.ok else report.errors[:3]}"
    )
    if pending:
        print("未確定の例:", json.dumps(pending[:8]))


def _load_judge(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = [row for _, row in read_jsonl(path) if "label" in row]
    return {r["id"]: r["label"] for r in rows}, {r["id"]: r.get("reason", "") for r in rows}


def cmd_consensus(args) -> None:
    """判定者のラベルから評価用のラベルを決め、割れた件の見直しシートを作る。"""
    records = load_drafts(args.drafts)
    base = args.dir
    base.mkdir(parents=True, exist_ok=True)
    ref_name = args.reference_name
    judges = {"draft": {r.id: r.target_hard for r in records}}
    all_reasons = {"draft": {r.id: r.verification.note or "" for r in records}}
    if args.reference:
        judges[ref_name], all_reasons[ref_name] = _load_judge(args.reference)
    for path in args.judge or []:
        name = path.stem.split("__")[0]
        judges[name], all_reasons[name] = _load_judge(path)
    strong = [n for n in judges if n not in set(args.weak or [])]
    if not args.reference:
        # 正解に使う判定者がないときは、強い判定者の多数決にする。同数なら下書きを優先する
        majority, reasons = {}, {}
        for r in records:
            votes = Counter(judges[n][r.id] for n in strong if r.id in judges[n])
            top = max(votes.values())
            tied = [label for label, c in votes.items() if c == top]
            majority[r.id] = r.target_hard if r.target_hard in tied else tied[0]
            reasons[r.id] = f"多数決 {dict(votes)}"
        ref_name = "majority"
        judges[ref_name], all_reasons[ref_name] = majority, reasons
        strong.append(ref_name)

    adj_path = base / "adjudication.csv"
    adjudication = _read_adjudication(adj_path)
    done, review = consensus(records, ref_name, judges, all_reasons[ref_name], strong, adjudication)

    out = args.out or base / "final.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in done:
            f.write(r.model_dump_json() + "\n")
    report = validate_files([out])

    by_id = {r.id: r for r in records}
    lines = [f"# {base.name} 判定者が割れた件", "",
             f"{', '.join(strong)} の間でラベルが割れた {len(review)} 件。最終ラベルを adjudication.csv の final に書く。", ""]  # fmt: skip
    for rid in review:
        r = by_id[rid]
        lines += [render_item(r, records.index(r) + 1, len(records)), ""]
        for name, labels in judges.items():
            if rid in labels:
                reason = all_reasons[name].get(rid, "")
                lines.append(f"- {name}: {labels[rid]}" + (f"。{reason}" if reason else ""))
        lines += ["", "---", ""]
    (base / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not adj_path.exists():
        with adj_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "final", "memo"])
            writer.writerows([rid, "", ""] for rid in review)
    human = sum(r.label_source.value == "human" for r in done)
    print(
        f"{len(done)} 件を {out} に書いた。人が裁定した件 {human}、判定者が割れたままの件 {len(review)}。"
        f"validator: {'OK' if report.ok else report.errors[:3]}"
    )
    print(f"見直しシート {base / 'review.md'}、裁定の記入先 {adj_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sheet", "compare", "finalize", "consensus"])
    p.add_argument("--drafts", nargs="+", type=Path, required=True, help="下書きの JSONL")
    p.add_argument("--dir", type=Path, required=True, help="作業用のファイルを置くフォルダ")
    p.add_argument("--out", type=Path, help="確定したデータの出力先。省略すると <dir>/final.jsonl")
    p.add_argument("--judge", nargs="*", type=Path, help="LLM の判定（judge_with_llm.py の出力）")
    p.add_argument(
        "--reference", type=Path, help="consensus で正解に使う判定ファイル。省略すると多数決"
    )
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--weak", nargs="*", help="割れたかどうかの判断に入れない判定者の名前")
    args = p.parse_args()
    commands = {
        "sheet": cmd_sheet,
        "compare": cmd_compare,
        "finalize": cmd_finalize,
        "consensus": cmd_consensus,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
