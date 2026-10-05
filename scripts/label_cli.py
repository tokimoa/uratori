"""端末で人手ラベルを付ける。1 件ごとに保存するので、途中でやめて続きから再開できる。

  uv run python scripts/label_cli.py --drafts data/drafts/*.jsonl --dir data/myset            # 続きから
  uv run python scripts/label_cli.py --drafts data/drafts/*.jsonl --dir data/myset --redo ID  # 1 件だけ付け直す

下書きのラベルと理由は表示しない。ラベルは <dir>/labels.csv に書く（human_set.py が読む）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uratori.data.human import OwnerLabel, read_labels, render_item, shuffled, write_labels
from uratori.data.validate import load_records

HELP = "番号で回答。noul は y / n でも可。? を付けると迷いの印（例 2?）。m メモ、b 1 件戻る、s 後回し、q 終了"


def parse(answer: str, labels: list[str], is_noul: bool) -> tuple[str, bool] | None:
    unsure = answer.endswith("?")
    core = answer.rstrip("?").strip().lower()
    if is_noul and core in ("y", "n"):
        return ("true" if core == "y" else "false"), unsure
    if core.isdigit() and 1 <= int(core) <= len(labels):
        return labels[int(core) - 1], unsure
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drafts", nargs="+", type=Path, required=True, help="下書きの JSONL")
    p.add_argument("--dir", type=Path, required=True, help="labels.csv を置くフォルダ")
    p.add_argument("--redo")
    args = p.parse_args()

    records = shuffled([r for path in sorted(args.drafts) for r in load_records(path)])
    out = args.dir / "labels.csv"
    labels = read_labels(out)
    order = [r for r in records if r.id == args.redo] if args.redo else records
    print(f"{len(records)} 件中 {len(labels)} 件が済み。{HELP}\n")

    i, memo = 0, ""
    while i < len(order):
        record = order[i]
        if record.id in labels and not args.redo:
            i += 1
            continue
        print("\n" + "=" * 72)
        print(render_item(record, records.index(record) + 1, len(records)))
        answer = input("\n> ").strip()
        if answer == "q":
            break
        if answer == "s":
            i += 1
            continue
        if answer == "b":
            if i > 0:
                i -= 1
                labels.pop(order[i].id, None)
            continue
        if answer == "m":
            memo = input("メモ> ").strip()
            continue
        parsed = parse(answer, record.labels, record.type.value == "noul")
        if parsed is None:
            print(HELP)
            continue
        labels[record.id] = OwnerLabel(parsed[0], parsed[1], memo)
        write_labels(out, labels)
        memo = ""
        i += 1
    print(f"\n{len(labels)}/{len(records)} 件を {out} に保存した")


if __name__ == "__main__":
    main()
