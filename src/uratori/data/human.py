"""評価セットのラベル付けと突き合わせ。

下書きのラベルを見せずに人（以下、オーナー）がラベルを付け、後から下書きや LLM の判定と
突き合わせる。食い違った件は理由を並べて見直し、裁定を adjudication.csv に書く。
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from uratori.data.schema import LabelSource, Record, Verification
from uratori.serialize import render_state

LABEL_FIELDS = ["id", "label", "unsure", "memo"]
NOUL_SURFACE = {"false": "いいえ", "true": "はい"}


@dataclass(frozen=True)
class OwnerLabel:
    label: str
    unsure: bool = False
    memo: str = ""


def shuffled(records: list[Record], seed: int = 0) -> list[Record]:
    """同じ family の最小対が並ばないよう、全体を混ぜる。順番は seed で固定。"""
    out = sorted(records, key=lambda r: r.id)
    random.Random(seed).shuffle(out)
    return out


def option_lines(record: Record) -> list[str]:
    question = record.to_question()
    lines = []
    for i, (label, desc) in enumerate(zip(question.labels, question.descriptions, strict=True), 1):
        if record.type.value == "noul":
            head = NOUL_SURFACE[label]
        elif record.type.value == "score":
            head = f"段階{label}"
        else:
            head = label
        lines.append(f"{i}. {head}" + (f"  {desc}" if desc else ""))
    return lines


def render_item(record: Record, position: int, total: int) -> str:
    """ラベル付けの画面とシートに出す 1 件。下書きのラベル、理由、タグは出さない。"""
    return "\n".join(
        [
            f"### {position}/{total}  {record.id}",
            "",
            render_state(record.state),
            "",
            f"質問: {record.question}",
            "",
            *option_lines(record),
        ]
    )


def read_labels(path: Path) -> dict[str, OwnerLabel]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            row["id"]: OwnerLabel(row["label"], row.get("unsure") == "1", row.get("memo", ""))
            for row in csv.DictReader(f)
            if row.get("label")
        }


def write_labels(path: Path, labels: dict[str, OwnerLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for rid, v in labels.items():
            writer.writerow({"id": rid, "label": v.label, "unsure": int(v.unsure), "memo": v.memo})


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """2 人の一致を偶然の一致で補正する。ラベル集合が件ごとに違うので、型ごとなどに分けて使う。"""
    if not a:
        return float("nan")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return float("nan") if expected == 1.0 else (observed - expected) / (1.0 - expected)


def compare(records: list[Record], owner: dict[str, OwnerLabel], other: dict[str, str]) -> dict:
    """オーナーのラベルと、別の判定（下書き、LLM）の一致。"""
    both = [r for r in records if r.id in owner and r.id in other]
    a = [owner[r.id].label for r in both]
    b = [other[r.id] for r in both]
    out = {
        "n": len(both),
        "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(both)
        if both
        else float("nan"),
    }
    for qtype in ("noul", "choice", "score"):
        idx = [i for i, r in enumerate(both) if r.type.value == qtype]
        if idx:
            xa, xb = [a[i] for i in idx], [b[i] for i in idx]
            out[qtype] = {
                "n": len(idx),
                "agreement": sum(x == y for x, y in zip(xa, xb, strict=True)) / len(idx),
                "kappa": cohen_kappa(xa, xb),
            }
    return out


def needs_review(record: Record, owner: OwnerLabel) -> bool:
    return owner.unsure or owner.label != record.target_hard


def finalize(
    records: list[Record], owner: dict[str, OwnerLabel], adjudication: dict[str, str]
) -> tuple[list[Record], list[str]]:
    """最終ラベルを決めて label_source を human にする。まだ決まらない件の id も返す。

    オーナーと下書きが一致し、迷いの印もない件はそのまま確定。それ以外は裁定が要る。
    裁定前のラベルは human_votes に残す。
    """
    done, pending = [], []
    for record in records:
        if record.id not in owner:
            pending.append(record.id)
            continue
        mine = owner[record.id]
        if mine.label not in record.labels:
            raise ValueError(f"{record.id}: ラベル {mine.label!r} が候補にない")
        if needs_review(record, mine):
            if record.id not in adjudication:
                pending.append(record.id)
                continue
            final = adjudication[record.id]
            if final not in record.labels:
                raise ValueError(f"{record.id}: 裁定 {final!r} が候補にない")
            note = f"裁定あり。下書きは {record.target_hard}、最初のラベルは {mine.label}"
            by = ["owner", "adjudicated"]
        else:
            final, note, by = mine.label, None, ["owner", "claude-draft"]
        if mine.memo:
            note = f"{note}。メモ: {mine.memo}" if note else f"メモ: {mine.memo}"
        done.append(
            record.model_copy(
                update={
                    "target_hard": final,
                    "label_source": LabelSource.HUMAN,
                    "human_votes": [mine.label],
                    "verification": Verification(status="verified", by=by, note=note),
                }
            )
        )
    return done, pending


def consensus(
    records: list[Record],
    reference_name: str,
    judges: dict[str, dict[str, str]],
    reasons: dict[str, str],
    strong: list[str],
    adjudication: dict[str, str],
) -> tuple[list[Record], list[str]]:
    """複数の判定者のラベルから評価用のラベルを決める。人が見直すべき件の id も返す。

    正解には reference_name の判定者のラベルを使い、label_source は teacher のままにする。
    人が付けたラベルではないことをデータに残すため。strong の判定者が全員一致した件は
    verified、割れた件は disputed にして見直しの対象にする。人が裁定した件だけ
    label_source が human になる。

    judges は判定者の名前から {id: ラベル} への対応。下書きのラベルも判定者の 1 つとして渡す。
    """
    done, review = [], []
    for record in records:
        votes = {name: labels[record.id] for name, labels in judges.items() if record.id in labels}
        label = votes[reference_name]
        if label not in record.labels:
            raise ValueError(f"{record.id}: {reference_name} のラベル {label!r} が候補にない")
        counts = [sum(v == x for v in votes.values()) for x in record.labels]
        split = len({votes[name] for name in strong if name in votes}) > 1
        update = {
            "teacher_vote_distribution": [c / len(votes) for c in counts],
            "teacher_reported_probs": None,
        }
        if record.id in adjudication:
            final = adjudication[record.id]
            if final not in record.labels:
                raise ValueError(f"{record.id}: 裁定 {final!r} が候補にない")
            update |= {
                "target_hard": final,
                "label_source": LabelSource.HUMAN,
                "human_votes": [final],
                "verification": Verification(
                    status="verified",
                    by=["owner"],
                    note=f"判定者が割れたので人が裁定した。{reference_name} は {label}",
                ),
            }
        else:
            if split:
                review.append(record.id)
            update |= {
                "target_hard": label,
                "label_source": LabelSource.TEACHER,
                "verification": Verification(
                    status="disputed" if split else "verified",
                    by=sorted(name for name, v in votes.items() if v == label),
                    note=reasons.get(record.id),
                ),
            }
        done.append(record.model_copy(update=update))
    return done, review
