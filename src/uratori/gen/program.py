"""コードだけで作る最小対。合成データの 3 経路のうち「プログラムで正解が決まる」もの。

対象は gr-number-match（主張の数値が根拠と一致するか）に限る。この rubric は定義上、
主張の数値を根拠にない値へ変えれば必ず false になり、表記だけを変えれば true のままになる。
他の rubric では、数値を変えてもラベルがどう変わるかがコードでは決まらない
（「7日以内」を「5日以内」に変えた主張は、支持されたままになる）。
"""

from __future__ import annotations

import random
import re

from uratori.data.schema import LabelSource, Record, Verification

RUBRIC_ID = "gr-number-match"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TO_ZENKAKU = str.maketrans("0123456789,.", "０１２３４５６７８９，．")


def _format_like(value: int, original: str) -> str:
    return f"{value:,}" if "," in original else str(value)


def perturb_number(claim: str, evidence: str, rng: random.Random) -> str | None:
    """主張の数値を 1 つ、根拠に出てこない値へ変える。変えられる数値がなければ None。"""
    matches = [m for m in _NUMBER.finditer(claim) if "." not in m.group()]
    rng.shuffle(matches)
    evidence_numbers = {m.group().replace(",", "") for m in _NUMBER.finditer(evidence)}
    for m in matches:
        value = int(m.group().replace(",", ""))
        if value == 0:
            continue
        # 桁を保ったまま、丸めでは一致しない程度に変える
        candidates = (
            [value + d for d in (1, 2, 3, -1, -2)]
            if value < 20
            else [round(value * f) for f in (1.2, 0.8, 1.5, 0.5, 2.0)]
        )
        rng.shuffle(candidates)
        for new in candidates:
            if new > 0 and new != value and str(new) not in evidence_numbers:
                return claim[: m.start()] + _format_like(new, m.group()) + claim[m.end() :]
    return None


def restyle_number(claim: str, rng: random.Random) -> str | None:
    """数値の表記だけを変える（全角にする、桁区切りを付け外しする）。値は変えない。"""
    matches = list(_NUMBER.finditer(claim))
    if not matches:
        return None
    m = rng.choice(matches)
    text = m.group()
    options = [text.translate(_TO_ZENKAKU)]
    if "," in text:
        options.append(text.replace(",", ""))
    elif "." not in text and len(text) >= 4:
        options.append(f"{int(text):,}")
    new = rng.choice(options)
    return claim[: m.start()] + new + claim[m.end() :]


def number_pairs(records: list[Record], seed: int = 0) -> list[Record]:
    """検証済みの true の件から、数値を変えた false の件と、表記を変えた true の件を作る。"""
    rng = random.Random(seed)
    out = []
    for record in records:
        if record.tags.rubric_id != RUBRIC_ID or record.target_hard != "true":
            continue
        claim, evidence = record.state["主張"], record.state["根拠"]
        variants = [
            ("num", perturb_number(claim, evidence, rng), "false", ["数値"]),
            ("fmt", restyle_number(claim, rng), "true", ["全角半角"]),
        ]
        for suffix, new_claim, label, phenomena in variants:
            if new_claim is None or new_claim == claim:
                continue
            out.append(
                record.model_copy(
                    update={
                        "id": f"{record.id}-{suffix}",
                        "state": record.state | {"主張": new_claim},
                        "target_hard": label,
                        "target_probs": None,
                        "teacher_reported_probs": None,
                        "teacher_vote_distribution": None,
                        "label_source": LabelSource.PROGRAM,
                        "verification": Verification(status="verified", by=["program"]),
                        "provenance": record.provenance.model_copy(
                            update={"generator": "program-number-v1", "contrast_of": record.id}
                        ),
                        "tags": record.tags.model_copy(
                            update={
                                "phenomena": sorted(set(record.tags.phenomena) | set(phenomena))
                            }
                        ),
                    }
                )
            )
    return [Record.model_validate(r.model_dump()) for r in out]
