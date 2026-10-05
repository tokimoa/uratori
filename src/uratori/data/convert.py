"""既存データセットを uratori の Record に直す。合成データの 3 経路のうち「既存データの形式変換」。

変換したデータは元の license（JNLI は CC BY-SA 4.0）を引き継ぐ。git には入れない。
"""

from __future__ import annotations

import random

from uratori.data.rubrics import Rubric
from uratori.data.schema import LabelSource, Provenance, Record, Split, Tags, Verification

# JNLI の neutral は「前提からは決められない」で、こちらの情報不足に当たる
JNLI_TO_SUPPORT_3 = {"entailment": "支持", "contradiction": "矛盾", "neutral": "情報不足"}


def jnli_to_records(rows: list[dict], split: Split, rubric: Rubric, prefix: str) -> list[Record]:
    """JNLI の 1 行を gr-support-3 の 1 件にする。sentence1 が根拠、sentence2 が主張。

    yjcaptions_id の先頭は元の画像の id。同じ画像の説明文どうしを 1 family にまとめる。
    """
    if set(rubric.criteria) != set(JNLI_TO_SUPPORT_3.values()):
        raise ValueError(f"rubric {rubric.id} の候補が JNLI の対応と合わない")
    out = []
    for row in rows:
        image_id = row["yjcaptions_id"].split("-")[0]
        out.append(
            Record(
                id=f"{prefix}-{row['sentence_pair_id']}",
                family_id=f"jnli-img-{image_id}",
                split=split,
                state={"根拠": row["sentence1"], "主張": row["sentence2"]},
                type=rubric.type,
                question=rubric.question,
                criteria=rubric.criteria,
                target_hard=JNLI_TO_SUPPORT_3[row["label"]],
                label_source=LabelSource.DATASET,
                verification=Verification(status="verified", by=["JNLI"]),
                provenance=Provenance(
                    source=f"JGLUE/JNLI v1.3 {prefix} {row['sentence_pair_id']}",
                    source_license="CC-BY-SA-4.0",
                    generator="convert-jnli",
                ),
                tags=Tags(lang="ja", domain=rubric.domain, rubric_id=rubric.id),
            )
        )
    return out


# JNLI の 3 値を、真偽 2 値の rubric に直すときの「true になるラベル」
JNLI_TRUE_WHEN = {"gr-support-noul": "entailment", "gr-contradict-noul": "contradiction"}


def jnli_to_mixed_records(
    rows: list[dict], split: Split, rubrics: dict[str, Rubric], prefix: str, seed: int = 0
) -> list[Record]:
    """JNLI の各行を、3 つの rubric のどれか 1 つの形にする。

    半分を gr-support-3、残りを gr-support-noul と gr-contradict-noul に分ける。先に NLI で
    学習する段階で、選択と真偽の両方の形式に慣れさせるため。同じ文の組を複数の形式で
    重ねて使うことはしない。
    """
    rng = random.Random(seed)
    out = []
    for row in rows:
        rid = rng.choices(["gr-support-3", "gr-support-noul", "gr-contradict-noul"], [2, 1, 1])[0]
        if rid == "gr-support-3":
            out += jnli_to_records([row], split, rubrics[rid], prefix)
            continue
        rubric = rubrics[rid]
        base = jnli_to_records([row], split, rubrics["gr-support-3"], prefix)[0]
        out.append(
            base.model_copy(
                update={
                    "type": rubric.type,
                    "question": rubric.question,
                    "criteria": rubric.criteria,
                    "target_hard": "true" if row["label"] == JNLI_TRUE_WHEN[rid] else "false",
                    "tags": base.tags.model_copy(update={"rubric_id": rid}),
                }
            )
        )
    return [Record.model_validate(r.model_dump()) for r in out]
