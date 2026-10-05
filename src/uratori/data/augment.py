"""学習データの見せ方を変える拡張。中身と正解は変えない。

拡張なしで学習したモデルで見つかった 2 つの弱点に対応する。
- noul は criteria（はいといいえの条件）を付けないと不安定だった。学習データの noul がほぼ全て
  criteria つきだったため。一定の割合で criteria を外して見せる。
- choice は候補の並びを逆にすると正答率が 4 ポイント下がった。一定の割合で並びを入れ替える。

score は段階の順番に意味があるので触らない。
"""

from __future__ import annotations

import random

from uratori.data.schema import Record


def soften(records: list[Record], teacher: dict[str, list[float]]) -> tuple[list[Record], int]:
    """教師モデルの確率を target_probs に入れる。候補数が合わない件と、予測のない件はそのまま。"""
    out, used = [], 0
    for r in records:
        probs = teacher.get(r.id)
        if probs is None or len(probs) != len(r.labels):
            out.append(r)
            continue
        total = sum(probs)
        out.append(r.model_copy(update={"target_probs": [p / total for p in probs]}))
        used += 1
    return out, used


def augment(
    records: list[Record], seed: int, noul_drop: float = 0.0, choice_shuffle: float = 0.0
) -> list[Record]:
    rng = random.Random(seed)
    out = []
    for r in records:
        kind = r.type.value
        if kind == "noul" and r.criteria is not None and rng.random() < noul_drop:
            r = r.model_copy(update={"criteria": None})
        elif kind == "choice" and rng.random() < choice_shuffle:
            order = list(range(len(r.labels)))
            rng.shuffle(order)
            items = list(r.criteria.items())
            update: dict = {"criteria": dict(items[i] for i in order)}
            for name in ("target_probs", "teacher_reported_probs", "teacher_vote_distribution"):
                values = getattr(r, name)
                if values is not None:
                    update[name] = [values[i] for i in order]
            r = r.model_copy(update=update)
        out.append(r)
    return out
