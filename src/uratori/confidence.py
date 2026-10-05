"""TypeSafe が公開している confidence の式（https://docs.typesafe.ai/confidence）。

confidence は分布の尖り具合で、正答確率ではない。校正の評価には使わない。
ホスト版 API の実際の計算は公開式と一致しないという報告があるので、
ここは「公開式どおり」であって「Jev と同じ値」は保証しない。
"""

from __future__ import annotations

from collections.abc import Sequence


def noul_confidence(p: float) -> float:
    return abs(2.0 * p - 1.0)


def choice_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n < 2:
        raise ValueError("候補が 2 個以上必要")
    return (max(probs) - 1.0 / n) / (1.0 - 1.0 / n)


def score_confidence(probs: Sequence[float]) -> float:
    """最頻段階 m からの確率加重距離を、一様分布のときの距離で割って 1 から引く。

    最頻段階が複数あるときは低い段階を m にする。公開例の (0, 0.5, 0.5) → 0.25 が
    この取り方でだけ再現する。
    """
    n = len(probs)
    if n < 2:
        raise ValueError("段階が 2 個以上必要")
    m = max(range(n), key=lambda i: (probs[i], -i))
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_unif = sum(abs(i - m) for i in range(n)) / n
    return max(0.0, 1.0 - spread / mad_unif)


def score_value(probs: Sequence[float]) -> float:
    """段階番号の期待値。wire の `score` フィールドに入れる値。"""
    return sum(i * p for i, p in enumerate(probs))
