"""temperature scaling。モデルを固定してから、校正用のデータで温度を合わせる。

質問の型と候補数で過信の度合いが違うので、(型, 候補数) ごとに温度を持つ。件数の少ない
組は型ごとの温度、それも足りなければ全体の温度を使う。確率しか保存していないので、
対数確率を温度で割る（ロジットを温度で割るのと同じ結果になる）。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from uratori.data.schema import Record

MIN_GROUP = 30
_GOLDEN = (math.sqrt(5) - 1) / 2


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    out = np.exp(logits)
    return out / out.sum()


def _nll(probs: list[np.ndarray], y: list[int], temperature: float) -> float:
    return float(
        np.mean(
            [
                -math.log(max(apply_temperature(p, temperature)[t], 1e-12))
                for p, t in zip(probs, y, strict=True)
            ]
        )
    )


def fit_temperature(probs: list[np.ndarray], y: list[int]) -> float:
    """NLL が最小になる温度を、log 温度の黄金分割探索で求める。"""
    lo, hi = math.log(0.05), math.log(20.0)
    a, b = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    fa, fb = _nll(probs, y, math.exp(a)), _nll(probs, y, math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - _GOLDEN * (hi - lo)
            fa = _nll(probs, y, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + _GOLDEN * (hi - lo)
            fb = _nll(probs, y, math.exp(b))
    return math.exp((lo + hi) / 2)


def group_keys(record: Record) -> list[str]:
    """細かい順。最初に見つかった温度を使う。"""
    return [f"{record.type.value}:{len(record.labels)}", record.type.value, "all"]


def fit_by_group(records: list[Record], preds: dict[str, np.ndarray]) -> dict[str, float]:
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for record in records:
        if record.target_index is None or record.id not in preds:
            continue
        for key in group_keys(record):
            groups[key][0].append(preds[record.id])
            groups[key][1].append(record.target_index)
    return {
        key: fit_temperature(probs, y)
        for key, (probs, y) in groups.items()
        if len(y) >= MIN_GROUP or key == "all"
    }


def apply_by_group(
    records: list[Record], preds: dict[str, np.ndarray], temperatures: dict[str, float]
) -> dict[str, np.ndarray]:
    out = {}
    for record in records:
        if record.id not in preds:
            continue
        key = next(k for k in group_keys(record) if k in temperatures)
        out[record.id] = apply_temperature(preds[record.id], temperatures[key])
    return out
