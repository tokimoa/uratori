"""評価指標。採用した定義を各関数に書いてある。

入力は基本的に numpy 配列。候補数が件ごとに違うので、確率は list[np.ndarray] で持つ。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

ECE_BINS = 15


# ---- 分類 ----------------------------------------------------------------------------


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[dict[str, float]]:
    out = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({"precision": precision, "recall": recall, "f1": f1, "support": tp + fn})
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """正解に 1 件も現れないクラスは平均から除く。"""
    rows = [r for r in per_class_prf(y_true, y_pred, n_classes) if r["support"] > 0]
    return float(np.mean([r["f1"] for r in rows])) if rows else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (y_true, y_pred), 1)
    return m


# ---- 順序尺度 ------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    observed = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    idx = np.arange(n_classes)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n_classes - 1) ** 2
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return float("nan")
    return 1.0 - float((weights * observed).sum()) / denom


# ---- 確率 ----------------------------------------------------------------------------


def nll(probs: Sequence[np.ndarray], y_true: np.ndarray, eps: float = 1e-12) -> float:
    return float(
        np.mean([-math.log(max(float(p[y]), eps)) for p, y in zip(probs, y_true, strict=True)])
    )


def brier(probs: Sequence[np.ndarray], y_true: np.ndarray) -> float:
    """各クラスの平方誤差の和を件数で平均する（多クラスの定義）。

    2 クラスでは (p - y)^2 のちょうど 2 倍になる。二値の慣用値と比べるときは半分にする。
    """
    total = 0.0
    for p, y in zip(probs, y_true, strict=True):
        onehot = np.zeros_like(p)
        onehot[y] = 1.0
        total += float(np.sum((p - onehot) ** 2))
    return total / len(probs)


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> dict:
    """最頻ラベルの確率を確信度として、等幅 bin で測る。bin ごとの件数も返す。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    value, bins = 0.0, []
    for b in range(n_bins):
        sel = which == b
        count = int(sel.sum())
        if count == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0})
            continue
        acc, conf = float(correct[sel].mean()), float(confidence[sel].mean())
        value += count / len(confidence) * abs(acc - conf)
        bins.append(
            {"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": count, "accuracy": acc,
             "confidence": conf}
        )  # fmt: skip
    return {"ece": value, "n_bins": n_bins, "bins": bins}


# ---- 選択的判断 ----------------------------------------------------------------------


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """確信度の高い順に受理していったときの (coverage, 誤り率)。"""
    order = np.argsort(-confidence, kind="stable")
    errors = np.cumsum(~correct[order].astype(bool))
    k = np.arange(1, len(confidence) + 1)
    return k / len(confidence), errors / k


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean(risk_coverage(confidence, correct)[1]))


def wilson_upper(errors: int, n: int, z: float = 1.96) -> float:
    """誤り率の 95% 区間の上限（Wilson）。誤り 0 件でも 0 にならない。"""
    if n == 0:
        return float("nan")
    p = errors / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / (1 + z * z / n))


def selective_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> dict:
    order = np.argsort(-confidence, kind="stable")
    n_accept = max(1, math.ceil(coverage * len(confidence)))
    errors = int((~correct[order][:n_accept].astype(bool)).sum())
    return {
        "coverage": n_accept / len(confidence),
        "accepted": n_accept,
        "errors": errors,
        "error_rate": errors / n_accept,
        "error_rate_upper95": wilson_upper(errors, n_accept),
        "threshold": float(confidence[order][n_accept - 1]),
    }


def selective_at_threshold(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """閾値を別の分割（calibration）で決めてから test に当てるための関数。"""
    accept = confidence >= threshold
    n_accept = int(accept.sum())
    errors = int((~correct[accept].astype(bool)).sum())
    return {
        "coverage": n_accept / len(confidence),
        "accepted": n_accept,
        "errors": errors,
        "error_rate": errors / n_accept if n_accept else float("nan"),
        "error_rate_upper95": wilson_upper(errors, n_accept),
        "threshold": threshold,
    }


# ---- 頑健性 --------------------------------------------------------------------------


def pair_accuracy(correct_by_id: dict[str, bool], pairs: Sequence[tuple[str, str]]) -> dict:
    """最小対の両方に正解した割合。片方だけ当たるのは、差を読めていない兆候。"""
    usable = [(a, b) for a, b in pairs if a in correct_by_id and b in correct_by_id]
    if not usable:
        return {"pairs": 0, "both_correct": float("nan")}
    both = sum(correct_by_id[a] and correct_by_id[b] for a, b in usable)
    return {"pairs": len(usable), "both_correct": both / len(usable)}


def permutation_consistency(labels_by_variant: Sequence[Sequence[str]]) -> float:
    """候補を並べ替えた各変種で選ばれたラベルを件ごとに並べたもの。全変種で一致した件の割合。"""
    return float(np.mean([len(set(v)) == 1 for v in labels_by_variant]))


# ---- 判定器どうしの誤り相関 ----------------------------------------------------------


def error_correlation(
    a_pred: np.ndarray,
    a_conf: np.ndarray,
    b_pred: np.ndarray,
    y_true: np.ndarray,
    top_fraction: float = 1.0,
) -> dict:
    """A が誤った例のうち確信度の高い top_fraction について、B が同じ誤答をした割合。

    b_error_rate_overall は B の全体の誤り率。誤りが独立なら same_wrong_rate はこれを
    上回らない（2 択では等しくなる）ので、差が相関の強さの目安になる。
    """
    a_wrong = np.flatnonzero(a_pred != y_true)
    if len(a_wrong) == 0:
        return {"a_errors": 0, "same_wrong_rate": float("nan")}
    order = a_wrong[np.argsort(-a_conf[a_wrong], kind="stable")]
    picked = order[: max(1, math.ceil(top_fraction * len(order)))]
    return {
        "a_errors": int(len(a_wrong)),
        "examined": int(len(picked)),
        "same_wrong_rate": float(np.mean(b_pred[picked] == a_pred[picked])),
        "b_error_rate_on_examined": float(np.mean(b_pred[picked] != y_true[picked])),
        "b_error_rate_overall": float(np.mean(b_pred != y_true)),
    }


# ---- 区間推定 ------------------------------------------------------------------------


def family_bootstrap(
    families: Sequence[str],
    metric: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """family 単位で復元抽出して 95% 区間を出す。

    同じ原文から作った派生例を独立な標本として数えないため。metric は「件の添字の配列」を
    受けて値を返す関数。
    """
    by_family: dict[str, list[int]] = {}
    for i, fam in enumerate(families):
        by_family.setdefault(fam, []).append(i)
    groups = [np.array(v) for v in by_family.values()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(groups), len(groups))
        value = metric(np.concatenate([groups[g] for g in picked]))
        if not math.isnan(value):
            values.append(value)
    point = metric(np.arange(len(families)))
    if not values:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_families": len(groups)}
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_families": len(groups)}
