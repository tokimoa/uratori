"""予測の JSONL を受けてレポートを作る。どのモデルもこのコードで測る。

予測は 1 行 1 件で {"id": ..., "probs": [...]}。probs はその件の labels の順。

  uv run uratori-eval --data data/dev.jsonl --pred runs/x/pred.jsonl --out runs/x/report
  uv run uratori-eval --data ... --pred a.jsonl --pred-b b.jsonl   # 誤り相関も出す
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uratori.data.schema import PROB_SUM_TOLERANCE, Record
from uratori.data.validate import load_records, read_jsonl
from uratori.eval import metrics as M

SELECTIVE_COVERAGE = 0.7
TOP_ERRORS = 20
SLICE_KEYS = ("type", "domain", "lang", "difficulty", "phenomena")


@dataclass
class Scored:
    records: list[Record]
    probs: list[np.ndarray]
    y_true: np.ndarray
    y_pred: np.ndarray
    conf: np.ndarray
    correct: np.ndarray
    families: list[str]
    # 同じラベル集合を持つ件のまとまり。Macro F1 はこの単位で計算する
    spaces: list[tuple[str, tuple[str, ...]]]


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    return {row["id"]: np.asarray(row["probs"], dtype=float) for _, row in read_jsonl(path)}


def align(records: list[Record], preds: dict[str, np.ndarray]) -> Scored:
    """正解ラベルのある件に予測を対応させる。予測の欠けや形の違いは黙って落とさずエラーにする。"""
    kept = [r for r in records if r.target_hard is not None]
    missing = [r.id for r in kept if r.id not in preds]
    if missing:
        raise ValueError(f"予測がない件が {len(missing)} 件ある（例: {missing[:3]}）")
    probs = []
    for r in kept:
        p = preds[r.id]
        if len(p) != len(r.labels):
            raise ValueError(f"{r.id}: 予測の長さ {len(p)} が候補数 {len(r.labels)} と合わない")
        if abs(float(p.sum()) - 1.0) > PROB_SUM_TOLERANCE or (p < 0).any():
            raise ValueError(f"{r.id}: 予測が確率分布になっていない")
        probs.append(p)
    y_true = np.array([r.target_index for r in kept])
    y_pred = np.array([int(p.argmax()) for p in probs])
    return Scored(
        records=kept,
        probs=probs,
        y_true=y_true,
        y_pred=y_pred,
        conf=np.array([float(p.max()) for p in probs]),
        correct=y_true == y_pred,
        families=[r.family_id for r in kept],
        spaces=[(r.type.value, tuple(r.labels)) for r in kept],
    )


def _grouped(s: Scored, idx: np.ndarray, fn: Callable[[np.ndarray, int], float]) -> float:
    """ラベル集合ごとに fn を計算し、件数で重み付けして平均する。"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in idx:
        groups[s.spaces[i]].append(int(i))
    total, weight = 0.0, 0
    for (_, labels), members in groups.items():
        value = fn(np.array(members), len(labels))
        if not math.isnan(value):
            total += value * len(members)
            weight += len(members)
    return total / weight if weight else float("nan")


def macro_f1(s: Scored, idx: np.ndarray) -> float:
    return _grouped(s, idx, lambda m, n: M.macro_f1(s.y_true[m], s.y_pred[m], n))


def summarize(s: Scored, idx: np.ndarray) -> dict:
    probs = [s.probs[i] for i in idx]
    out = {
        "n": int(len(idx)),
        "accuracy": M.accuracy(s.y_true[idx], s.y_pred[idx]),
        "macro_f1": macro_f1(s, idx),
        "nll": M.nll(probs, s.y_true[idx]),
        "brier": M.brier(probs, s.y_true[idx]),
        "ece": M.ece(s.conf[idx], s.correct[idx])["ece"],
        "aurc": M.aurc(s.conf[idx], s.correct[idx]),
    }
    score_idx = np.array([i for i in idx if s.spaces[i][0] == "score"], dtype=int)
    if len(score_idx):
        expected = np.array(
            [float(np.dot(np.arange(len(s.probs[i])), s.probs[i])) for i in score_idx]
        )
        out["score"] = {
            "n": int(len(score_idx)),
            "mae": M.mae(s.y_true[score_idx], s.y_pred[score_idx]),
            "mae_expected": M.mae(s.y_true[score_idx], expected),
            "qwk": _grouped(
                s, score_idx, lambda m, n: M.quadratic_weighted_kappa(s.y_true[m], s.y_pred[m], n)
            ),
        }
    return out


def _slice_values(r: Record, key: str) -> list[str]:
    if key == "type":
        return [r.type.value]
    if key == "phenomena":
        return r.tags.phenomena
    return [getattr(r.tags, key)]


def build_report(
    records: list[Record],
    preds: dict[str, np.ndarray],
    preds_b: dict[str, np.ndarray] | None = None,
    n_resamples: int = 1000,
) -> dict:
    s = align(records, preds)
    everything = np.arange(len(s.records))
    report: dict = {"overall": summarize(s, everything)}

    report["overall"]["accuracy_ci"] = M.family_bootstrap(
        s.families, lambda i: M.accuracy(s.y_true[i], s.y_pred[i]), n_resamples
    )
    report["overall"]["macro_f1_ci"] = M.family_bootstrap(
        s.families, lambda i: macro_f1(s, i), n_resamples
    )
    report["selective"] = M.selective_at_coverage(s.conf, s.correct, SELECTIVE_COVERAGE)
    report["calibration"] = M.ece(s.conf, s.correct)

    report["slices"] = {}
    for key in SLICE_KEYS:
        members: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(s.records):
            for value in _slice_values(r, key):
                members[value].append(i)
        report["slices"][key] = {v: summarize(s, np.array(m)) for v, m in sorted(members.items())}

    per_space: dict[tuple, list[int]] = defaultdict(list)
    for i, space in enumerate(s.spaces):
        per_space[space].append(i)
    report["label_spaces"] = []
    for (qtype, labels), m in sorted(per_space.items(), key=lambda kv: -len(kv[1])):
        m = np.array(m)
        rows = M.per_class_prf(s.y_true[m], s.y_pred[m], len(labels))
        report["label_spaces"].append(
            {
                "type": qtype,
                "labels": list(labels),
                "n": int(len(m)),
                "per_class": dict(zip(labels, rows, strict=True)),
                "confusion": M.confusion_matrix(s.y_true[m], s.y_pred[m], len(labels)).tolist(),
            }
        )

    correct_by_id = {r.id: bool(c) for r, c in zip(s.records, s.correct, strict=True)}
    pairs = [(r.provenance.contrast_of, r.id) for r in s.records if r.provenance.contrast_of]
    report["minimal_pairs"] = M.pair_accuracy(correct_by_id, pairs)

    wrong = np.flatnonzero(~s.correct)
    worst = wrong[np.argsort(-s.conf[wrong], kind="stable")][:TOP_ERRORS]
    report["high_confidence_errors"] = [
        {
            "id": s.records[i].id,
            "confidence": float(s.conf[i]),
            "predicted": s.records[i].labels[s.y_pred[i]],
            "target": s.records[i].target_hard,
            "domain": s.records[i].tags.domain,
        }
        for i in worst
    ]

    if preds_b is not None:
        b = align(records, preds_b)
        report["error_correlation"] = {
            "all_errors": M.error_correlation(s.y_pred, s.conf, b.y_pred, s.y_true, 1.0),
            "top_quarter": M.error_correlation(s.y_pred, s.conf, b.y_pred, s.y_true, 0.25),
        }
    return report


def _f(x: float | None) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def render_markdown(report: dict, title: str = "評価レポート") -> str:
    o = report["overall"]
    acc, f1 = o["accuracy_ci"], o["macro_f1_ci"]
    sel = report["selective"]
    lines = [
        f"# {title}",
        "",
        f"{o['n']} 件、{acc['n_families']} family。区間は family 単位の bootstrap による 95% 区間。",
        "",
        "| 指標 | 値 | 95% 区間 |",
        "|------|----|----------|",
        f"| Accuracy | {_f(o['accuracy'])} | {_f(acc['lo'])} 〜 {_f(acc['hi'])} |",
        f"| Macro F1 | {_f(o['macro_f1'])} | {_f(f1['lo'])} 〜 {_f(f1['hi'])} |",
        f"| NLL | {_f(o['nll'])} | |",
        f"| Brier | {_f(o['brier'])} | |",
        f"| ECE（{report['calibration']['n_bins']} bin） | {_f(o['ece'])} | |",
        f"| AURC | {_f(o['aurc'])} | |",
        "",
        f"確信度の高い順に {sel['coverage']:.0%} を受理すると、{sel['accepted']} 件中 {sel['errors']} 件が誤り"
        f"（誤り率 {_f(sel['error_rate'])}、95% 上限 {_f(sel['error_rate_upper95'])}）。",
    ]
    if "score" in o:
        sc = o["score"]
        lines += [
            "",
            f"段階評価 {sc['n']} 件は MAE {_f(sc['mae'])}（期待値では {_f(sc['mae_expected'])}）、"
            f"重み付き κ {_f(sc['qwk'])}。",
        ]
    for key, table in report["slices"].items():
        if not table:
            continue
        lines += [
            "",
            f"## {key} 別",
            "",
            "| | 件数 | Accuracy | Macro F1 | NLL | ECE |",
            "|---|---|---|---|---|---|",
        ]
        for value, row in table.items():
            lines.append(
                f"| {value} | {row['n']} | {_f(row['accuracy'])} | {_f(row['macro_f1'])} "
                f"| {_f(row['nll'])} | {_f(row['ece'])} |"
            )
    lines += ["", "## ラベル集合別", ""]
    for space in report["label_spaces"]:
        lines += [
            f"{space['type']}、{space['n']} 件",
            "",
            "| ラベル | 適合率 | 再現率 | F1 | 件数 |",
            "|--------|--------|--------|----|------|",
        ]
        for label, row in space["per_class"].items():
            lines.append(
                f"| {label} | {_f(row['precision'])} | {_f(row['recall'])} | {_f(row['f1'])} "
                f"| {row['support']} |"
            )
        lines.append("")
    mp = report["minimal_pairs"]
    if mp["pairs"]:
        lines += [
            f"最小対 {mp['pairs']} 組のうち、両方に正解したのは {mp['both_correct']:.1%}。",
            "",
        ]
    lines += ["## 校正", "", "| 確信度の bin | 件数 | 正答率 | 平均確信度 |", "|---|---|---|---|"]
    for b in report["calibration"]["bins"]:
        if b["count"]:
            lines.append(
                f"| {b['lo']:.2f}〜{b['hi']:.2f} | {b['count']} | {_f(b['accuracy'])} "
                f"| {_f(b['confidence'])} |"
            )
    if report["high_confidence_errors"]:
        lines += [
            "",
            "## 確信度の高い誤り",
            "",
            "| id | 確信度 | 予測 | 正解 | 用途 |",
            "|---|---|---|---|---|",
        ]
        for e in report["high_confidence_errors"]:
            lines.append(
                f"| {e['id']} | {_f(e['confidence'])} | {e['predicted']} | {e['target']} "
                f"| {e['domain']} |"
            )
    if "error_correlation" in report:
        lines += [
            "",
            "## 誤り相関",
            "",
            "A（--pred）が誤った例で、B（--pred-b）が同じ誤答をした割合。",
            "",
            "| 対象 | 件数 | 同じ誤答 | B の誤り率（対象内） | B の誤り率（全体） |",
            "|---|---|---|---|---|",
        ]
        for name, key in (("A の誤り全て", "all_errors"), ("確信度の上位 25%", "top_quarter")):
            c = report["error_correlation"][key]
            if c["a_errors"]:
                lines.append(
                    f"| {name} | {c['examined']} | {_f(c['same_wrong_rate'])} "
                    f"| {_f(c['b_error_rate_on_examined'])} | {_f(c['b_error_rate_overall'])} |"
                )
    lines += [
        "",
        "## 定義",
        "",
        "Macro F1 はラベル集合が同じ件ごとに計算し、件数で重み付けして平均した。"
        "Brier は各クラスの平方誤差の和の平均で、2 クラスでは二値の定義の 2 倍になる。"
        "ECE は最頻ラベルの確率を確信度とした等幅 bin。"
        "受理率を指定した誤り率は確信度の順位で切っており、閾値をこのデータで選んだことになるので、"
        "運用の閾値は calibration 分割で決めて別途測る。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="予測を採点してレポートを作る")
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path)
    parser.add_argument("--out", type=Path, help="拡張子なしの出力先。.json と .md を書く")
    parser.add_argument("--title", default="評価レポート")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    records = [r for path in args.data for r in load_records(path)]
    preds_b = load_predictions(args.pred_b) if args.pred_b else None
    report = build_report(records, load_predictions(args.pred), preds_b, args.resamples)
    markdown = render_markdown(report, args.title)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
