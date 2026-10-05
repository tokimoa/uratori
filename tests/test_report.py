import json
from pathlib import Path

import numpy as np
import pytest

from uratori.data.validate import load_records
from uratori.eval.report import align, build_report, main, render_markdown

SAMPLES = Path(__file__).parent.parent / "examples" / "sample_records.jsonl"


def perfect_predictions(records, wrong_ids=()):
    preds = {}
    for r in records:
        p = np.full(len(r.labels), 0.1 / (len(r.labels) - 1))
        target = r.target_index
        if r.id in wrong_ids:
            target = (target + 1) % len(r.labels)
        p[target] = 0.9
        preds[r.id] = p
    return preds


def test_perfect_predictions_score_one():
    records = load_records(SAMPLES)
    report = build_report(records, perfect_predictions(records), n_resamples=50)
    assert report["overall"]["n"] == len(records)
    assert report["overall"]["accuracy"] == 1.0
    assert report["overall"]["macro_f1"] == pytest.approx(1.0)
    assert report["overall"]["score"]["mae"] == 0.0
    assert report["minimal_pairs"] == {"pairs": 1, "both_correct": 1.0}
    assert report["high_confidence_errors"] == []
    assert set(report["slices"]["domain"]) == {"grounding", "compare", "rag", "writing"}


def test_errors_show_up_in_slices_pairs_and_error_list():
    records = load_records(SAMPLES)
    report = build_report(records, perfect_predictions(records, {"ex-0002"}), n_resamples=50)
    assert report["slices"]["type"]["noul"]["accuracy"] == 0.5
    assert report["slices"]["type"]["choice"]["accuracy"] == 1.0
    assert report["minimal_pairs"]["both_correct"] == 0.0
    assert [e["id"] for e in report["high_confidence_errors"]] == ["ex-0002"]
    ci = report["overall"]["accuracy_ci"]
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_error_correlation_section_needs_second_prediction_file():
    records = load_records(SAMPLES)
    a = perfect_predictions(records, {"ex-0002", "ex-0004"})
    report = build_report(records, a, preds_b=a, n_resamples=20)
    assert report["error_correlation"]["all_errors"]["same_wrong_rate"] == 1.0
    assert "誤り相関" in render_markdown(report)


def test_missing_or_malformed_predictions_are_errors():
    records = load_records(SAMPLES)
    preds = perfect_predictions(records)
    with pytest.raises(ValueError, match="予測がない"):
        align(records, {k: v for k, v in preds.items() if k != "ex-0001"})
    with pytest.raises(ValueError, match="長さ"):
        align(records, preds | {"ex-0001": np.array([0.2, 0.3, 0.5])})
    with pytest.raises(ValueError, match="確率分布"):
        align(records, preds | {"ex-0001": np.array([0.9, 0.9])})


def test_cli_writes_json_and_markdown(tmp_path):
    records = load_records(SAMPLES)
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        "".join(
            json.dumps({"id": k, "probs": v.tolist()}) + "\n"
            for k, v in perfect_predictions(records).items()
        )
    )
    out = tmp_path / "report"
    assert (
        main(["--data", str(SAMPLES), "--pred", str(pred), "--out", str(out), "--resamples", "20"])
        == 0
    )
    assert json.loads(out.with_suffix(".json").read_text())["overall"]["accuracy"] == 1.0
    assert out.with_suffix(".md").read_text().startswith("# 評価レポート")
