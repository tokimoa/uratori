import math

import numpy as np
import pytest

from uratori.eval import metrics as M


def test_accuracy_and_macro_f1_on_a_known_case():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    assert M.accuracy(y_true, y_pred) == pytest.approx(4 / 6)
    rows = M.per_class_prf(y_true, y_pred, 3)
    assert rows[1] == {
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
        "support": 2,
    }
    assert M.macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_ignores_classes_absent_from_truth():
    assert M.macro_f1(np.array([0, 0, 1]), np.array([0, 0, 1]), 3) == pytest.approx(1.0)


def test_confusion_matrix_rows_are_truth():
    m = M.confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), 2)
    assert m.tolist() == [[1, 1], [0, 1]]


def test_quadratic_weighted_kappa():
    y = np.array([0, 1, 2, 3, 4] * 4)
    assert M.quadratic_weighted_kappa(y, y, 5) == pytest.approx(1.0)
    assert M.quadratic_weighted_kappa(y, 4 - y, 5) == pytest.approx(-1.0)
    near = M.quadratic_weighted_kappa(y, np.clip(y + 1, 0, 4), 5)
    far = M.quadratic_weighted_kappa(y, (y + 2) % 5, 5)
    assert 0 < far < near < 1 or far < near < 1


def test_nll_and_brier_definitions():
    probs = [np.array([0.8, 0.2]), np.array([0.5, 0.5])]
    y = np.array([0, 1])
    assert M.nll(probs, y) == pytest.approx(-(math.log(0.8) + math.log(0.5)) / 2)
    # 2 クラスでは二値 Brier (p - y)^2 の 2 倍
    assert M.brier(probs, y) == pytest.approx(2 * (0.2**2 + 0.5**2) / 2)
    assert M.nll([np.array([1.0, 0.0])], np.array([1])) < 30  # 0 の対数で発散しない


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.array([0.75] * 4 + [0.95] * 20)
    correct = np.array([True, True, True, False] + [True] * 19 + [False])
    out = M.ece(conf, correct)
    assert out["ece"] == pytest.approx(0.0, abs=1e-9)
    assert sum(b["count"] for b in out["bins"]) == 24


def test_ece_measures_overconfidence():
    out = M.ece(np.full(10, 0.99), np.array([True] * 5 + [False] * 5))
    assert out["ece"] == pytest.approx(0.49)


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    coverage, risk = M.risk_coverage(conf, correct)
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.25])
    assert M.aurc(conf, correct) == pytest.approx((1 / 3 + 0.25) / 4)
    # 確信度が正誤をよく分けるほど AURC は小さい
    assert M.aurc(conf, correct) < M.aurc(conf[::-1].copy(), correct)


def test_selective_at_coverage_reports_upper_bound_even_with_zero_errors():
    conf = np.linspace(1.0, 0.5, 300)
    out = M.selective_at_coverage(conf, np.ones(300, dtype=bool), 1.0)
    assert out["errors"] == 0 and out["error_rate"] == 0.0
    # 0/300 でも上限は約 1.3%（「3 ÷ 件数」の目安と同じ桁）
    assert 0.008 < out["error_rate_upper95"] < 0.015


def test_selective_at_threshold():
    conf = np.array([0.9, 0.8, 0.6, 0.4])
    out = M.selective_at_threshold(conf, np.array([True, False, True, False]), 0.7)
    assert out["accepted"] == 2 and out["errors"] == 1 and out["coverage"] == 0.5


def test_pair_accuracy_requires_both_sides():
    correct = {"a": True, "b": True, "c": True, "d": False}
    out = M.pair_accuracy(correct, [("a", "b"), ("c", "d"), ("a", "missing")])
    assert out == {"pairs": 2, "both_correct": 0.5}


def test_permutation_consistency():
    assert M.permutation_consistency([["x", "x", "x"], ["x", "y", "x"]]) == 0.5


def test_error_correlation_distinguishes_shared_from_independent_errors():
    y = np.zeros(8, dtype=int)
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.9, 0.9, 0.9])
    shared = M.error_correlation(a, conf, a.copy(), y)
    assert shared["same_wrong_rate"] == 1.0 and shared["b_error_rate_overall"] == 0.5
    complementary = M.error_correlation(a, conf, 1 - a, y)
    assert complementary["same_wrong_rate"] == 0.0
    top = M.error_correlation(a, conf, np.array([1, 0, 0, 0, 0, 0, 0, 0]), y, top_fraction=0.25)
    assert top["examined"] == 1 and top["same_wrong_rate"] == 1.0


def test_family_bootstrap_is_wider_when_items_share_a_family():
    correct = np.array([True] * 50 + [False] * 50)
    metric = lambda idx: float(correct[idx].mean())  # noqa: E731
    independent = M.family_bootstrap([f"f{i}" for i in range(100)], metric, 500)
    clustered = M.family_bootstrap([f"f{i // 25}" for i in range(100)], metric, 500)
    assert independent["point"] == clustered["point"] == 0.5
    assert clustered["hi"] - clustered["lo"] > independent["hi"] - independent["lo"]
    assert clustered["n_families"] == 4
