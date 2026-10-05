from pathlib import Path

import numpy as np
import pytest

from uratori.data.validate import load_records
from uratori.eval import metrics as M
from uratori.eval.calibrate import (
    apply_by_group,
    apply_temperature,
    fit_by_group,
    fit_temperature,
    group_keys,
)

SAMPLES = Path(__file__).parent.parent / "examples" / "sample_records.jsonl"


def test_temperature_one_is_identity_and_higher_is_flatter():
    p = np.array([0.7, 0.2, 0.1])
    assert apply_temperature(p, 1.0) == pytest.approx(p)
    hot = apply_temperature(p, 3.0)
    assert hot.sum() == pytest.approx(1.0) and hot.max() < p.max() and hot.argmax() == 0
    assert apply_temperature(p, 0.3).max() > p.max()


def make_overconfident(n: int, accuracy: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    probs, y = [], []
    for _ in range(n):
        pred = int(rng.integers(0, 3))
        p = np.full(3, 0.005)
        p[pred] = 0.99
        probs.append(p)
        y.append(pred if rng.random() < accuracy else (pred + 1) % 3)
    return probs, y


def test_fit_temperature_cools_overconfident_predictions():
    probs, y = make_overconfident(600, accuracy=0.7)
    t = fit_temperature(probs, y)
    assert t > 2.0
    calibrated = [apply_temperature(p, t) for p in probs]
    correct = np.array([int(p.argmax()) == t_ for p, t_ in zip(probs, y, strict=True)])
    before = M.ece(np.array([p.max() for p in probs]), correct)["ece"]
    after = M.ece(np.array([p.max() for p in calibrated]), correct)["ece"]
    assert before > 0.25 and after < 0.05
    assert M.nll(calibrated, np.array(y)) < M.nll(probs, np.array(y))


def test_fit_temperature_leaves_calibrated_predictions_alone():
    rng = np.random.default_rng(1)
    probs, y = [], []
    for _ in range(2000):
        p = rng.dirichlet([1.0, 1.0, 1.0])
        probs.append(p)
        y.append(int(rng.choice(3, p=p)))
    assert fit_temperature(probs, y) == pytest.approx(1.0, abs=0.12)


def test_groups_fall_back_from_type_and_option_count_to_all():
    records = load_records(SAMPLES)
    preds = {r.id: np.full(len(r.labels), 1 / len(r.labels)) for r in records}
    temps = fit_by_group(records, preds)
    assert set(temps) == {"all"}  # 7 件ではどの組も件数が足りない
    assert group_keys(records[0]) == ["noul:2", "noul", "all"]
    out = apply_by_group(records, preds, {"all": 2.0, "noul": 1.0})
    assert set(out) == set(preds)
    assert all(v.sum() == pytest.approx(1.0) for v in out.values())
