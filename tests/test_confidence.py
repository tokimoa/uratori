import pytest

from uratori.confidence import choice_confidence, noul_confidence, score_confidence, score_value


def test_noul_confidence():
    assert noul_confidence(0.5) == 0.0
    assert noul_confidence(1.0) == 1.0
    assert noul_confidence(0.1) == pytest.approx(0.8)


def test_choice_confidence_depends_only_on_peak():
    # 公開ドキュメントの例。どちらも 0.4 になる
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    assert choice_confidence([0.25] * 4) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0]) == pytest.approx(1.0)


def test_score_confidence_documented_examples():
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_score_confidence_penalizes_distance():
    near = score_confidence([0.7, 0.3, 0.0, 0.0, 0.0])
    far = score_confidence([0.7, 0.0, 0.0, 0.0, 0.3])
    assert near > far


def test_score_value_is_expected_level():
    assert score_value([0.0, 0.57, 0.43]) == pytest.approx(1.43)
