from uratori.data.augment import augment, soften
from uratori.data.schema import Record


def _record(kind: str, **extra) -> Record:
    base = {
        "id": f"t-{kind}", "family_id": "f1", "split": "train", "state": {"根拠": "あ", "主張": "い"},
        "type": kind, "question": "`主張` は `根拠` から支持されるか", "label_source": "teacher",
        "provenance": {"source": "authored", "source_license": "CC-BY-4.0"},
        "tags": {"lang": "ja", "domain": "grounding", "difficulty": "clear"},
    }  # fmt: skip
    return Record.model_validate(base | extra)


def test_choice_shuffle_keeps_label_and_permutes_probs():
    r = _record("choice", criteria={"支持": "a", "矛盾": "b", "情報不足": "c"}, target_hard="矛盾",
                target_probs=[0.1, 0.7, 0.2])  # fmt: skip
    seen = set()
    for seed in range(20):
        (a,) = augment([r], seed, choice_shuffle=1.0)
        assert a.target_hard == "矛盾" and set(a.labels) == set(r.labels)
        assert dict(zip(a.labels, a.target_probs, strict=True)) == {
            "支持": 0.1,
            "矛盾": 0.7,
            "情報不足": 0.2,
        }
        assert a.criteria == {k: r.criteria[k] for k in a.labels}
        seen.add(tuple(a.labels))
    assert len(seen) > 1


def test_noul_drop_removes_criteria_only():
    r = _record("noul", criteria={"true": "成り立つ", "false": "成り立たない"}, target_hard="true")
    (a,) = augment([r], 0, noul_drop=1.0)
    assert a.criteria is None and a.target_hard == "true" and a.labels == r.labels
    (b,) = augment([r], 0, noul_drop=0.0)
    assert b.criteria == r.criteria


def test_score_is_untouched_and_soften_normalizes():
    r = _record("score", criteria=["低", "中", "高"], target_hard="2")
    (a,) = augment([r], 0, noul_drop=1.0, choice_shuffle=1.0)
    assert a.criteria == r.criteria
    out, used = soften([r], {"t-score": [0.2, 0.2, 0.4]})
    assert used == 1 and out[0].target_probs == [0.25, 0.25, 0.5]
    out, used = soften([r], {"t-score": [0.5, 0.5]})
    assert used == 0 and out[0].target_probs is None
