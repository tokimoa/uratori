import math
from pathlib import Path

import pytest

from uratori.data.human import (
    OwnerLabel,
    cohen_kappa,
    compare,
    finalize,
    read_labels,
    render_item,
    shuffled,
    write_labels,
)
from uratori.data.validate import load_records

SAMPLES = Path(__file__).parent.parent / "examples" / "sample_records.jsonl"


@pytest.fixture
def records():
    return load_records(SAMPLES)


def test_rendered_item_hides_the_draft_label_and_tags(records):
    record = next(r for r in records if r.id == "ex-0003")
    text = render_item(record, 1, 7)
    assert "質問: `事例` は `規約` の返金条件を満たすか" in text
    assert "3. 情報不足  確認できない条件が残っている。" in text
    assert "target" not in text and "hard" not in text and "条件" in text
    noul = render_item(next(r for r in records if r.type.value == "noul"), 2, 7)
    assert "1. いいえ" in noul and "2. はい" in noul


def test_shuffle_is_deterministic_and_keeps_every_record(records):
    a, b = shuffled(records), shuffled(list(reversed(records)))
    assert [r.id for r in a] == [r.id for r in b]
    assert sorted(r.id for r in a) == sorted(r.id for r in records)
    assert [r.id for r in a] != sorted(r.id for r in records)


def test_labels_round_trip(tmp_path):
    labels = {
        "a": OwnerLabel("支持"),
        "b": OwnerLabel("true", unsure=True, memo="時点が不明, 要確認"),
    }
    write_labels(tmp_path / "x" / "labels.csv", labels)
    assert read_labels(tmp_path / "x" / "labels.csv") == labels
    assert read_labels(tmp_path / "missing.csv") == {}


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert math.isnan(cohen_kappa(["a", "a"], ["a", "a"]))


def test_compare_reports_agreement_by_type(records):
    owner = {r.id: OwnerLabel(r.target_hard) for r in records}
    owner["ex-0001"] = OwnerLabel("true")
    out = compare(records, owner, {r.id: r.target_hard for r in records})
    assert out["n"] == 7 and out["agreement"] == pytest.approx(6 / 7)
    assert out["noul"]["agreement"] == 0.5 and out["choice"]["agreement"] == 1.0


def test_finalize_needs_adjudication_for_disagreements_and_unsure_items(records):
    owner = {r.id: OwnerLabel(r.target_hard) for r in records}
    owner["ex-0001"] = OwnerLabel("true", memo="10日は範囲外では")
    owner["ex-0004"] = OwnerLabel("矛盾", unsure=True)
    del owner["ex-0007"]

    done, pending = finalize(records, owner, {})
    assert sorted(pending) == ["ex-0001", "ex-0004", "ex-0007"]
    assert all(
        r.label_source.value == "human" and r.verification.status == "verified" for r in done
    )
    assert all(r.verification.by == ["owner", "claude-draft"] for r in done)

    done, pending = finalize(records, owner, {"ex-0001": "false", "ex-0004": "矛盾"})
    assert pending == ["ex-0007"]
    fixed = next(r for r in done if r.id == "ex-0001")
    assert fixed.target_hard == "false" and fixed.human_votes == ["true"]
    assert "裁定あり" in fixed.verification.note and "10日は範囲外では" in fixed.verification.note


def test_finalize_rejects_labels_outside_the_options(records):
    owner = {r.id: OwnerLabel(r.target_hard) for r in records}
    with pytest.raises(ValueError, match="候補にない"):
        finalize(records, owner | {"ex-0003": OwnerLabel("保留")}, {})


def test_consensus_keeps_llm_labels_marked_as_teacher_and_flags_split_items(records):
    from uratori.data.human import consensus

    draft = {r.id: r.target_hard for r in records}
    ref = dict(draft) | {"ex-0001": "true"}
    other = dict(draft) | {"ex-0003": "不成立"}
    judges = {"draft": draft, "ref": ref, "other": other}
    done, review = consensus(records, "ref", judges, {"ex-0001": "理由"}, ["draft", "ref"], {})
    by_id = {r.id: r for r in done}

    assert review == ["ex-0001"]
    assert all(r.label_source.value == "teacher" for r in done)
    split = by_id["ex-0001"]
    assert split.target_hard == "true" and split.verification.status == "disputed"
    assert split.verification.by == ["ref"] and split.verification.note == "理由"
    assert split.teacher_vote_distribution == pytest.approx([2 / 3, 1 / 3])
    # strong に入っていない判定者だけが違う件は、見直しの対象にしない
    assert by_id["ex-0003"].verification.status == "verified"
    assert by_id["ex-0003"].teacher_vote_distribution == pytest.approx([0, 1 / 3, 2 / 3])

    done, review = consensus(records, "ref", judges, {}, ["draft", "ref"], {"ex-0001": "false"})
    fixed = next(r for r in done if r.id == "ex-0001")
    assert review == [] and fixed.target_hard == "false"
    assert fixed.label_source.value == "human" and fixed.verification.by == ["owner"]
