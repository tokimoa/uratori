import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from uratori.data.schema import Record
from uratori.data.validate import load_records, validate_files

SAMPLES = Path(__file__).parent.parent / "examples" / "sample_records.jsonl"


def base(**over):
    row = {
        "id": "r1",
        "family_id": "f1",
        "split": "train",
        "state": {"根拠": "購入から7日以内なら返金できる。", "主張": "10日後でも返金できる。"},
        "type": "choice",
        "question": "`主張` は `根拠` から支持されるか",
        "criteria": {"支持": "成り立つ", "矛盾": "食い違う", "情報不足": "決められない"},
        "target_hard": "矛盾",
        "label_source": "human",
        "provenance": {"source": "authored"},
        "tags": {"lang": "ja", "domain": "grounding"},
    }
    return row | over


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_samples_cover_every_type_and_pass():
    records = load_records(SAMPLES)
    by_type = {t: sum(r.type == t for r in records) for t in ("noul", "choice", "score")}
    assert all(n >= 2 for n in by_type.values()), by_type
    assert validate_files([SAMPLES]).ok


def test_labels_and_target_index_follow_criteria_order():
    rec = Record.model_validate(base())
    assert rec.labels == ["支持", "矛盾", "情報不足"]
    assert rec.target_index == 1
    noul = Record.model_validate(base(type="noul", criteria=None, target_hard="true"))
    assert noul.labels == ["false", "true"] and noul.target_index == 1


@pytest.mark.parametrize(
    "over, message",
    [
        ({"target_hard": "保留"}, "候補"),
        ({"target_probs": [0.5, 0.5]}, "長さ"),
        ({"target_probs": [0.5, 0.4, 0.2]}, "総和"),
        ({"target_probs": [1.2, -0.2, 0.0]}, "範囲外"),
        ({"teacher_reported_probs": [0.9, 0.0, 0.0]}, "総和"),
        ({"human_votes": ["矛盾", "わからない"]}, "human_votes"),
        ({"target_hard": None}, "ラベルが必要"),
        ({"target_hard": None, "target_probs": [0.1, 0.8, 0.1]}, "target_hard が必要"),
        ({"label_source": "none"}, "none"),
        ({"type": "score"}, "score"),
        ({"evidence_spans": [{"path": "根拠", "start": 0, "end": 999}]}, "範囲外"),
        ({"evidence_spans": [{"path": "無い", "start": 0, "end": 1}]}, "path"),
        ({"evidence_spans": [{"start": 0, "end": 1}]}, "path が必要"),
        ({"extra_field": 1}, "Extra"),
    ],
)
def test_invalid_records_are_rejected(over, message):
    with pytest.raises(ValidationError, match=message):
        Record.model_validate(base(**over))


def test_teacher_only_soft_label_and_unlabeled_pool_are_allowed():
    Record.model_validate(
        base(target_hard=None, target_probs=[0.1, 0.8, 0.1], label_source="teacher")
    )
    Record.model_validate(base(split="pool", target_hard=None, label_source="none"))


def test_soft_hard_mismatch_is_a_warning_not_an_error(tmp_path):
    path = write(tmp_path, "a.jsonl", [base(target_probs=[0.7, 0.2, 0.1])])
    report = validate_files([path])
    assert report.ok
    assert any("最頻ラベル" in w for w in report.warnings)


def test_family_must_not_cross_train_and_eval(tmp_path):
    train = write(tmp_path, "train.jsonl", [base()])
    test = write(tmp_path, "test.jsonl", [base(id="r2", split="test", question="別の質問")])
    report = validate_files([train, test])
    assert not report.ok
    assert any("またがっている" in e for e in report.errors)


def test_duplicate_id_and_duplicate_content_are_errors(tmp_path):
    path = write(tmp_path, "a.jsonl", [base(), base(), base(id="r3")])
    errors = validate_files([path]).errors
    assert any("id 'r1'" in e for e in errors)
    assert sum("完全に重複" in e for e in errors) == 2


def test_schema_error_reports_file_and_line(tmp_path):
    path = write(tmp_path, "a.jsonl", [base(), base(id="r2", target_hard="保留", question="q2")])
    report = validate_files([path])
    assert report.n_records == 1
    assert len(report.errors) == 1 and "a.jsonl:2" in report.errors[0]


def test_missing_contrast_partner_is_a_warning(tmp_path):
    row = base(provenance={"source": "authored", "contrast_of": "missing"})
    report = validate_files([write(tmp_path, "a.jsonl", [row])])
    assert report.ok and any("contrast_of" in w for w in report.warnings)
