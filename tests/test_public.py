import json

from uratori.data.public import record_from_public_row
from uratori.data.validate import validate_files
from uratori.serialize import OPTION_MARKER, build_input

COMMON = {
    "agreement": "unanimous",
    "domain": "grounding",
    "difficulty": "hard",
    "phenomena": ["数値"],
    "template": None,
    "pair_of": None,
}
STATE = [
    {"name": "根拠", "text": "未使用の商品に限り、購入から7日以内であれば返金できる。"},
    {"name": "主張", "text": "購入から10日後でも、未使用なら返金できる。"},
]
ROWS = [
    COMMON
    | {
        "id": "test-0001",
        "family_id": "test-refund",
        "type": "noul",
        "state": STATE,
        "question": "`主張` は `根拠` から支持されるか",
        "options": [
            {"label": "false", "description": "成り立つと言えない部分がある。"},
            {"label": "true", "description": "根拠だけから成り立つ。"},
        ],
        "answer": "false",
        "answer_index": 0,
    },
    COMMON
    | {
        "id": "test-0002",
        "family_id": "test-refund",
        "type": "noul",
        "state": STATE,
        "question": "`主張` は `根拠` と矛盾するか",
        "options": [
            {"label": "false", "description": ""},
            {"label": "true", "description": ""},
        ],
        "answer": "true",
        "answer_index": 1,
        "agreement": "split",
        "pair_of": "test-0001",
    },
    COMMON
    | {
        "id": "test-0003",
        "family_id": "test-refund",
        "type": "choice",
        "state": STATE,
        "question": "`主張` は `根拠` から支持されるか",
        "options": [
            {"label": "支持", "description": "成り立つ。"},
            {"label": "矛盾", "description": "食い違う。"},
            {"label": "情報不足", "description": ""},
        ],
        "answer": "矛盾",
        "answer_index": 1,
        "template": "gr-support-3",
    },
    COMMON
    | {
        "id": "test-0004",
        "family_id": "test-refund",
        "type": "score",
        "state": STATE,
        "question": "`主張` はどの程度 `根拠` から支持されるか",
        "options": [
            {"label": "0", "description": "支持されない。"},
            {"label": "1", "description": "一部だけ支持される。"},
            {"label": "2", "description": "全体が支持される。"},
        ],
        "answer": "0",
        "answer_index": 0,
    },
]


def test_public_rows_become_records_with_the_same_options_and_answer():
    for row in ROWS:
        record = record_from_public_row(row, "test")
        assert record.labels == [o["label"] for o in row["options"]]
        assert record.target_index == row["answer_index"]
        assert record.state == {"根拠": STATE[0]["text"], "主張": STATE[1]["text"]}
        assert record.split.value == "test" and record.label_source.value == "teacher"
        assert record.provenance.source == "tokimoa/uratori-ja-eval"
        # モデルに渡る文字列は、候補ごとに「表示名: 説明」とマーカーが 1 つ
        text = build_input(record.state, record.to_question()).text
        assert text.count(OPTION_MARKER) == len(row["options"])
        for option in row["options"]:
            if option["description"]:
                assert f": {option['description']} {OPTION_MARKER}" in text


def test_empty_descriptions_and_side_fields_are_mapped():
    plain, split, choice, _ = (record_from_public_row(row, "test") for row in ROWS)
    assert split.criteria is None and plain.criteria == {
        "false": "成り立つと言えない部分がある。",
        "true": "根拠だけから成り立つ。",
    }
    assert choice.criteria["情報不足"] is None and choice.tags.rubric_id == "gr-support-3"
    assert plain.verification.status == "verified" and split.verification.status == "disputed"
    assert split.provenance.contrast_of == "test-0001"
    assert plain.tags.phenomena == ["数値"] and plain.tags.difficulty == "hard"


def test_converted_records_pass_the_validator(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        "".join(record_from_public_row(row, "test").model_dump_json() + "\n" for row in ROWS),
        encoding="utf-8",
    )
    report = validate_files([path])
    assert report.ok and report.n_records == len(ROWS)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["id"] == "test-0001"
