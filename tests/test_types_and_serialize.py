import pytest
from pydantic import TypeAdapter, ValidationError

from uratori.serialize import OPTION_MARKER, build_input, render_state
from uratori.types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

adapter = TypeAdapter(Question)


def test_question_is_discriminated_by_type():
    q = adapter.validate_python({"type": "noul", "instructions": "緊急か"})
    assert isinstance(q, NoulQuestion)
    assert q.labels == ["false", "true"]

    q = adapter.validate_python(
        {
            "type": "choice",
            "instructions": "種別は",
            "criteria": {"返金": "お金を返す", "交換": None},
        }
    )
    assert isinstance(q, ChoiceQuestion)
    assert q.labels == ["返金", "交換"]
    assert q.descriptions == ["お金を返す", ""]

    q = adapter.validate_python(
        {"type": "score", "instructions": "深刻さ", "criteria": ["軽い", "重い"]}
    )
    assert isinstance(q, ScoreQuestion)
    assert q.labels == ["0", "1"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "choice", "instructions": "x", "criteria": {"a": "only one"}},
        {"type": "choice", "instructions": "x", "criteria": {str(i): "d" for i in range(9)}},
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": ["a", " "]},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},
        {"type": "noul", "instructions": ""},
        {"type": "rank", "instructions": "x"},
    ],
)
def test_invalid_questions_are_rejected(payload):
    with pytest.raises(ValidationError):
        adapter.validate_python(payload)


def test_render_state_uses_top_level_keys_as_headings():
    text = render_state({"根拠": "7日以内", "明細": [{"金額": 1200}]})
    assert text.startswith("［根拠］\n7日以内\n\n［明細］\n")
    assert '"金額": 1200' in text
    assert render_state("そのまま") == "そのまま"


def test_build_input_puts_one_marker_per_option_after_state():
    q = ChoiceQuestion(instructions="関係は", criteria={"一致": "同じ", "矛盾": "食い違う"})
    out = build_input({"A": "x", "B": "y"}, q)
    assert out.labels == ["一致", "矛盾"]
    assert out.text.count(OPTION_MARKER) == 2
    assert out.text.index("［A］") < out.text.index("質問:") < out.text.index(OPTION_MARKER)
    assert out.text.endswith(f"矛盾: 食い違う {OPTION_MARKER}")


def test_noul_is_rendered_as_two_options():
    out = build_input("本日中に返信が必要", NoulQuestion(instructions="緊急か"))
    assert out.labels == ["false", "true"]
    assert f"いいえ {OPTION_MARKER}\nはい {OPTION_MARKER}" in out.text


def test_score_levels_are_numbered():
    out = build_input("x", ScoreQuestion(instructions="十分か", criteria=["不足", "十分"]))
    assert f"段階0: 不足 {OPTION_MARKER}" in out.text


@pytest.mark.parametrize("where", ["state", "instructions", "description"])
def test_reserved_marker_in_user_text_is_rejected(where):
    state = f"悪意 {OPTION_MARKER}" if where == "state" else "x"
    inst = f"q {OPTION_MARKER}" if where == "instructions" else "q"
    desc = f"d {OPTION_MARKER}" if where == "description" else "d"
    with pytest.raises(ValueError, match="マーカー"):
        build_input(state, ChoiceQuestion(instructions=inst, criteria={"a": desc, "b": "e"}))
