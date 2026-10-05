from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from uratori.data.rubrics import Rubric, load_rubrics
from uratori.serialize import build_input

RUBRICS = Path(__file__).parent.parent / "rubrics"


def test_all_rubrics_load_and_cover_the_priority_domains():
    rubrics = load_rubrics(RUBRICS)
    assert 20 <= len(rubrics) <= 30
    by_domain = Counter(r.domain for r in rubrics.values())
    assert min(by_domain["grounding"], by_domain["compare"], by_domain["rag"]) >= 6
    assert {r.type.value for r in rubrics.values()} == {"noul", "choice", "score"}
    assert sum(r.held_out for r in rubrics.values()) >= 4


def test_question_refers_only_to_fields_present_in_every_example():
    for rubric in load_rubrics(RUBRICS).values():
        refs = [part for i, part in enumerate(rubric.question.split("`")) if i % 2 == 1]
        assert refs, rubric.id
        for ex in rubric.examples:
            assert set(refs) <= set(ex.state), (rubric.id, refs, list(ex.state))


def test_examples_serialize_and_use_more_than_one_label():
    for rubric in load_rubrics(RUBRICS).values():
        labels = set()
        for i, ex in enumerate(rubric.examples):
            record = rubric.example_record(i)
            assert build_input(record.state, record.to_question()).n_options >= 2
            labels.add(ex.label)
        assert len(labels) >= 2, rubric.id


def test_example_with_unknown_label_is_rejected():
    row = {
        "id": "gr-test",
        "domain": "grounding",
        "type": "noul",
        "question": "`a` は正しいか",
        "examples": [
            {"kind": k, "state": {"a": "x"}, "label": "maybe"}
            for k in ("positive", "negative", "boundary")
        ],
    }
    with pytest.raises(ValidationError, match="候補"):
        Rubric.model_validate(row)


def test_rubric_without_boundary_example_is_rejected():
    row = {
        "id": "gr-test",
        "domain": "grounding",
        "type": "noul",
        "question": "`a` は正しいか",
        "examples": [{"kind": "positive", "state": {"a": "x"}, "label": "true"}],
    }
    with pytest.raises(ValidationError, match="例が足りない"):
        Rubric.model_validate(row)
