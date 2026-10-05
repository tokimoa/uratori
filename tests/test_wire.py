import pytest
from pydantic import ValidationError

from uratori.serve.wire import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    answer_from_probs,
)

# https://docs.typesafe.ai/api.md の例と同じ形
DOC_REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        }
    },
}


def test_documented_request_parses():
    req = SystemOneRequest.model_validate(DOC_REQUEST)
    assert req.questions["is_urgent"].labels == ["false", "true"]


def test_request_needs_at_least_one_question():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(DOC_REQUEST | {"questions": {}})


def test_noul_answer_is_probability_of_true():
    q = SystemOneRequest.model_validate(DOC_REQUEST).questions["is_urgent"]
    answer = answer_from_probs(q, [0.05, 0.95])
    assert isinstance(answer, NoulAnswer)
    assert answer.model_dump() == {"type": "noul", "noul": 0.95}


def test_choice_answer():
    req = DOC_REQUEST | {
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "What is the request?",
                "criteria": {"refund": "money back", "rebooking": "new flight", "info": "question"},
            }
        }
    }
    q = SystemOneRequest.model_validate(req).questions["kind"]
    answer = answer_from_probs(q, [0.6, 0.3, 0.1])
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "refund"
    assert answer.probabilities == {"refund": 0.6, "rebooking": 0.3, "info": 0.1}
    assert answer.confidence == pytest.approx(0.4)


def test_score_answer_matches_documented_shape():
    levels = ["Cosmetic", "Degraded, workaround exists", "Blocking"]
    req = DOC_REQUEST | {
        "questions": {"sev": {"type": "score", "instructions": "How severe?", "criteria": levels}}
    }
    q = SystemOneRequest.model_validate(req).questions["sev"]
    answer = answer_from_probs(q, [0.0, 0.57, 0.43])
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(1.43)
    assert answer.legend == {"0": levels[0], "1": levels[1], "2": levels[2]}
    assert answer.probabilities == {"0": 0.0, "1": 0.57, "2": 0.43}


def test_wrong_number_of_probabilities_is_rejected():
    q = SystemOneRequest.model_validate(DOC_REQUEST).questions["is_urgent"]
    with pytest.raises(ValueError, match="候補数"):
        answer_from_probs(q, [1.0])
