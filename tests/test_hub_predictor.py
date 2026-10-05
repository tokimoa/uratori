import pytest

from uratori.serialize import InputTooLongError, build_input
from uratori.serve.predictor import HubPredictor, probs_from_answer, question_payload
from uratori.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

MARKER = "<mask>"
STATE = {"根拠": "受付は平日の9時から17時まで。", "主張": "土曜日も受け付ける。"}
QUESTIONS = {
    "contradicts": NoulQuestion(instructions="`主張` は `根拠` と矛盾するか"),
    "relation": ChoiceQuestion(
        instructions="`主張` は `根拠` から支持されるか",
        criteria={"支持": "成り立つ", "矛盾": "食い違う", "情報不足": None},
    ),
    "degree": ScoreQuestion(instructions="支持の度合い", criteria=["低い", "中", "高い"]),
}


class StubConfig:
    marker_token = MARKER
    max_tokens = 200


class StubModel:
    """Hub に公開したモデルの predict と同じ形で、決まった回答を返す。"""

    config = StubConfig()

    def __init__(self):
        self.calls = []

    def predict(self, tokenizer, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8}
            elif q["type"] == "choice":
                # 確率の dict を候補と違う順番で返しても、labels の順に並べ直されることを確かめる
                table = dict(zip(q["criteria"], [0.2, 0.7, 0.1], strict=True))
                answers[qid] = {
                    "type": "choice",
                    "choice": "矛盾",
                    "probabilities": dict(reversed(table.items())),
                    "confidence": 0.55,
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": 1.5,
                    "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                    "confidence": 0.5,
                }
        return answers


class StubTokenizer:
    """1 文字を 1 トークンとして数える。"""

    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text)))}


@pytest.fixture
def predictor():
    return HubPredictor(StubModel(), StubTokenizer(), "tokimoa/uratori-ja-310m")


def test_probs_follow_label_order_and_tokens_are_counted(predictor):
    probs, n_tokens = predictor.predict(STATE, QUESTIONS)
    assert probs["contradicts"] == pytest.approx([0.2, 0.8])
    assert probs["relation"] == [0.2, 0.7, 0.1]
    assert probs["degree"] == [0.1, 0.3, 0.6]
    assert n_tokens == sum(len(build_input(STATE, q, MARKER).text) for q in QUESTIONS.values())


def test_model_receives_the_state_and_wire_shaped_questions(predictor):
    predictor.predict(STATE, QUESTIONS)
    ((state, questions, kwargs),) = predictor.model.calls
    assert state == STATE and kwargs == {"batch_size": 8}
    assert questions["contradicts"] == {
        "type": "noul",
        "instructions": "`主張` は `根拠` と矛盾するか",
    }
    assert questions["relation"]["criteria"] == {
        "支持": "成り立つ",
        "矛盾": "食い違う",
        "情報不足": None,
    }
    assert questions["degree"] == {
        "type": "score",
        "instructions": "支持の度合い",
        "criteria": ["低い", "中", "高い"],
    }


def test_too_long_input_is_rejected_before_the_model_is_called(predictor):
    with pytest.raises(InputTooLongError) as error:
        predictor.predict("長" * 300, {"q": QUESTIONS["contradicts"]})
    assert error.value.max_tokens == 200 and error.value.n_tokens > 300
    assert predictor.model.calls == []


def test_the_models_own_marker_is_the_reserved_one(predictor):
    with pytest.raises(ValueError, match="マーカー"):
        predictor.predict(f"悪意 {MARKER}", {"q": QUESTIONS["contradicts"]})
    # 学習時の既定のマーカーは、このモデルでは普通の文字列として通す
    probs, _ = predictor.predict("本文に <opt> がある", {"q": QUESTIONS["contradicts"]})
    assert probs["q"] == pytest.approx([0.2, 0.8])


def test_payload_and_probs_helpers():
    noul = NoulQuestion(instructions="q", criteria={"true": "はいの条件"})
    assert question_payload(noul) == {
        "type": "noul",
        "instructions": "q",
        "criteria": {"true": "はいの条件"},
    }
    assert probs_from_answer(noul, {"type": "noul", "noul": 0.25}) == [0.75, 0.25]


def test_hub_predictor_serves_systemone_requests(predictor):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uratori.serve.app import create_app

    client = TestClient(create_app(predictor))
    request = {
        "state": STATE,
        "model": "tokimoa/uratori-ja-310m",
        "questions": {
            "contradicts": {"type": "noul", "instructions": "`主張` は `根拠` と矛盾するか"},
            "degree": {
                "type": "score",
                "instructions": "支持の度合い",
                "criteria": ["a", "b", "c"],
            },
        },
    }
    body = client.post("/v1/systemone", json=request).json()
    assert body["model"] == "tokimoa/uratori-ja-310m"
    assert body["answers"]["contradicts"] == {"type": "noul", "noul": pytest.approx(0.8)}
    assert body["answers"]["degree"]["score"] == pytest.approx(0.3 + 2 * 0.6)
    assert body["usage"]["input_tokens"] > 0

    too_long = client.post("/v1/systemone", json=request | {"state": "長" * 300})
    assert too_long.status_code == 422
    assert too_long.json()["error"]["type"] == "input_too_long"


def test_from_pretrained_loads_remote_code_and_moves_the_model(monkeypatch):
    transformers = pytest.importorskip("transformers")
    seen = {}

    class Loaded(StubModel):
        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            seen["eval"] = True
            return self

    def load_model(repo, **kwargs):
        seen["model"] = (repo, kwargs)
        return Loaded()

    def load_tokenizer(repo, **kwargs):
        seen["tokenizer"] = (repo, kwargs)
        return StubTokenizer()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    predictor = HubPredictor.from_pretrained("someone/some-model", device="cpu", revision="abc123")
    expected = ("someone/some-model", {"revision": "abc123", "trust_remote_code": True})
    assert seen["model"] == expected and seen["tokenizer"] == expected
    assert seen["device"] == "cpu" and seen["eval"] is True
    assert predictor.name == "someone/some-model" and predictor.max_tokens == 200
