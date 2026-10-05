import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from uratori.serialize import InputTooLongError  # noqa: E402
from uratori.serve.app import create_app  # noqa: E402
from uratori.serve.predictor import temperature_for  # noqa: E402
from uratori.types import ChoiceQuestion, NoulQuestion  # noqa: E402


class FakePredictor:
    name = "uratori-test"

    def predict(self, state, questions):
        if isinstance(state, str) and len(state) > 50:
            raise InputTooLongError(5000, 2048)
        out = {}
        for qid, q in questions.items():
            n = len(q.labels)
            out[qid] = [0.7] + [0.3 / (n - 1)] * (n - 1)
        return out, 123


REQUEST = {
    "state": {"根拠": "受付は平日の9時から17時まで。", "主張": "土曜日も受け付ける。"},
    "model": "jev-latest",
    "questions": {
        "contradicts": {"type": "noul", "instructions": "`主張` は `根拠` と矛盾するか"},
        "relation": {
            "type": "choice",
            "instructions": "`主張` は `根拠` から支持されるか",
            "criteria": {"支持": "成り立つ", "矛盾": "食い違う", "情報不足": "決められない"},
        },
        "degree": {
            "type": "score",
            "instructions": "支持の度合い",
            "criteria": ["低い", "中", "高い"],
        },
    },
}


@pytest.fixture
def client():
    return TestClient(create_app(FakePredictor()))


def test_response_follows_the_systemone_shape(client):
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "uratori-test" and body["usage"] == {
        "input_tokens": 123,
        "output_tokens": 0,
    }
    assert body["answers"]["contradicts"] == {"type": "noul", "noul": pytest.approx(0.3)}
    relation = body["answers"]["relation"]
    assert relation["choice"] == "支持" and set(relation["probabilities"]) == {
        "支持",
        "矛盾",
        "情報不足",
    }
    assert relation["confidence"] == pytest.approx((0.7 - 1 / 3) / (2 / 3))
    degree = body["answers"]["degree"]
    assert degree["legend"] == {"0": "低い", "1": "中", "2": "高い"}
    assert degree["score"] == pytest.approx(0 * 0.7 + 1 * 0.15 + 2 * 0.15)


def test_invalid_requests_get_422(client):
    assert client.post("/v1/systemone", json=REQUEST | {"questions": {}}).status_code == 422
    too_many = {"type": "choice", "instructions": "q", "criteria": {str(i): "d" for i in range(9)}}
    assert (
        client.post("/v1/systemone", json=REQUEST | {"questions": {"x": too_many}}).status_code
        == 422
    )
    unknown = {"type": "rank", "instructions": "q"}
    assert (
        client.post("/v1/systemone", json=REQUEST | {"questions": {"x": unknown}}).status_code
        == 422
    )


def test_too_long_input_is_an_explicit_error_not_a_truncated_answer(client):
    response = client.post("/v1/systemone", json=REQUEST | {"state": "長" * 60})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "input_too_long"
    assert response.json()["error"]["max_tokens"] == 2048


def test_health_and_models(client):
    assert client.get("/healthz").json() == {"status": "ok", "model": "uratori-test"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "uratori-test"


def test_temperature_lookup_goes_from_specific_to_general():
    choice = ChoiceQuestion(instructions="q", criteria={"a": "x", "b": "y", "c": "z"})
    noul = NoulQuestion(instructions="q")
    temps = {"choice:3": 1.5, "choice": 1.2, "all": 2.0}
    assert temperature_for(choice, temps) == 1.5
    assert temperature_for(noul, temps) == 2.0
    assert temperature_for(noul, {}) == 1.0


def test_onnx_scores_are_gathered_at_marker_positions_and_normalized():
    import numpy as np

    from uratori.serve.onnx_export import onnx_scores_to_probs

    scores = np.array([[0.0, 2.0, 0.0, 1.0, 9.0], [5.0, 0.0, 5.0, 9.0, 9.0]], dtype=np.float32)
    input_ids = np.array([[7, 3, 8, 3, 0], [3, 7, 3, 0, 0]])
    first, second = onnx_scores_to_probs(scores, input_ids, marker_id=3)
    assert first == pytest.approx([np.e / (1 + np.e), 1 / (1 + np.e)])
    assert second == pytest.approx([0.5, 0.5])
