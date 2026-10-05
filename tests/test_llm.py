import json

import httpx
import pytest

from uratori.llm.client import ChatClient, Endpoint
from uratori.llm.judge import JudgeParseError, build_messages, parse_judgement
from uratori.types import ChoiceQuestion, NoulQuestion

LABELS = ["支持", "矛盾", "情報不足"]


def test_parse_judgement_orders_probs_by_labels_and_normalizes():
    text = '前置き {"reason": "根拠にない", "label": "情報不足", "probs": {"情報不足": 0.9, "支持": 0.06, "矛盾": 0.06}}'
    j = parse_judgement(text, LABELS)
    assert j.label == "情報不足"
    assert j.probs == pytest.approx([0.06 / 1.02, 0.06 / 1.02, 0.9 / 1.02])
    assert sum(j.probs) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "text",
    [
        "JSON ではない",
        '{"label": "支持", "probs": {"支持": 1.0}}',
        '{"label": "保留", "probs": {"支持": 0.5, "矛盾": 0.3, "情報不足": 0.2}}',
        '{"label": "支持", "probs": {"支持": "高い", "矛盾": 0.3, "情報不足": 0.2}}',
        '{"label": "支持", "probs": {"支持": -1, "矛盾": 0.3, "情報不足": 0.2}}',
        '{"label": "支持", "probs": {"支持": 0.5, "矛盾": 0.3, "情報不足": 0.2',
    ],
)
def test_malformed_judgements_raise(text):
    with pytest.raises(JudgeParseError):
        parse_judgement(text, LABELS)


def test_messages_show_state_question_and_every_option():
    q = ChoiceQuestion(
        instructions="`主張` は支持されるか", criteria={"支持": "成り立つ", "矛盾": None}
    )
    user = build_messages({"根拠": "x", "主張": "y"}, q)[1]["content"]
    assert "［根拠］\nx" in user and '- "支持": 成り立つ' in user and '- "矛盾"' in user
    noul = build_messages("s", NoulQuestion(instructions="緊急か"))[1]["content"]
    assert '- "false": いいえ' in noul and '- "true": はい' in noul


def test_client_caches_identical_requests_and_counts_usage(tmp_path, monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    monkeypatch.setenv("TEST_KEY", "secret")
    endpoint = Endpoint(
        "t", "https://example.invalid/v1", "m", "TEST_KEY", price_in=1.0, price_out=2.0
    )
    client = ChatClient(endpoint, cache_dir=tmp_path)
    client._http = httpx.Client(base_url=endpoint.base_url, transport=httpx.MockTransport(handler))

    messages = [{"role": "user", "content": "hi"}]
    first = client.chat(messages, json_mode=True)
    second = client.chat(messages, json_mode=True)
    assert (first.text, first.cached, second.cached) == ("ok", False, True)
    assert len(calls) == 1 and calls[0]["response_format"] == {"type": "json_object"}
    assert (client.usage.calls, client.usage.cached_calls) == (2, 1)
    assert client.usage.cost_usd(endpoint) == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)


def test_client_requires_the_api_key_env(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        ChatClient(Endpoint("t", "https://example.invalid/v1", "m", "MISSING_KEY"))


def test_empty_responses_are_not_cached(tmp_path):
    n = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        )

    endpoint = Endpoint("t", "https://example.invalid/v1", "m")
    client = ChatClient(endpoint, cache_dir=tmp_path)
    client._http = httpx.Client(base_url=endpoint.base_url, transport=httpx.MockTransport(handler))
    result = client.chat([{"role": "user", "content": "hi"}])
    assert (result.text, result.finish_reason) == ("", "length")
    client.chat([{"role": "user", "content": "hi"}])
    assert n["calls"] == 2
