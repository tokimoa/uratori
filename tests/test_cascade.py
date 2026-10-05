import numpy as np

from uratori.cascade import Layer1, layer1_check, numbers_in, route


def test_numbers_are_normalized():
    assert numbers_in("会員数は1,250,000人、約125万人。") == {"1250000"}
    assert numbers_in("３，５００円と3500円") == {"3500"}
    assert numbers_in("2億円") == {"200000000"}


def test_layer1_flags_only_numbers_absent_from_the_source():
    state = {
        "根拠": "日当は2,000円。第3条。",
        "主張": "日当は2000円で、第3条に基づき5,000円が加算される。",
    }
    out = layer1_check(state, ["根拠"])
    assert out.unsupported_numbers == ["5000"] and out.flagged
    assert not layer1_check({"根拠": "日当は2,000円。", "主張": "日当は2000円。"}, ["根拠"]).flagged


def test_route_escalates_on_rules_or_low_confidence():
    high, low = np.array([0.95, 0.05]), np.array([0.55, 0.45])
    assert route(high, Layer1([]), 0.8, True) == "accept"
    assert route(low, Layer1([]), 0.8, True) == "escalate"
    assert route(high, Layer1(["5000"]), 0.8, True) == "escalate"
    assert route(high, Layer1(["5000"]), 0.8, False) == "accept"
