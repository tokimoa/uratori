import random
from pathlib import Path

from uratori.data.rubrics import load_rubrics
from uratori.gen.program import number_pairs, perturb_number, restyle_number

RUBRIC = load_rubrics(Path(__file__).parent.parent / "rubrics")["gr-number-match"]
EVIDENCE = "会員数は2024年度末に1,250,000人に達し、前年度から8%増えた。"


def test_perturbed_number_never_matches_a_number_in_the_evidence():
    for seed in range(50):
        new = perturb_number(
            "2024年度末の会員数は1,250,000人だった。", EVIDENCE, random.Random(seed)
        )
        assert new is not None and new != "2024年度末の会員数は1,250,000人だった。"
        changed = [n for n in ("2024", "1,250,000") if n not in new]
        assert len(changed) == 1  # 変えるのは 1 箇所だけ
        assert "1250000" not in new.replace(",", "") or "2024" not in new
    assert perturb_number("数値のない主張。", EVIDENCE, random.Random(0)) is None


def test_small_numbers_move_by_a_few_and_keep_comma_style():
    new = perturb_number("増加率は8%だった。", EVIDENCE, random.Random(1))
    assert new != "増加率は8%だった。" and new.endswith("%だった。")
    big = perturb_number("会員数は1,250,000人。", "会員数は1,250,000人。", random.Random(2))
    assert "," in big and "1,250,000" not in big


def test_restyle_keeps_the_value():
    for seed in range(20):
        new = restyle_number("会員数は1,250,000人。", random.Random(seed))
        normalized = new.translate(str.maketrans("０１２３４５６７８９，", "0123456789,")).replace(
            ",", ""
        )
        assert "1250000" in normalized and new != "会員数は1,250,000人。"


def test_number_pairs_only_use_true_number_match_records():
    true_rec = RUBRIC.example_record(0).model_copy(update={"id": "a"})
    false_rec = RUBRIC.example_record(1).model_copy(update={"id": "b"})
    out = number_pairs([true_rec, false_rec], seed=0)
    assert {r.id for r in out} == {"a-num", "a-fmt"}
    by_id = {r.id: r for r in out}
    assert by_id["a-num"].target_hard == "false" and by_id["a-fmt"].target_hard == "true"
    assert all(r.label_source.value == "program" and r.provenance.contrast_of == "a" for r in out)
    assert all(r.family_id == true_rec.family_id for r in out)
    assert by_id["a-num"].state["根拠"] == true_rec.state["根拠"]
