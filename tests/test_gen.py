import json
import random
from pathlib import Path

import pytest

from uratori.data.rubrics import load_rubrics
from uratori.gen.pipeline import (
    Candidate,
    SourceDoc,
    Stats,
    _to_record,
    code_checks,
    shingles,
    verify,
)
from uratori.gen.plan import (
    GENRE_ONLY,
    SOURCE_FIELD,
    rubrics_for,
    sample_doc_seeds,
    sentence_excerpt,
    split_sentences,
)
from uratori.gen.prompts import (
    Draft,
    GenParseError,
    item_messages,
    parse_doc,
    parse_items,
    written_fields,
)
from uratori.llm.judge import JudgeParseError

RUBRICS = load_rubrics(Path(__file__).parent.parent / "rubrics")
DOC = SourceDoc(
    "d1",
    "出張旅費規程\n日帰り出張の日当は2,000円とする。宿泊を伴う場合は1泊につき3,500円を加える。",
    "社内規程",
    "generated:d1",
    "generated",
)


def test_every_trainable_rubric_has_a_source_field_that_exists_in_its_examples():
    trainable = {rid for rid, r in RUBRICS.items() if not r.held_out}
    assert trainable == set(SOURCE_FIELD)
    for rid, source_field in SOURCE_FIELD.items():
        keys = list(RUBRICS[rid].examples[0].state)
        assert source_field is None or source_field in keys, rid
        assert written_fields(RUBRICS[rid], source_field), rid
    assert set(GENRE_ONLY) <= trainable


def test_doc_seeds_are_reproducible_and_cover_every_genre():
    a, b = sample_doc_seeds(30, 1, "t"), sample_doc_seeds(30, 1, "t")
    assert a == b and len({s.doc_id for s in a}) == 30
    assert len({s.genre for s in a}) == 10
    assert sample_doc_seeds(30, 2, "t") != a


def test_genre_restricted_rubrics_are_only_used_with_matching_documents():
    ids = sorted(SOURCE_FIELD)
    rng = random.Random(0)
    assert "gr-decided-vs-considered" not in rubrics_for("FAQ", ids, len(ids), rng)
    assert "gr-decided-vs-considered" in rubrics_for("議事録", ids, len(ids), rng)
    assert not set(rubrics_for(None, ids, len(ids), rng)) & set(GENRE_ONLY)


def test_sentence_excerpt_returns_consecutive_sentences_from_the_text():
    text = "第1条 この規程は出張旅費について定める。\n日帰り出張の日当は2,000円とする。宿泊を伴う場合は3,500円を加える。以上。"
    assert len(split_sentences(text)) == 4
    excerpt = sentence_excerpt(text, 2, random.Random(0))
    assert excerpt in text.replace("\n", "") and excerpt.count("。") == 2
    assert sentence_excerpt("短い。", 2, random.Random(0)) is None


def test_item_prompt_shows_the_source_once_and_asks_only_for_written_fields():
    rubric = RUBRICS["gr-support-3"]
    user = item_messages(rubric, DOC.text, "根拠", ["矛盾", "支持", "情報不足"])[1]["content"]
    assert user.count(DOC.text) == 1 and "書くキー: `主張`" in user
    assert "作るラベル（この順に 1 件ずつ、計 3 件）: 矛盾、支持、情報不足" in user
    assert "根拠" not in user.split("例（形式の参考")[1].split("書くキー")[0].replace(
        "根拠だけ", ""
    )


def test_parse_items_keeps_positions_and_marks_malformed_items_as_none():
    payload = {
        "items": [
            {
                "label": "支持",
                "fields": {"主張": "日帰りの日当は2,000円だ。"},
                "reason": "r",
                "difficulty": "hard",
                "phenomena": ["数値", "存在しない現象"],
                "pair": None,
            },
            {"label": "保留", "fields": {"主張": "x"}},
            {"label": "矛盾", "fields": {"根拠": "余計なキー", "主張": "x"}},
            {"label": "矛盾", "fields": {"主張": "日帰りの日当は5,000円だ。"}, "pair": 0},
            "壊れた要素",
        ]
    }
    drafts = parse_items(
        json.dumps(payload, ensure_ascii=False), ["主張"], ["支持", "矛盾", "情報不足"]
    )
    assert [d is not None for d in drafts] == [True, False, False, True, False]
    assert (
        drafts[0].phenomena == ["数値"]
        and drafts[0].difficulty == "hard"
        and drafts[0].pair is None
    )
    assert drafts[3].pair == 0
    with pytest.raises(GenParseError):
        parse_items("JSON ではない", ["主張"], ["支持"])
    with pytest.raises(GenParseError):
        parse_doc('{"title": "t", "body": "短い"}')


def test_partner_requires_a_valid_index_and_a_different_label():
    from uratori.gen.pipeline import _partner

    a = Draft("支持", {"主張": "a"}, "", "clear", [])
    b = Draft("矛盾", {"主張": "b"}, "", "clear", [], pair=0)
    same = Draft("支持", {"主張": "c"}, "", "clear", [], pair=0)
    out_of_range = Draft("矛盾", {"主張": "d"}, "", "clear", [], pair=9)
    to_missing = Draft("矛盾", {"主張": "e"}, "", "clear", [], pair=4)
    drafts = [a, b, same, out_of_range, None, to_missing]
    assert [_partner(drafts, k) for k in (0, 1, 2, 3, 5)] == [None, 0, None, None, None]


def test_item_prompt_asks_for_pairs_target_phenomena_and_optional_injection():
    rubric = RUBRICS["gr-support-3"]
    user = item_messages(
        rubric,
        DOC.text,
        "根拠",
        ["矛盾", "支持", "情報不足"],
        n_pairs=2,
        target_phenomena=["数値", "二重否定"],
        inject=True,
    )[1]["content"]
    assert "対の件を 2 件追加する。全部で 5 件。" in user
    assert "できるだけ使う現象: 数値、二重否定" in user and "採点者への指示" in user
    plain = item_messages(rubric, DOC.text, "根拠", ["支持"])[1]["content"]
    assert "採点者への指示" not in plain


def test_copy_of_the_source_is_rejected_only_for_supported_labels():
    from uratori.gen.pipeline import copy_ratio

    copied = "宿泊を伴う場合は1泊につき3,500円を加える。"
    assert copy_ratio(DOC.text, copied) == 1.0 and copy_ratio(DOC.text, "短い") == 0.0
    candidates = [
        make_candidate("支持", copied, 0),
        make_candidate(
            "支持", "泊まりがけの出張では、1泊ごとに3,500円が上乗せされる決まりになっている。", 1
        ),
        make_candidate("矛盾", "宿泊を伴う場合は1泊につき4,500円を加える。", 2),
    ]
    stats = Stats()
    code_checks(candidates, [], stats)
    assert [c.status for c in candidates] == ["rejected", "pending", "pending"]
    assert candidates[0].reject_reason == "code: 原文の写し"


def test_dev_split_is_decided_per_document():
    from uratori.gen.plan import is_dev_doc

    ids = [f"doc-{i}" for i in range(2000)]
    share = sum(is_dev_doc(i) for i in ids) / len(ids)
    assert 0.05 < share < 0.11
    assert is_dev_doc("doc-1") == is_dev_doc("doc-1")


def make_candidate(label: str, claim: str, k: int) -> Candidate:
    rubric = RUBRICS["gr-support-3"]
    draft = Draft(label, {"主張": claim}, "理由", "clear", [])
    record = _to_record(DOC, rubric, DOC.text, "根拠", draft, f"t-{k}", [{"content": "sys"}])
    return Candidate(record=record, reason="理由")


def test_record_puts_the_source_in_the_source_field_and_keeps_rubric_order():
    record = make_candidate("支持", "日帰り出張の日当は2,000円である。", 0).record
    assert list(record.state) == ["根拠", "主張"] and record.state["根拠"] == DOC.text
    assert record.family_id == "syn-d1" and record.split.value == "train"
    assert record.question == RUBRICS["gr-support-3"].question


def test_code_checks_reject_duplicates_short_text_and_overlap_with_protected_set():
    protected = [
        make_candidate(
            "支持", "宿泊を伴う出張では1泊につき3,500円が日当に加わると定められている。", 9
        ).record
    ]
    candidates = [
        make_candidate("矛盾", "日帰り出張の日当は9,000円である。", 0),
        make_candidate("矛盾", "日帰り出張の日当は9,000円である。", 1),
        make_candidate("矛盾", "違う", 2),
    ]
    stats = Stats()
    code_checks(candidates, [], stats)
    assert [c.status for c in candidates] == ["pending", "rejected", "rejected"]
    assert "重複" in candidates[1].reject_reason and "短すぎる" in candidates[2].reject_reason

    fresh = [make_candidate("矛盾", "日帰り出張の日当は9,000円である。", 3)]
    code_checks(fresh, protected, Stats())
    assert fresh[0].status == "rejected" and "人手評価セット" in fresh[0].reject_reason
    assert len(shingles("あ" * 25)) == 1 and shingles("短い") == set()


def test_verify_accepts_only_when_the_blind_judge_agrees():
    candidates = [
        make_candidate("支持", "日帰り出張の日当は2,000円である。", 0),
        make_candidate("矛盾", "日帰り出張の日当は5,000円である。", 1),
        make_candidate("情報不足", "海外出張の日当は別に定める。", 2),
    ]

    def fake_judge(record):
        if "海外" in record.state["主張"]:
            raise JudgeParseError("空")
        return "支持", [0.8, 0.1, 0.1]

    stats = Stats()
    verify(candidates, fake_judge, "verifier-x", "generator-y", stats, workers=1)
    assert [c.status for c in candidates] == ["accepted", "rejected", "rejected"]
    ok = candidates[0].record
    assert ok.verification.status == "verified" and ok.verification.by == [
        "generator-y",
        "verifier-x",
    ]
    assert (
        ok.teacher_reported_probs == [0.8, 0.1, 0.1]
        and ok.provenance.teacher_model == "generator-y"
    )
    assert candidates[1].record.verification.status == "disputed"
    assert candidates[1].reject_reason == "verify: 判定者は 支持"
    assert stats.counts == {"accepted": 1, "reject_verify_mismatch": 1, "reject_verify_parse": 1}


def test_open_question_is_parsed_and_validated():
    from uratori.gen.prompts import open_messages, parse_open

    payload = {
        "type": "choice",
        "question": "`資料` に照らして、`申請` は誰の承認が必要か",
        "criteria": {
            "課長": "課長だけ",
            "部長": "部長の承認も要る",
            "判断できない": "金額が分からない",
        },
        "written_keys": ["申請"],
        "domain": "grounding",
        "items": [
            {
                "label": "課長",
                "fields": {"申請": "3万円の備品を買いたい。"},
                "reason": "r",
                "difficulty": "clear",
                "phenomena": ["数値"],
                "pair": None,
            },
            {
                "label": "部長",
                "fields": {"申請": "30万円の備品を買いたい。"},
                "reason": "r",
                "difficulty": "hard",
                "phenomena": ["数値"],
                "pair": 0,
            },
            {"label": "社長", "fields": {"申請": "x"}},
        ],
    }
    spec, drafts = parse_open(json.dumps(payload, ensure_ascii=False), "資料")
    assert spec.type == "choice" and spec.labels == ["課長", "部長", "判断できない"]
    assert spec.written_keys == ["申請"] and [d is not None for d in drafts] == [True, True, False]
    assert drafts[1].pair == 0

    for broken in (
        payload | {"written_keys": ["資料"]},
        payload | {"type": "rank"},
        payload | {"criteria": {"課長": "1 つだけ"}},
        payload | {"items": payload["items"][:1]},
    ):
        with pytest.raises(GenParseError):
            parse_open(json.dumps(broken, ensure_ascii=False), "資料")

    user = open_messages(DOC.text, "資料", "承認や許可が必要か、誰の承認か", ["数値", "例外条件"])[
        1
    ]["content"]
    assert "`資料`" in user and "承認や許可" in user


def test_weighted_phenomena_sampling_returns_distinct_known_names():
    from uratori.gen.plan import PHENOMENA, sample_phenomena

    rng = random.Random(0)
    for _ in range(200):
        picked = sample_phenomena(rng)
        assert len(picked) == 2 and len(set(picked)) == 2 and set(picked) <= PHENOMENA
        assert "指示の混入" not in picked


def test_open_parser_tolerates_common_deviations():
    from uratori.gen.prompts import parse_open

    payload = {
        "type": "score",
        "question": "`申請` はどの程度急ぎか",
        "criteria": {"1": "数日以内", "0": "急がない", "2": "当日中"},
        "written_keys": ["資料", "申請"],
        "items": [
            {"label": "0", "fields": {"申請": "来月でよいので備品を買いたい。"}},
            {"label": "2", "fields": {"申請": "本日中に備品が必要です。"}},
        ],
    }
    spec, drafts = parse_open(json.dumps(payload, ensure_ascii=False), "資料")
    assert spec.written_keys == ["申請"]
    assert spec.criteria == ["急がない", "数日以内", "当日中"] and spec.labels == ["0", "1", "2"]
    assert all(d is not None for d in drafts)


def test_family_items_resolve_document_references_and_validate_keys():
    from uratori.gen.family import family_messages, parse_family

    picked = {rid: RUBRICS[rid] for rid in ("gr-support-3", "cmp-contradict-noul")}
    doc = "出張旅費規程。日帰り出張の日当は2,000円とする。ただし管理職には支給しない。"
    payload = {
        "documents": {"D1": doc, "D1b": doc.replace("2,000", "3,000")},
        "items": [
            {
                "rubric": "gr-support-3",
                "state": {"根拠": "@D1", "主張": "課長が日帰り出張をしても日当は出ない。"},
                "label": "支持",
                "reason": "r",
                "difficulty": "hard",
                "phenomena": ["例外条件"],
                "pair": None,
            },
            {
                "rubric": "cmp-contradict-noul",
                "state": {"文書A": "@D1", "文書B": "@D1b"},
                "label": "true",
                "pair": None,
            },
            {"rubric": "gr-support-3", "state": {"根拠": "@D9", "主張": "x"}, "label": "支持"},
            {"rubric": "gr-support-3", "state": {"根拠": "@D1"}, "label": "支持"},
            {"rubric": "gr-number-match", "state": {"根拠": "@D1", "主張": "x"}, "label": "true"},
            {"rubric": "gr-support-3", "state": {"根拠": "@D1", "主張": "x"}, "label": "保留"},
        ],
    }
    items = parse_family(json.dumps(payload, ensure_ascii=False), picked)
    assert [x is not None for x in items] == [True, True, False, False, False, False]
    assert items[0]["state"]["根拠"] == doc and list(items[0]["state"]) == ["根拠", "主張"]
    assert items[1]["state"]["文書B"].count("3,000") == 1

    user = family_messages(
        "社内規程", "製造業", "経費精算", "short", list(picked.values()), 2, ["数値"]
    )[1]["content"]
    assert "rubric: gr-support-3" in user and "state のキー: 文書A、文書B" in user
