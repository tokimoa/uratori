from pathlib import Path

from uratori.data.convert import jnli_to_records
from uratori.data.rubrics import load_rubrics
from uratori.data.schema import Split

RUBRIC = load_rubrics(Path(__file__).parent.parent / "rubrics")["gr-support-3"]
ROWS = [
    {"sentence_pair_id": "0", "yjcaptions_id": "100124-104404-104405",
     "sentence1": "二人の男性がジャンボジェット機を見ています。",
     "sentence2": "2人の男性が、白い飛行機を眺めています。", "label": "neutral"},
    {"sentence_pair_id": "7", "yjcaptions_id": "100124-104406-104404",
     "sentence1": "男性が飛行機を見ています。", "sentence2": "誰も飛行機を見ていません。",
     "label": "contradiction"},
    {"sentence_pair_id": "9", "yjcaptions_id": "200001-1-2",
     "sentence1": "犬が走っています。", "sentence2": "動物が動いています。", "label": "entailment"},
]  # fmt: skip


def test_jnli_rows_become_support_3_records_grouped_by_image():
    records = jnli_to_records(ROWS, Split.TRAIN, RUBRIC, "jnli-train")
    assert [r.target_hard for r in records] == ["情報不足", "矛盾", "支持"]
    assert records[0].family_id == records[1].family_id != records[2].family_id
    first = records[0]
    assert first.id == "jnli-train-0" and first.question == RUBRIC.question
    assert first.state == {"根拠": ROWS[0]["sentence1"], "主張": ROWS[0]["sentence2"]}
    assert (
        first.label_source.value == "dataset" and first.provenance.source_license == "CC-BY-SA-4.0"
    )
    assert first.tags.rubric_id == "gr-support-3"


def test_mixed_conversion_uses_each_pair_once_and_maps_binary_labels():
    from collections import Counter

    from uratori.data.convert import jnli_to_mixed_records

    rubrics = load_rubrics(Path(__file__).parent.parent / "rubrics")
    rows = [dict(ROWS[i % 3], sentence_pair_id=str(i)) for i in range(300)]
    records = jnli_to_mixed_records(rows, Split.TRAIN, rubrics, "jnli-train", seed=1)
    assert len(records) == 300 and len({r.id for r in records}) == 300
    by_rubric = Counter(r.tags.rubric_id for r in records)
    assert by_rubric["gr-support-3"] > by_rubric["gr-support-noul"] > 40
    for record, row in zip(records, rows, strict=True):
        if record.tags.rubric_id == "gr-support-noul":
            assert record.target_hard == ("true" if row["label"] == "entailment" else "false")
            assert record.type.value == "noul" and record.labels == ["false", "true"]
        if record.tags.rubric_id == "gr-contradict-noul":
            assert record.target_hard == ("true" if row["label"] == "contradiction" else "false")
