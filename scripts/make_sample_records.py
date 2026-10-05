"""形式の見本を examples/sample_records.jsonl に書き出す。

ラベルは見本用に Claude が付けたもので、人手の確認を経ていない。評価や学習には使わない。
"""

from pathlib import Path

from uratori.data.schema import Record

PROV = {
    "source": "authored",
    "source_license": "CC0-1.0",
    "generator": "handwritten-example",
    "teacher_model": "claude-fable-5-1",
}
REFUND_RULE = "未使用の商品に限り、購入から7日以内であれば返金できる。"
MINUTES_A = "4月10日の定例で、新料金プランの公開日を5月1日に決定した。担当は佐藤。"
MINUTES_B = "4月10日の定例では新料金プランの公開日を検討したが、結論は次回に持ち越した。"
SUPPORT_3 = {
    "支持": "根拠だけから主張が成り立つ。",
    "矛盾": "根拠が主張と食い違う。",
    "情報不足": "根拠からは成否を決められない。",
}


def span(text: str, part: str, path: str) -> dict:
    start = text.index(part)
    return {"path": path, "start": start, "end": start + len(part)}


def tags(domain: str, difficulty: str = "clear", phenomena: list[str] | None = None) -> dict:
    return {"lang": "ja", "domain": domain, "difficulty": difficulty, "phenomena": phenomena or []}


rows = [
    {
        "id": "ex-0001",
        "family_id": "ex-refund",
        "state": {"根拠": REFUND_RULE, "主張": "購入から10日後でも、未使用なら返金できる。"},
        "type": "noul",
        "question": "`主張` は `根拠` から支持されるか",
        "target_hard": "false",
        "tags": tags("grounding", "hard", ["数値"]),
        "evidence_spans": [span(REFUND_RULE, "購入から7日以内", "根拠")],
    },
    {
        "id": "ex-0002",
        "family_id": "ex-refund",
        "state": {"根拠": REFUND_RULE, "主張": "購入から5日後で、未使用なら返金できる。"},
        "type": "noul",
        "question": "`主張` は `根拠` から支持されるか",
        "target_hard": "true",
        "tags": tags("grounding", "hard", ["数値"]),
        "provenance": PROV | {"contrast_of": "ex-0001"},
        "evidence_spans": [span(REFUND_RULE, "購入から7日以内", "根拠")],
    },
    {
        "id": "ex-0003",
        "family_id": "ex-refund",
        "state": {
            "規約": REFUND_RULE,
            "事例": "購入5日後に返金の申し出があった。使用したかは不明。",
        },
        "type": "choice",
        "question": "`事例` は `規約` の返金条件を満たすか",
        "criteria": {
            "成立": "必要な条件が全て確認できる。",
            "不成立": "満たさない条件が確認できる。",
            "情報不足": "確認できない条件が残っている。",
        },
        "target_hard": "情報不足",
        "target_probs": [0.03, 0.02, 0.95],
        "tags": tags("rag", "hard", ["情報不足", "条件"]),
    },
    {
        "id": "ex-0004",
        "family_id": "ex-minutes",
        "state": {"文書A": MINUTES_A, "文書B": MINUTES_B},
        "type": "choice",
        "question": "新料金プランの公開日について、`文書A` と `文書B` の記述はどういう関係か",
        "criteria": {
            "一致": "同じ内容を述べている。",
            "矛盾": "両立しない内容を述べている。",
            "片方のみ": "片方にしか記述がなく、食い違いはない。",
            "無関係": "どちらもこの点に触れていない。",
        },
        "target_hard": "矛盾",
        "tags": tags("compare", "hard", ["決定と検討"]),
    },
    {
        "id": "ex-0005",
        "family_id": "ex-minutes",
        "state": {"原文": MINUTES_B, "主張": "新料金プランは5月1日に公開される。"},
        "type": "choice",
        "question": "`主張` は `原文` から支持されるか",
        "criteria": SUPPORT_3,
        "target_hard": "情報不足",
        "tags": tags("grounding", "hard", ["決定と検討", "情報不足"]),
    },
    {
        "id": "ex-0006",
        "family_id": "ex-leave",
        "state": {
            "質問": "育児休業は子どもが何歳になるまで取れますか。延長はできますか。",
            "検索結果": "育児休業は、原則として子が1歳に達するまで取得できる。",
        },
        "type": "score",
        "question": "`検索結果` は `質問` に答えるのに十分か",
        "criteria": [
            "質問に答える情報がない。",
            "一部には答えられるが、答えられない部分が残る。",
            "質問の全てに答えられる。",
        ],
        "target_hard": "1",
        "tags": tags("rag", "clear", ["部分回答"]),
    },
    {
        "id": "ex-0007",
        "family_id": "ex-minutes",
        "state": {
            "原文": MINUTES_A,
            "要約": "新料金プランは佐藤の担当で、5月1日の公開が決まった。",
        },
        "type": "score",
        "question": "`要約` は `原文` に忠実か",
        "criteria": [
            "原文と矛盾する内容が中心になっている。",
            "重要な点で原文と食い違う。",
            "大筋は合うが、原文にない内容が混じる。",
            "原文にない内容はないが、重要な点が抜けている。",
            "原文にない内容がなく、重要な点も抜けていない。",
        ],
        "target_hard": "3",
        "target_probs": [0.0, 0.0, 0.05, 0.6, 0.35],
        "tags": tags("writing", "ambiguous", ["省略"]),
    },
]

out = Path(__file__).parent.parent / "examples" / "sample_records.jsonl"
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        row = {"split": "dev", "label_source": "teacher", "provenance": PROV} | row
        rec = Record.model_validate(row)
        f.write(rec.model_dump_json(exclude_defaults=False) + "\n")
print(f"{len(rows)} 件を {out} に書いた")
