"""原文と問題を 1 回でまとめて作らせる生成経路（family 方式）。

最初の経路（pipeline.py）は、先に汎用の原文を用意し、後から問題を書かせる。その作り方だと
原文が問題のために設計されておらず、細部を読ませる問題になりにくい。こちらは教師に
「この問題を成り立たせるための原文」から書かせる。例外、但し書き、離れた 2 箇所の数値、
紛らわしい別の記述などを原文に仕込んだうえで、複数の rubric の問題と最小対を作らせる。

原文も教師が書くので、根拠側の文章が一字一句同じであることはコードで保証できない。
代わりに、原文を documents に 1 回だけ書かせ、問題からは名前で参照させる。
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

from uratori.data.rubrics import Rubric
from uratori.data.schema import LabelSource, Provenance, Record, Split, Tags, Verification
from uratori.gen.pipeline import Candidate, Stats
from uratori.gen.plan import (
    GENRES,
    INDUSTRIES,
    LENGTHS,
    PHENOMENA,
    SUBJECTS,
    is_dev_doc,
    sample_phenomena,
)
from uratori.gen.prompts import GenParseError, _load_json, prompt_hash
from uratori.llm.client import ChatClient
from uratori.llm.judge import RULES

FAMILY_VERSION = "family-v1"

FAMILY_SYSTEM = f"""あなたは日本語の判定モデルを鍛えるための、ひっかけのある問題を作る出題者です。架空の組織の業務文書を自分で書き、その文書について複数の問題を作ります。

判定の決まり（問題はこの決まりで正解が 1 つに決まるように作る）
{RULES}

原文の書き方
- 指定された種類の文書を、架空の組織名、人名、製品名で書く。実在の企業や人物は出さない。
- 問題を成り立たせるための仕掛けを原文に入れる。例外や但し書き、離れた 2 箇所を合わせないと分からない数値、相対的な日付（「翌営業日」「先月」）、主語を省いた文、似た語を使った別の話題の文、決定した事項と検討中の事項の混在、など。
- 原文は documents に名前（D1、D2 など）を付けて 1 回だけ書く。問題の state では、原文をそのまま使うキーの値を "@D1" のように名前で書く。同じ原文を何度も書かない。
- 最小対のために原文を 1 箇所だけ変えた版が要るときは、別の名前（D1b など）で全文を書く。

問題の作り方
- 指定された rubric ごとに、質問文とラベルの定義どおりに問題を作る。state のキーは rubric ごとに指定されたものを全て使う。
- 単語の重なりでは解けないようにする。正解が「支持」でも原文の言い回しを写さず言い換える。正解が「矛盾」「情報不足」の件にも原文と同じ語を使う。
- 半分以上を hard にする。細部を読み違えると逆の答えになる問題にする。
- 「情報不足」に当たるラベルは、原文が触れていないもっともらしい内容で作る。
- 最小対を 2 組以上入れる。対の件は、元の件の文章または原文を 1 箇所だけ変えたもので、正解のラベルが変わる。pair に元の件の番号（items の中の 0 から数えた位置）を入れる。元の件の pair は null。
- 外部の知識が要る問題にしない。ラベルが 1 つに決まらない問題にしない。

出力は JSON オブジェクト 1 つだけ。
{{"documents": {{"D1": "原文の全文"}}, "items": [{{"rubric": "rubric の id", "state": {{"キー": "@D1 または書いた文章"}}, "label": "ラベル", "reason": "そのラベルになる理由を 1 文で", "difficulty": "clear か hard", "phenomena": ["当てはまる現象"], "pair": null}}]}}"""


TRAP_NOTE = """ひっかけの作り方（この回は特にこれを優先する）
- 対象の現象を必ず使う。数値なら、原文の 2 箇所の数値を足す、引く、割合を出す、単位を換える（円と万円、分と時間、か月と日）ことで初めて決まる主張にする。相対時点なら、基準となる日付を原文の別の箇所に置き、「翌営業日」「先月末」「3 営業日以内」の読み替えで決まるようにする。決定と検討なら、同じ事項について「決定」「検討する方向」「持ち越し」の記述を原文の離れた場所に置く。範囲の含意なら、「平日」「〜のみ」「原則として」の有無で逆になる対を作る。
- 対の件の半分は、主張ではなく原文のほうを 1 箇所だけ変えた版（D1b など）で作る。
- 正解が「情報不足」の件には、原文と同じ語を多く使い、単語の重なりでは支持と区別できないようにする。
- 原文には、問題と同じ語を使った無関係な文（別の条件、別の対象についての記述）を必ず 1 つ以上入れる。
"""


def family_messages(
    genre: str,
    industry: str,
    subject: str,
    length: str,
    rubrics: list[Rubric],
    items_per_rubric: int,
    phenomena: list[str],
    trap: bool = False,
) -> list[dict]:
    lo, hi = LENGTHS[length]
    lines = [
        f"文書の種類: {genre}",
        f"組織の業種: {industry}",
        f"題材: {subject}",
        f"原文の長さ: {lo} 字から {hi} 字",
        f"できるだけ使う現象: {'、'.join(phenomena)}",
        "",
        f"次の rubric それぞれについて {items_per_rubric} 件ずつ作り、最後に対の件を足す。",
    ]
    for rubric in rubrics:
        q = rubric.example_record(0).to_question()
        lines += [
            "",
            f"rubric: {rubric.id}",
            f"質問: {rubric.question}",
            f"state のキー: {'、'.join(rubric.examples[0].state)}",
            "ラベル:",
        ]
        lines += [
            f'- "{label}": {desc}' for label, desc in zip(q.labels, q.descriptions, strict=True)
        ]
        if rubric.notes:
            lines.append(f"補足: {rubric.notes}")
    lines += ["", f"phenomena に使える語: {'、'.join(sorted(PHENOMENA))}"]
    if trap:
        lines = [TRAP_NOTE] + lines
    return [
        {"role": "system", "content": FAMILY_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_family(text: str, rubrics: dict[str, Rubric]) -> list[dict | None]:
    """items を順番どおりに返す。形の合わない件は None。state の "@名前" は原文に置き換える。"""
    data = _load_json(text)
    documents = data.get("documents")
    items = data.get("items")
    if not isinstance(documents, dict) or not isinstance(items, list):
        raise GenParseError("documents か items がない")
    documents = {str(k): v for k, v in documents.items() if isinstance(v, str) and len(v) >= 20}
    out: list[dict | None] = []
    for item in items:
        out.append(_parse_family_item(item, documents, rubrics))
    return out


def _parse_family_item(
    item: object, documents: dict[str, str], rubrics: dict[str, Rubric]
) -> dict | None:
    if not isinstance(item, dict) or item.get("rubric") not in rubrics:
        return None
    rubric = rubrics[item["rubric"]]
    state = item.get("state")
    keys = list(rubric.examples[0].state)
    if not isinstance(state, dict) or set(state) != set(keys):
        return None
    resolved = {}
    for key in keys:
        value = state[key]
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        if value.startswith("@"):
            if value[1:] not in documents:
                return None
            value = documents[value[1:]]
        resolved[key] = value
    if item.get("label") not in rubric.example_record(0).labels:
        return None
    pair = item.get("pair")
    return {
        "rubric": rubric,
        "state": resolved,
        "label": item["label"],
        "reason": str(item.get("reason", "")),
        "difficulty": item.get("difficulty")
        if item.get("difficulty") in ("clear", "hard")
        else "clear",
        "phenomena": [p for p in item.get("phenomena") or [] if p in PHENOMENA],
        "pair": pair if isinstance(pair, int) and not isinstance(pair, bool) else None,
    }


def make_family_candidates(
    client: ChatClient,
    rubrics: dict[str, Rubric],
    n_families: int,
    seed: int,
    stats: Stats,
    workers: int,
    id_prefix: str,
    rubrics_per_family: int = 3,
    items_per_rubric: int = 2,
    trap: bool = False,
) -> list[Candidate]:
    rng = random.Random(seed)
    usable = {rid: r for rid, r in rubrics.items() if not r.held_out}
    lengths, weights = ["short", "medium", "long"], [0.35, 0.45, 0.2]
    jobs = []
    for i in range(n_families):
        jobs.append(
            (
                f"{id_prefix}-f{seed}-{i:05d}",
                GENRES[i % len(GENRES)],
                rng.choice(INDUSTRIES),
                rng.choice(SUBJECTS),
                rng.choices(lengths, weights)[0],
                [usable[rid] for rid in rng.sample(sorted(usable), rubrics_per_family)],
                sample_phenomena(rng, 3),
            )
        )

    def run(job) -> list[Candidate]:
        fam_id, genre, industry, subject, length, picked, phenomena = job
        messages = family_messages(
            genre, industry, subject, length, picked, items_per_rubric, phenomena, trap=trap
        )
        try:
            result = client.chat(messages, json_mode=True)
            items = parse_family(result.text, {r.id: r for r in picked})
        except GenParseError:
            stats.add("family_parse_error")
            return []
        stats.add("families")
        stats.add("drafts", sum(x is not None for x in items))
        stats.add("malformed_items", sum(x is None for x in items))
        out = []
        for k, item in enumerate(items):
            if item is None:
                continue
            partner = item["pair"]
            valid_partner = (
                partner is not None
                and partner != k
                and 0 <= partner < len(items)
                and items[partner] is not None
                and items[partner]["rubric"].id == item["rubric"].id
                and items[partner]["label"] != item["label"]
            )
            rubric = item["rubric"]
            record = Record(
                id=f"{fam_id}-{k}",
                family_id=f"syn-{fam_id}",
                split=Split.DEV if is_dev_doc(fam_id) else Split.TRAIN,
                state=item["state"],
                type=rubric.type,
                question=rubric.question,
                criteria=rubric.criteria,
                target_hard=item["label"],
                label_source=LabelSource.TEACHER,
                verification=Verification(status="unverified", note=item["reason"]),
                provenance=Provenance(
                    source=f"generated:{fam_id}",
                    source_license="generated",
                    generator=f"{FAMILY_VERSION}-trap" if trap else FAMILY_VERSION,
                    prompt_hash=prompt_hash(messages[0]["content"]),
                    contrast_of=f"{fam_id}-{partner}" if valid_partner else None,
                ),
                tags=Tags(
                    lang="ja",
                    domain=rubric.domain,
                    difficulty=item["difficulty"],
                    phenomena=item["phenomena"],
                    rubric_id=rubric.id,
                ),
            )
            out.append(Candidate(record=record, reason=item["reason"]))
        return out

    with ThreadPoolExecutor(workers) as pool:
        return [c for group in pool.map(run, jobs) for c in group]


def dump_candidates(candidates: list[Candidate]) -> str:
    return "".join(
        json.dumps(
            {
                "status": c.status,
                "reject_reason": c.reject_reason,
                "record": json.loads(c.record.model_dump_json()),
            },
            ensure_ascii=False,
        )
        + "\n"
        for c in candidates
    )
