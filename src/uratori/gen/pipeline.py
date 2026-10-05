"""合成データの生成から検証まで。

  原文を用意する → 教師が問題を書く → コードで検査する → 別の判定者がラベルを隠して判定する
  → 設計したラベルと一致した件だけ採用する

採用しなかった件は理由つきで残す。採用率と棄却理由が、次に直す場所を教えてくれる。
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from uratori.data.rubrics import Rubric
from uratori.data.schema import LabelSource, Provenance, Record, Split, Tags, Verification
from uratori.gen.plan import (
    COPY_RATIO_LIMIT,
    PHENOMENA,
    POSITIVE_LABELS,
    QUESTION_KINDS,
    SENTENCE_SOURCE,
    SOURCE_FIELD,
    SOURCE_KEY_NAMES,
    DocSeed,
    is_dev_doc,
    rubrics_for,
    sample_phenomena,
    sentence_excerpt,
)
from uratori.gen.prompts import (
    PROMPT_VERSION,
    Draft,
    GenParseError,
    doc_messages,
    item_messages,
    open_messages,
    parse_doc,
    parse_items,
    parse_open,
    prompt_hash,
    written_fields,
)
from uratori.llm.client import ChatClient
from uratori.llm.judge import JudgeParseError, judge
from uratori.serialize import render_state

SHINGLE = 20
INJECT_RATE = 0.08
TARGETABLE = sorted(PHENOMENA - {"指示の混入"})


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    text: str
    genre: str | None  # 外部データは None
    source: str  # provenance.source に入れる出典
    license: str


@dataclass
class Candidate:
    record: Record
    reason: str
    status: str = "pending"  # accepted / rejected
    reject_reason: str | None = None
    verifier_label: str | None = None


@dataclass
class Stats:
    counts: Counter = field(default_factory=Counter)

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] += n


def make_docs(
    client: ChatClient, seeds: list[DocSeed], stats: Stats, workers: int
) -> list[SourceDoc]:
    def run(seed: DocSeed) -> SourceDoc | None:
        try:
            result = client.chat(doc_messages(seed), json_mode=True)
            title, body = parse_doc(result.text)
        except GenParseError:
            stats.add("doc_parse_error")
            return None
        return SourceDoc(
            seed.doc_id, f"{title}\n{body}", seed.genre, f"generated:{seed.doc_id}", "generated"
        )

    with ThreadPoolExecutor(workers) as pool:
        return [doc for doc in pool.map(run, seeds) if doc is not None]


def _labels_for(rubric: Rubric, rng: random.Random) -> list[str]:
    """1 回の呼び出しで作らせるラベル。全ラベルを 1 件ずつ、順番は混ぜる。段階の多い rubric は 4 つまで。"""
    labels = rubric.example_record(0).labels[:]
    rng.shuffle(labels)
    return labels[:4]


def make_candidates(
    client: ChatClient,
    docs: list[SourceDoc],
    rubrics: dict[str, Rubric],
    rubrics_per_doc: int,
    seed: int,
    stats: Stats,
    workers: int,
    id_prefix: str,
) -> list[Candidate]:
    rng = random.Random(seed)
    rubric_ids = sorted(rid for rid, r in rubrics.items() if not r.held_out and rid in SOURCE_FIELD)
    jobs = []
    for doc in docs:
        for rid in rubrics_for(doc.genre, rubric_ids, rubrics_per_doc, rng):
            source = doc.text
            if rid in SENTENCE_SOURCE:
                source = sentence_excerpt(doc.text, SENTENCE_SOURCE[rid], rng)
                if source is None:
                    stats.add("no_excerpt")
                    continue
            jobs.append(
                (
                    doc,
                    rubrics[rid],
                    source,
                    _labels_for(rubrics[rid], rng),
                    sample_phenomena(rng),
                    rng.random() < INJECT_RATE,
                )
            )

    def run(job) -> list[Candidate]:
        doc, rubric, source, labels, phenomena, inject = job
        source_field = SOURCE_FIELD[rubric.id]
        fields = written_fields(rubric, source_field)
        messages = item_messages(
            rubric, source, source_field, labels, target_phenomena=phenomena, inject=inject
        )
        try:
            result = client.chat(messages, json_mode=True)
            drafts = parse_items(result.text, fields, rubric.example_record(0).labels)
        except GenParseError:
            stats.add("item_parse_error")
            return []
        stats.add("drafts", sum(d is not None for d in drafts))
        stats.add("malformed_items", sum(d is None for d in drafts))
        base = f"{id_prefix}-{doc.doc_id}-{rubric.id}"
        out = []
        for k, draft in enumerate(drafts):
            if draft is None:
                continue
            partner = _partner(drafts, k)
            record = _to_record(
                doc, rubric, source, source_field, draft, f"{base}-{k}", messages,
                contrast_of=f"{base}-{partner}" if partner is not None else None,
            )  # fmt: skip
            out.append(Candidate(record=record, reason=draft.reason))
        return out

    with ThreadPoolExecutor(workers) as pool:
        return [c for group in pool.map(run, jobs) for c in group]


def make_open_candidates(
    client: ChatClient,
    docs: list[SourceDoc],
    per_doc: int,
    seed: int,
    stats: Stats,
    workers: int,
    id_prefix: str,
) -> list[Candidate]:
    """rubric にない質問を教師に考えさせて問題を作る。見たことのない質問への汎化のため。"""
    rng = random.Random(seed + 7919)
    jobs = [
        (doc, n, rng.choice(SOURCE_KEY_NAMES), rng.choice(QUESTION_KINDS), sample_phenomena(rng))
        for doc in docs
        for n in range(per_doc)
    ]

    def run(job) -> list[Candidate]:
        doc, n, source_key, kind, phenomena = job
        messages = open_messages(doc.text, source_key, kind, phenomena)
        try:
            result = client.chat(messages, json_mode=True)
            spec, drafts = parse_open(result.text, source_key)
        except GenParseError:
            stats.add("open_parse_error")
            return []
        stats.add("open_questions")
        stats.add("drafts", sum(d is not None for d in drafts))
        stats.add("malformed_items", sum(d is None for d in drafts))
        base = f"{id_prefix}-{doc.doc_id}-open{n}"
        out = []
        for k, draft in enumerate(drafts):
            if draft is None:
                continue
            partner = _partner(drafts, k)
            record = Record(
                id=f"{base}-{k}",
                family_id=f"syn-{doc.doc_id}",
                split=Split.DEV if is_dev_doc(doc.doc_id) else Split.TRAIN,
                state={source_key: doc.text} | draft.fields,
                type=spec.type,
                question=spec.question,
                criteria=spec.criteria,
                target_hard=draft.label,
                label_source=LabelSource.TEACHER,
                verification=Verification(status="unverified", note=draft.reason),
                provenance=Provenance(
                    source=doc.source,
                    source_license=doc.license,
                    generator=f"{PROMPT_VERSION}-open",
                    prompt_hash=prompt_hash(messages[0]["content"]),
                    contrast_of=f"{base}-{partner}" if partner is not None else None,
                ),
                tags=Tags(
                    lang="ja",
                    domain=spec.domain,
                    difficulty=draft.difficulty,
                    phenomena=draft.phenomena,
                    rubric_id=None,
                ),
            )
            out.append(Candidate(record=record, reason=draft.reason))
        return out

    with ThreadPoolExecutor(workers) as pool:
        return [c for group in pool.map(run, jobs) for c in group]


def _partner(drafts: list[Draft | None], k: int) -> int | None:
    """最小対の相手の位置。相手がいて、ラベルが違う場合だけ対として扱う。"""
    pair = drafts[k].pair
    if pair is None or pair == k or not 0 <= pair < len(drafts) or drafts[pair] is None:
        return None
    return pair if drafts[pair].label != drafts[k].label else None


def _to_record(
    doc: SourceDoc,
    rubric: Rubric,
    source: str,
    source_field: str | None,
    draft: Draft,
    rid: str,
    messages: list[dict],
    contrast_of: str | None = None,
) -> Record:
    state = {}
    for key in rubric.examples[0].state:  # rubric の例と同じ並びにする
        state[key] = source if key == source_field else draft.fields[key]
    return Record(
        id=rid,
        family_id=f"syn-{doc.doc_id}",
        split=Split.DEV if is_dev_doc(doc.doc_id) else Split.TRAIN,
        state=state,
        type=rubric.type,
        question=rubric.question,
        criteria=rubric.criteria,
        target_hard=draft.label,
        label_source=LabelSource.TEACHER,
        verification=Verification(status="unverified", note=draft.reason),
        provenance=Provenance(
            source=doc.source,
            source_license=doc.license,
            generator=PROMPT_VERSION,
            teacher_model=None,
            prompt_hash=prompt_hash(messages[0]["content"]),
            contrast_of=contrast_of,
        ),
        tags=Tags(
            lang="ja",
            domain=rubric.domain,
            difficulty=draft.difficulty,
            phenomena=draft.phenomena,
            rubric_id=rubric.id,
        ),
    )


# ---- コードによる検査 ----------------------------------------------------------------


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}


def copy_ratio(source: str, written: str) -> float:
    """書いた文章のうち、原文と連続して一致する最長の部分が占める割合。"""
    if len(written) < 20:
        return 0.0
    match = SequenceMatcher(None, source, written, autojunk=False).find_longest_match(
        0, len(source), 0, len(written)
    )
    return match.size / len(written)


def code_checks(candidates: list[Candidate], protected: Iterable[Record], stats: Stats) -> None:
    """LLM を使わずに落とせるものを落とす。

    protected は人手評価セットなど、学習データと原文が重なってはいけないもの。
    """
    protected_shingles: set[str] = set()
    for record in protected:
        protected_shingles |= shingles(render_state(record.state))
    seen: set[str] = set()
    for c in candidates:
        r = c.record
        rubric_id = r.tags.rubric_id or ""
        source_field = SOURCE_FIELD.get(rubric_id)
        written = [v for k, v in r.state.items() if k != source_field]
        reason = None
        if r.content_hash() in seen:
            reason = "重複"
        elif any(len(v) < 4 for v in written):
            reason = "書いた文章が短すぎる"
        elif len(shingles(render_state(r.state)) & protected_shingles) >= 3:
            reason = "人手評価セットと文面が重なる"
        elif (
            source_field
            and POSITIVE_LABELS.get(rubric_id) == r.target_hard
            and max(copy_ratio(r.state[source_field], v) for v in written) >= COPY_RATIO_LIMIT
        ):
            reason = "原文の写し"
        seen.add(r.content_hash())
        if reason:
            c.status, c.reject_reason = "rejected", f"code: {reason}"
            stats.add(f"reject_code_{reason}")


# ---- 判定者による検証 ----------------------------------------------------------------


def verify(
    candidates: list[Candidate],
    judge_fn: Callable[[Record], tuple[str, list[float]]],
    verifier_name: str,
    generator_name: str,
    stats: Stats,
    workers: int,
) -> None:
    """ラベルを隠して判定させ、設計したラベルと一致した件だけ採用する。"""

    def run(c: Candidate) -> None:
        if c.status == "rejected":
            return
        try:
            label, probs = judge_fn(c.record)
        except JudgeParseError:
            c.status, c.reject_reason = "rejected", "verify: 判定を読めなかった"
            stats.add("reject_verify_parse")
            return
        c.verifier_label = label
        agreed = label == c.record.target_hard
        c.record = c.record.model_copy(
            update={
                "teacher_reported_probs": probs,
                "verification": Verification(
                    status="verified" if agreed else "disputed",
                    by=[generator_name, verifier_name] if agreed else [generator_name],
                    note=c.reason,
                ),
                "provenance": c.record.provenance.model_copy(
                    update={"teacher_model": generator_name}
                ),
            }
        )
        if agreed:
            c.status = "accepted"
            stats.add("accepted")
        else:
            c.status, c.reject_reason = "rejected", f"verify: 判定者は {label}"
            stats.add("reject_verify_mismatch")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, candidates))


def llm_judge_fn(client: ChatClient) -> Callable[[Record], tuple[str, list[float]]]:
    def run(record: Record) -> tuple[str, list[float]]:
        j = judge(client, record.state, record.to_question())
        return j.label, j.probs

    return run
