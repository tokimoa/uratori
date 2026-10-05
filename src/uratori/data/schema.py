"""学習と評価に使うデータ 1 件の schema。1 件は「state と 1 つの質問」。

同じ原文から作った別質問、別候補、反例は同じ family_id にまとめ、分割は family 単位で行う。
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from uratori.serialize import render_state
from uratori.types import Json, Question, QuestionType

PROB_SUM_TOLERANCE = 1e-3

_question_adapter: TypeAdapter[Question] = TypeAdapter(Question)


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"
    CHALLENGE = "challenge"
    POOL = "pool"  # 未ラベル。能動学習の候補


class LabelSource(StrEnum):
    HUMAN = "human"
    PROGRAM = "program"  # コードで確定した正解（最小対の置換など）
    DATASET = "dataset"  # 既存データセットのラベルを変換
    TEACHER = "teacher"
    NONE = "none"


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "unverified", "disputed"] = "unverified"
    by: list[str] = Field(default_factory=list)  # 確認した人、教師、プログラムの名前
    note: str | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # 原文の出典。データセット名と ID、URL、"authored" など
    source_license: str | None = None
    generator: str | None = None  # 生成経路の名前と版
    teacher_model: str | None = None
    teacher_revision: str | None = None
    prompt_hash: str | None = None
    seed: int | None = None
    contrast_of: str | None = None  # 最小対の相手の id


class Tags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lang: Literal["ja", "en", "mixed"]
    domain: Literal["grounding", "compare", "rag", "writing", "routing"]
    difficulty: Literal["clear", "hard", "ambiguous"] = "clear"
    phenomena: list[str] = Field(default_factory=list)
    rubric_id: str | None = None


class EvidenceSpan(BaseModel):
    """根拠の位置。state が dict なら path に最上位キーを入れ、その値の文字位置で示す。"""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    start: int = Field(ge=0)
    end: int = Field(gt=0)


def _check_probs(name: str, probs: list[float], n: int) -> None:
    if len(probs) != n:
        raise ValueError(f"{name} の長さ {len(probs)} が候補数 {n} と合わない")
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
        raise ValueError(f"{name} に 0〜1 の範囲外の値がある")
    if abs(sum(probs) - 1.0) > PROB_SUM_TOLERANCE:
        raise ValueError(f"{name} の総和が {sum(probs):.4f} で 1 にならない")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    split: Split

    state: Json
    type: QuestionType
    question: str = Field(min_length=1)
    criteria: dict[str, str | None] | list[str] | None = None

    # ラベルは labels の順（noul は false, true。choice は criteria の順。score は低い順）
    target_hard: str | None = None
    target_probs: list[float] | None = None
    label_source: LabelSource
    verification: Verification = Field(default_factory=Verification)

    # 教師が申告した確率と、複数教師の投票割合は別物として残す
    teacher_reported_probs: list[float] | None = None
    teacher_vote_distribution: list[float] | None = None
    # 人のラベルが割れた例は、裁定前のラベルも残す
    human_votes: list[str] | None = None

    provenance: Provenance
    tags: Tags
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)

    def to_question(self) -> Question:
        payload: dict[str, object] = {"type": self.type.value, "instructions": self.question}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return _question_adapter.validate_python(payload)

    @property
    def labels(self) -> list[str]:
        return self.to_question().labels

    @property
    def target_index(self) -> int | None:
        return None if self.target_hard is None else self.labels.index(self.target_hard)

    def content_hash(self) -> str:
        """state、質問、criteria が同じなら同じ値。完全重複の検出に使う。"""
        blob = json.dumps(
            [self.state, self.type.value, self.question, self.criteria],
            ensure_ascii=False,
            sort_keys=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _span_text(self, span: EvidenceSpan) -> str:
        if span.path is None:
            if isinstance(self.state, dict):
                raise ValueError("state が dict のときは evidence_spans に path が必要")
            return render_state(self.state)
        if not isinstance(self.state, dict) or span.path not in self.state:
            raise ValueError(f"evidence_spans の path {span.path!r} が state にない")
        value = self.state[span.path]
        if not isinstance(value, str):
            raise ValueError(f"evidence_spans の path {span.path!r} の値が文字列でない")
        return value

    @model_validator(mode="after")
    def _check(self) -> Record:
        labels = self.labels  # criteria の形の検査も兼ねる
        n = len(labels)

        if self.target_hard is not None and self.target_hard not in labels:
            raise ValueError(f"target_hard {self.target_hard!r} が候補 {labels} にない")
        for name in ("target_probs", "teacher_reported_probs", "teacher_vote_distribution"):
            probs = getattr(self, name)
            if probs is not None:
                _check_probs(name, probs, n)
        if self.human_votes is not None:
            unknown = [v for v in self.human_votes if v not in labels]
            if unknown:
                raise ValueError(f"human_votes に候補にないラベルがある: {unknown}")

        labeled = self.target_hard is not None or self.target_probs is not None
        if self.split is Split.POOL:
            if self.label_source is not LabelSource.NONE and not labeled:
                raise ValueError("label_source があるのにラベルがない")
        elif not labeled:
            raise ValueError("pool 以外の分割にはラベルが必要")
        if self.label_source is LabelSource.NONE and labeled:
            raise ValueError("ラベルがあるのに label_source が none")
        if (
            self.label_source in (LabelSource.HUMAN, LabelSource.PROGRAM, LabelSource.DATASET)
            and self.target_hard is None
        ):
            raise ValueError(f"label_source={self.label_source.value} には target_hard が必要")

        for span in self.evidence_spans:
            text = self._span_text(span)
            if span.start >= span.end or span.end > len(text):
                raise ValueError(
                    f"evidence_spans [{span.start}, {span.end}) が範囲外（長さ {len(text)}）"
                )
        return self

    def soft_hard_mismatch(self) -> bool:
        """target_probs の最頻ラベルが target_hard と食い違うか。エラーではなく警告にする。"""
        if self.target_hard is None or self.target_probs is None:
            return False
        top = max(range(len(self.target_probs)), key=self.target_probs.__getitem__)
        return self.labels[top] != self.target_hard
