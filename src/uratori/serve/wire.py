"""/v1/systemone のリクエストとレスポンスの型（https://docs.typesafe.ai/api.md に合わせた）。

ここは型と、確率分布から回答を組み立てる部分だけ。HTTP サーバ本体は app.py にある。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from uratori.confidence import choice_confidence, score_confidence, score_value
from uratori.types import ChoiceQuestion, Json, NoulQuestion, Question, ScoreQuestion


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Json
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def answer_from_probs(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion, probs: Sequence[float]
) -> Answer:
    """probs は question.labels の順。"""
    labels = question.labels
    if len(probs) != len(labels):
        raise ValueError(f"確率の数 {len(probs)} が候補数 {len(labels)} と合わない")
    probs = [float(p) for p in probs]
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=probs[labels.index("true")])
    table = dict(zip(labels, probs, strict=True))
    if isinstance(question, ChoiceQuestion):
        top = max(range(len(probs)), key=probs.__getitem__)
        return ChoiceAnswer(
            choice=labels[top], probabilities=table, confidence=choice_confidence(probs)
        )
    return ScoreAnswer(
        score=score_value(probs),
        legend=dict(zip(labels, question.criteria, strict=True)),
        probabilities=table,
        confidence=score_confidence(probs),
    )
