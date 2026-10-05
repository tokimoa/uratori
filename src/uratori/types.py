"""質問の型。形は TypeSafe の /v1/systemone の questions と同じにしてある。

criteria の形は型ごとに違う。
  noul    {"true": "...", "false": "..."}（省略可）
  choice  {"候補キー": "説明", ...}
  score   ["段階 0 の説明", "段階 1 の説明", ...]（低い順）
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 学習と評価で扱う上限。wire 上は choice 255、score 10 まで許されているが、
# 学習していない候補数を黙って受けないよう、ここで絞る。
MAX_CHOICE_OPTIONS = 8
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

NOUL_LABELS: tuple[str, str] = ("false", "true")

Json = str | int | float | bool | None | dict[str, Any] | list[Any]


class QuestionType(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1)


class NoulQuestion(_QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], str] | None = None

    @property
    def labels(self) -> list[str]:
        return list(NOUL_LABELS)

    @property
    def descriptions(self) -> list[str]:
        c = self.criteria or {}
        return [c.get("false", ""), c.get("true", "")]


class ChoiceQuestion(_QuestionBase):
    type: Literal["choice"] = "choice"
    criteria: dict[str, str | None]

    @model_validator(mode="after")
    def _check(self) -> ChoiceQuestion:
        n = len(self.criteria)
        if not 2 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice の候補は 2〜{MAX_CHOICE_OPTIONS} 個（{n} 個あった）")
        if any(not k.strip() for k in self.criteria):
            raise ValueError("choice の候補キーが空")
        return self

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)

    @property
    def descriptions(self) -> list[str]:
        return [v or "" for v in self.criteria.values()]


class ScoreQuestion(_QuestionBase):
    type: Literal["score"] = "score"
    criteria: list[str]

    @model_validator(mode="after")
    def _check(self) -> ScoreQuestion:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score の段階は {MIN_SCORE_LEVELS}〜{MAX_SCORE_LEVELS} 個（{n} 個あった）"
            )
        if any(not c.strip() for c in self.criteria):
            raise ValueError("score の段階の説明が空")
        return self

    @property
    def labels(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]

    @property
    def descriptions(self) -> list[str]:
        return list(self.criteria)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]
