"""rubric の読み込みと検査。rubric は質問文と criteria の定義で、人手セットと合成データの両方が使う。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from uratori.data.schema import LabelSource, Provenance, Record, Split, Tags
from uratori.types import Json, QuestionType

EXAMPLE_KINDS = ("positive", "negative", "boundary")


class RubricExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["positive", "negative", "boundary"]
    state: Json
    label: str
    note: str | None = None


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+(-[a-z0-9]+)+$")
    domain: Literal["grounding", "compare", "rag", "writing", "routing"]
    type: QuestionType
    question: str = Field(min_length=1)
    criteria: dict[str, str | None] | list[str] | None = None
    notes: str | None = None
    # true の rubric は学習データに使わず、未知の rubric への汎化を測るために取っておく
    held_out: bool = False
    examples: list[RubricExample]

    def example_record(self, index: int) -> Record:
        ex = self.examples[index]
        return Record(
            id=f"{self.id}-ex{index}",
            family_id=f"rubric-{self.id}",
            split=Split.DEV,
            state=ex.state,
            type=self.type,
            question=self.question,
            criteria=self.criteria,
            target_hard=ex.label,
            label_source=LabelSource.TEACHER,
            provenance=Provenance(source="rubric-example"),
            tags=Tags(lang="ja", domain=self.domain, rubric_id=self.id),
        )

    @model_validator(mode="after")
    def _check(self) -> Rubric:
        missing = set(EXAMPLE_KINDS) - {ex.kind for ex in self.examples}
        if missing:
            raise ValueError(f"{self.id}: 例が足りない（{sorted(missing)}）")
        for i in range(len(self.examples)):
            self.example_record(i)  # ラベルと criteria の整合を schema で検査する
        return self


def load_rubrics(directory: Path) -> dict[str, Rubric]:
    rubrics: dict[str, Rubric] = {}
    for path in sorted(directory.glob("*.yaml")):
        for row in yaml.safe_load(path.read_text(encoding="utf-8")):
            rubric = Rubric.model_validate(row)
            if rubric.id in rubrics:
                raise ValueError(f"rubric の id {rubric.id!r} が重複している（{path.name}）")
            rubrics[rubric.id] = rubric
    return rubrics
