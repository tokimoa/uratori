"""state と質問をモデル入力の文字列に直す。

並びは 状態 → 質問 → 候補。候補ごとに説明の後ろへマーカーを 1 つ置き、モデルは
マーカー位置の hidden state を採点する。状態を先に、マーカーを各候補の末尾に置くのは、
因果 attention のデコーダでもマーカーが状態と候補の両方を見られるようにするため。
エンコーダとデコーダで同じ形式を使い、backbone の比較条件を揃える。

長さの超過はここでは判定しない（トークナイザに依存する）。モデル側で
InputTooLongError を投げ、黙って切らない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from uratori.types import ChoiceQuestion, Json, NoulQuestion, ScoreQuestion

# トークナイザ側で特殊トークンに置き換える前提の目印。
OPTION_MARKER = "<opt>"

_NOUL_SURFACE = {"false": "いいえ", "true": "はい"}


class InputTooLongError(ValueError):
    """入力が上限を超えた。根拠を切り捨てて判定させないために例外にする。"""

    def __init__(self, n_tokens: int, max_tokens: int):
        super().__init__(f"入力が {n_tokens} トークンで、上限 {max_tokens} を超えている")
        self.n_tokens = n_tokens
        self.max_tokens = max_tokens


@dataclass(frozen=True)
class SerializedInput:
    text: str
    labels: list[str]

    @property
    def n_options(self) -> int:
        return len(self.labels)


def _render_value(value: Json) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_state(state: Json) -> str:
    """dict の state は最上位のキーを見出しにして並べる。質問文から `キー名` で参照できる。"""
    if isinstance(state, dict):
        return "\n\n".join(f"［{k}］\n{_render_value(v)}" for k, v in state.items())
    return _render_value(state)


def _option_lines(
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion, marker: str
) -> list[str]:
    lines = []
    for label, desc in zip(question.labels, question.descriptions, strict=True):
        if isinstance(question, NoulQuestion):
            head = _NOUL_SURFACE[label]
        elif isinstance(question, ScoreQuestion):
            head = f"段階{label}"
        else:
            head = label
        body = f"{head}: {desc}" if desc else head
        lines.append(f"{body} {marker}")
    return lines


def build_input(
    state: Json,
    question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    marker: str = OPTION_MARKER,
) -> SerializedInput:
    rendered = render_state(state)
    user_texts = [rendered, question.instructions, *question.labels, *question.descriptions]
    if any(marker in t for t in user_texts):
        raise ValueError(f"入力に予約済みのマーカー {marker!r} が含まれている")
    parts = [
        "状態:",
        rendered,
        "",
        f"質問: {question.instructions}",
        "候補:",
        *_option_lines(question, marker),
    ]
    return SerializedInput(text="\n".join(parts), labels=question.labels)
