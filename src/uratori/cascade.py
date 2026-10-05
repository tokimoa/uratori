"""判定を 3 層に分ける構成。

第 1 層はコードの検査。モデルに判定させる前に、機械的に分かることを調べる。ここでは
「書かれた側の文章にある数値が、根拠側の文章に出てこない」ことを検出する。これは正誤の
判定ではなく、第 3 層へ回す合図として使う（根拠から計算で導いた数値は正しいことがあるため）。

第 2 層は uratori の判定。第 3 層は強い LLM か人で、第 1 層の合図か第 2 層の低確信の件を回す。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ZENKAKU = str.maketrans("０１２３４５６７８９，．", "0123456789,.")


def numbers_in(text: str) -> set[str]:
    """文章中の数値を、桁区切りと全角を正規化して集める。「万」「億」は展開する。"""
    text = text.translate(_ZENKAKU)
    out = set()
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)(万|億)?", text):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if m.group(2) == "万":
            value *= 10_000
        elif m.group(2) == "億":
            value *= 100_000_000
        out.add(str(int(value)) if value == int(value) else repr(value))
    return out


@dataclass(frozen=True)
class Layer1:
    unsupported_numbers: list[str]

    @property
    def flagged(self) -> bool:
        return bool(self.unsupported_numbers)


def layer1_check(state: dict, source_keys: list[str]) -> Layer1:
    """根拠側のキー以外の文章にある数値のうち、根拠側に出てこないものを返す。"""
    source = " ".join(str(state[k]) for k in source_keys if k in state)
    written = " ".join(str(v) for k, v in state.items() if k not in source_keys)
    missing = sorted(numbers_in(written) - numbers_in(source))
    # 1 桁の数値（「2 つ」「第3条」など）は誤検出が多いので外す
    return Layer1([n for n in missing if len(n.replace(".", "")) >= 2])


def route(probs: np.ndarray, layer1: Layer1, threshold: float, use_rules: bool) -> str:
    """'accept'（uratori の答えを使う）か 'escalate'（第 3 層へ）を返す。"""
    if use_rules and layer1.flagged:
        return "escalate"
    return "accept" if float(probs.max()) >= threshold else "escalate"
