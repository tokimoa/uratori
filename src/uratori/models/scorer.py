"""候補マーカーの位置を採点する共通スコアラ。

backbone の出力から、入力中のマーカートークンの位置の hidden state を取り出して 1 つの
head で採点する。noul、choice、score のどれも同じ経路を通り、違うのは損失だけ。
候補数は入力ごとに変わってよく、足りない分は mask で埋める。
"""

from __future__ import annotations

import torch
from torch import nn

from uratori.serialize import OPTION_MARKER, InputTooLongError, SerializedInput


class OptionScorer(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, marker_id: int):
        super().__init__()
        self.backbone = backbone
        self.marker_id = marker_id
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """logits と option_mask を返す。どちらも [batch, その batch の最大候補数]。

        候補のない位置の logits は dtype の最小値で埋めてあり、softmax すると 0 になる。
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        scores = self.head(hidden.to(self.head[0].weight.dtype)).squeeze(-1)
        return gather_option_scores(scores, input_ids == self.marker_id)


def gather_option_scores(
    scores: torch.Tensor, is_marker: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[batch, seq] のスコアから、マーカー位置の値を出現順に左詰めで集める。"""
    counts = is_marker.sum(dim=1)
    if (counts < 2).any():
        raise ValueError("候補マーカーが 2 個未満の入力がある")
    k_max = int(counts.max())
    # マーカーの位置を前に寄せる。stable な sort なので出現順が保たれる
    order = torch.argsort((~is_marker).to(torch.int8), dim=1, stable=True)[:, :k_max]
    option_mask = torch.arange(k_max, device=scores.device)[None, :] < counts[:, None]
    logits = scores.gather(1, order).masked_fill(~option_mask, torch.finfo(scores.dtype).min)
    return logits, option_mask


def resolve_marker(tokenizer) -> tuple[str, int, bool]:
    """マーカーに使うトークンを決める。(文字列, id, 語彙に追加したか) を返す。

    mask トークンを持つエンコーダはそれを使う。MLM の事前学習で「この位置の中身を文脈から
    当てる」ように訓練されているので、採点位置として素性が良い。
    持たないトークナイザには専用トークンを足すので、呼び出し側で埋め込みを resize する。
    """
    if tokenizer.mask_token is not None:
        return tokenizer.mask_token, tokenizer.mask_token_id, False
    tokenizer.add_special_tokens({"additional_special_tokens": [OPTION_MARKER]})
    return OPTION_MARKER, tokenizer.convert_tokens_to_ids(OPTION_MARKER), True


def encode(
    tokenizer, inputs: list[SerializedInput], marker: str, marker_id: int, max_tokens: int
) -> dict[str, torch.Tensor]:
    """直列化済みの入力をトークン化する。上限を超える入力は切らずに例外にする。"""
    texts = [x.text.replace(OPTION_MARKER, marker) for x in inputs]
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    lengths = batch["attention_mask"].sum(dim=1)
    if int(lengths.max()) > max_tokens:
        raise InputTooLongError(int(lengths.max()), max_tokens)
    n_markers = (batch["input_ids"] == marker_id).sum(dim=1).tolist()
    expected = [x.n_options for x in inputs]
    if n_markers != expected:
        raise ValueError(f"マーカーの数が候補数と合わない（{n_markers} と {expected}）")
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
