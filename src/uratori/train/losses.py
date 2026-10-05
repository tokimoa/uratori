"""学習に使う損失。

どの関数も件ごとの損失 [batch] を返す。型ごとの正規化は combine_by_type で行う。
logits は OptionScorer の出力で、候補のない位置は dtype の最小値で埋まっている前提。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def hard_label_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """確定ラベルへの cross entropy。"""
    return F.cross_entropy(logits.float(), target, reduction="none")


def soft_label_loss(
    logits: torch.Tensor, option_mask: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    """教師分布 q に対する KL(q || p)。q のエントロピーを引いてあるので、p = q で 0 になる。"""
    log_p = F.log_softmax(logits.float(), dim=-1)
    q = target_probs * option_mask
    log_q = torch.log(q.clamp_min(1e-12))
    return (q * (log_q - log_p) * option_mask).sum(dim=-1)


def ordinal_loss(
    logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """累積分布の差の二乗和（ranked probability score）。

    正解から遠い段階に確率を置くほど大きくなる。隣の段階への誤りと遠い段階への誤りを
    区別するために、score 型の補助損失として足す。
    """
    p = F.softmax(logits.float(), dim=-1) * option_mask
    onehot = F.one_hot(target, logits.shape[-1]).to(p.dtype)
    diff = (p.cumsum(dim=-1) - onehot.cumsum(dim=-1)) * option_mask
    return (diff**2).sum(dim=-1)


def combine_by_type(per_example: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
    """型ごとに平均してから、batch に現れた型の間で平均する。

    件数の多い型や損失の大きい型に勾配が偏らないようにするため。
    """
    means = [per_example[type_ids == t].mean() for t in torch.unique(type_ids)]
    return torch.stack(means).mean()
