"""モデルを読み込んで、1 つの state と複数の質問から確率を返す。

UratoriPredictor は scripts/train.py が書いた checkpoint のフォルダを読む。HubPredictor は
Hugging Face Hub に公開した形（AutoModel + trust_remote_code）のモデルを読む。
どちらも predict の形は同じで、サーバと評価のスクリプトから同じように使える。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uratori.eval.calibrate import apply_temperature
from uratori.serialize import OPTION_MARKER, InputTooLongError, build_input
from uratori.types import ChoiceQuestion, Json, NoulQuestion, ScoreQuestion

AnyQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


def temperature_for(question: AnyQuestion, temperatures: dict[str, float]) -> float:
    """校正の温度を、(型, 候補数)、型、全体の順に探す。なければ 1。"""
    for key in (f"{question.type}:{len(question.labels)}", question.type, "all"):
        if key in temperatures:
            return temperatures[key]
    return 1.0


class UratoriPredictor:
    def __init__(
        self, ckpt: Path, device: str = "mps", max_tokens: int = 2048, batch_size: int = 8
    ):
        import torch

        from uratori.train.loop import load_checkpoint

        self._torch = torch
        self.model, self.tokenizer, self.marker, self.meta = load_checkpoint(ckpt, device)
        self.model.eval()
        self.device, self.max_tokens, self.batch_size = device, max_tokens, batch_size
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        calibration = ckpt / "calibration.json"
        self.temperatures: dict[str, float] = (
            json.loads(calibration.read_text(encoding="utf-8")) if calibration.exists() else {}
        )
        self.name = ckpt.name

    def predict(
        self, state: Json, questions: dict[str, AnyQuestion]
    ) -> tuple[dict[str, list[float]], int]:
        """質問 id ごとの確率（labels の順）と、入力トークン数の合計を返す。"""
        torch = self._torch
        encoded = []
        for qid, question in questions.items():
            text = build_input(state, question).text.replace(OPTION_MARKER, self.marker)
            ids = self.tokenizer(text, truncation=False)["input_ids"]
            if len(ids) > self.max_tokens:
                raise InputTooLongError(len(ids), self.max_tokens)
            encoded.append((qid, ids))
        out: dict[str, list[float]] = {}
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][1]))
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch = [encoded[i] for i in order[start : start + self.batch_size]]
                width = max(len(ids) for _, ids in batch)
                input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for row, (_, ids) in enumerate(batch):
                    input_ids[row, : len(ids)] = torch.tensor(ids)
                    attention[row, : len(ids)] = 1
                logits, mask = self.model(input_ids.to(self.device), attention.to(self.device))
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                for (qid, _), p, m in zip(batch, probs, mask.cpu().numpy(), strict=True):
                    raw = np.asarray(p[: int(m.sum())], dtype=float)
                    t = temperature_for(questions[qid], self.temperatures)
                    out[qid] = apply_temperature(raw, t).tolist() if t != 1.0 else raw.tolist()
        return out, sum(len(ids) for _, ids in encoded)


def question_payload(question: AnyQuestion) -> dict:
    """質問を wire と同じ形の dict に戻す。Hub のモデルの predict はこの形を受け取る。"""
    payload: dict = {"type": question.type, "instructions": question.instructions}
    if question.criteria is not None:
        payload["criteria"] = question.criteria
    return payload


def probs_from_answer(question: AnyQuestion, answer: dict) -> list[float]:
    """Hub のモデルが返す回答から、labels の順の確率を取り出す。"""
    if isinstance(question, NoulQuestion):
        return [1.0 - answer["noul"], answer["noul"]]
    return [answer["probabilities"][label] for label in question.labels]


class HubPredictor:
    """Hub に公開した形のモデルで判定する。

    判定はモデルに同梱されたコードの predict に任せる。校正の温度と入力の上限は、モデルの
    config に入っている値を使う。
    """

    def __init__(self, model, tokenizer, name: str, batch_size: int = 8):
        self.model, self.tokenizer, self.name, self.batch_size = model, tokenizer, name, batch_size
        self.marker: str = model.config.marker_token
        self.max_tokens: int = model.config.max_tokens

    @classmethod
    def from_pretrained(
        cls, repo: str, device: str = "cpu", revision: str | None = None, batch_size: int = 8
    ) -> HubPredictor:
        """repo は Hub の repo id（tokimoa/uratori-ja-310m など）か、同じ形のローカルのフォルダ。

        モデルに同梱されたコードを実行するので、信頼できる repo だけを指定する。
        """
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        return cls(model.to(device).eval(), tokenizer, repo, batch_size)

    def predict(
        self, state: Json, questions: dict[str, AnyQuestion]
    ) -> tuple[dict[str, list[float]], int]:
        """質問 id ごとの確率（labels の順）と、入力トークン数の合計を返す。"""
        payload, n_tokens = {}, 0
        for qid, question in questions.items():
            # モデルに同梱されたコードと同じ直列化で長さを数え、上限を超えたら先に止める
            text = build_input(state, question, self.marker).text
            n = len(self.tokenizer(text, truncation=False)["input_ids"])
            if n > self.max_tokens:
                raise InputTooLongError(n, self.max_tokens)
            n_tokens += n
            payload[qid] = question_payload(question)
        answers = self.model.predict(self.tokenizer, state, payload, batch_size=self.batch_size)
        probs = {qid: probs_from_answer(q, answers[qid]) for qid, q in questions.items()}
        return probs, n_tokens
