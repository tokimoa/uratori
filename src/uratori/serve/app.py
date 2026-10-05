"""/v1/systemone 互換の HTTP サーバ。

  # Hugging Face Hub に公開しているモデルで立てる
  uv run --group train --group serve python -m uratori.serve.app --model tokimoa/uratori-ja-310m --port 8000

  # 自分で学習した checkpoint で立てる
  uv run --group train --group serve python -m uratori.serve.app --ckpt runs/mbj310 --port 8000

リクエストとレスポンスの形は https://docs.typesafe.ai/api.md に合わせた。TypeSafe の SDK や
互換のベンチマーク runner の向き先を変えるだけで使える。認証はしない（ローカルで使う前提）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from uratori.serialize import InputTooLongError
from uratori.serve.wire import SystemOneRequest, SystemOneResponse, Usage, answer_from_probs
from uratori.types import Json


class Predictor(Protocol):
    name: str

    def predict(self, state: Json, questions: dict) -> tuple[dict[str, list[float]], int]: ...


def create_app(predictor: Predictor) -> FastAPI:
    app = FastAPI(title="uratori", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": predictor.name}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": predictor.name, "object": "model"}]}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def system_one(request: SystemOneRequest):
        try:
            probs, n_tokens = predictor.predict(request.state, request.questions)
        except InputTooLongError as e:
            # 入力を黙って切って判定することはしない
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "input_too_long", "message": str(e),
                                   "tokens": e.n_tokens, "max_tokens": e.max_tokens}},
            )  # fmt: skip
        answers = {qid: answer_from_probs(q, probs[qid]) for qid, q in request.questions.items()}
        return SystemOneResponse(
            model=predictor.name, answers=answers, usage=Usage(input_tokens=n_tokens)
        )

    return app


def main() -> None:
    import uvicorn

    from uratori.serve.predictor import HubPredictor, UratoriPredictor

    p = argparse.ArgumentParser(description="/v1/systemone 互換のサーバを立てる")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help="Hub の repo id（tokimoa/uratori-ja-310m など）か、同じ形のローカルのフォルダ",
    )
    source.add_argument("--ckpt", type=Path, help="scripts/train.py が書いた checkpoint のフォルダ")
    p.add_argument("--revision", help="--model で読む revision（commit の hash など）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cpu", help="cpu、cuda、mps のどれか")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="--ckpt のときの入力の上限。--model はモデルの config の値を使う",
    )
    args = p.parse_args()
    if args.model:
        predictor = HubPredictor.from_pretrained(args.model, args.device, args.revision)
    else:
        predictor = UratoriPredictor(args.ckpt, args.device, args.max_tokens)
    uvicorn.run(create_app(predictor), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
