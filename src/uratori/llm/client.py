"""OpenAI 互換 API の薄いクライアント。応答をディスクにキャッシュし、使用量を数える。

教師の生成、検証、ベースラインの LLM judge が全てここを通る。同じ入力を二度課金しない
ことと、何トークン使ったかを後から言えることが目的。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

RETRY_STATUS = {429, 500, 502, 503, 504, 529}
MAX_RETRIES = 5


@dataclass(frozen=True, eq=False)
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    price_in: float = 0.0
    price_out: float = 0.0
    max_tokens: int = 4096
    # 出力上限のパラメータ名。OpenAI の新しいモデルは max_completion_tokens しか受けない
    max_tokens_param: str = "max_tokens"
    # False の接続先には response_format を送らない（互換 API が受けない場合）
    json_mode: bool = True
    # リクエストにそのまま足すパラメータ（ローカルモデルの思考を切る指定など）
    extra_body: dict | None = None

    @classmethod
    def load(cls, name: str, path: Path) -> Endpoint:
        table = yaml.safe_load(path.read_text(encoding="utf-8"))
        if name not in table:
            raise KeyError(f"接続先 {name!r} が {path} にない（あるのは {sorted(table)}）")
        return cls(name=name, **table[name])


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int, cached: bool) -> None:
        with self._lock:
            self.calls += 1
            if cached:
                self.cached_calls += 1
            else:
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens

    def cost_usd(self, endpoint: Endpoint) -> float:
        return (
            self.input_tokens * endpoint.price_in + self.output_tokens * endpoint.price_out
        ) / 1e6


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool
    finish_reason: str | None = None


class ChatClient:
    def __init__(self, endpoint: Endpoint, cache_dir: Path | None = None, timeout: float = 120.0):
        self.endpoint = endpoint
        self.usage = Usage()
        self.cache_dir = cache_dir / endpoint.name if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        headers = {}
        if endpoint.api_key_env:
            key = os.environ.get(endpoint.api_key_env)
            if not key:
                raise RuntimeError(f"環境変数 {endpoint.api_key_env} がない")
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(base_url=endpoint.base_url, headers=headers, timeout=timeout)

    def chat(self, messages: list[dict], *, json_mode: bool = False, **params) -> ChatResult:
        body = {
            "model": self.endpoint.model,
            "messages": messages,
            self.endpoint.max_tokens_param: self.endpoint.max_tokens,
        }
        if self.endpoint.extra_body:
            body.update(self.endpoint.extra_body)
        if json_mode and self.endpoint.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(params)
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.usage.add(saved["input_tokens"], saved["output_tokens"], cached=True)
            return ChatResult(saved["text"], saved["input_tokens"], saved["output_tokens"], True)

        data = self._post(body)
        usage = data.get("usage") or {}
        result = ChatResult(
            text=data["choices"][0]["message"]["content"] or "",
            finish_reason=data["choices"][0].get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached=False,
        )
        # 空の応答は保存しない。推論で出力上限を使い切った場合などで、再試行すれば通ることがある
        if path and result.text.strip():
            saved = {"text": result.text, "input_tokens": result.input_tokens,
                     "output_tokens": result.output_tokens, "model": data.get("model")}  # fmt: skip
            path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        self.usage.add(result.input_tokens, result.output_tokens, cached=False)
        return result

    def _post(self, body: dict) -> dict:
        for attempt in range(MAX_RETRIES):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("到達しない")
