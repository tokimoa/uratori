"""データ JSONL を LLM に判定させ、uratori-eval が読める予測ファイルを書く。

  DEEPSEEK_API_KEY=... uv run python scripts/judge_with_llm.py \
      --endpoint deepseek-flash --data examples/sample_records.jsonl --out runs/judge/x.jsonl

接続先は configs/endpoints.yaml から選ぶ。API キーは接続先ごとの環境変数（api_key_env）から読む。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from uratori.data.rubrics import load_rubrics
from uratori.data.schema import Record
from uratori.data.validate import load_records
from uratori.llm.client import ChatClient, Endpoint
from uratori.llm.judge import JudgeParseError, judge

ROOT = Path(__file__).parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--data", nargs="*", type=Path, default=[])
    p.add_argument("--rubric-examples", action="store_true", help="rubric の例も判定する")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    records: list[Record] = [r for path in args.data for r in load_records(path)]
    if args.rubric_examples:
        for rubric in load_rubrics(ROOT / "rubrics").values():
            records += [rubric.example_record(i) for i in range(len(rubric.examples))]

    endpoint = Endpoint.load(args.endpoint, ROOT / "configs" / "endpoints.yaml")
    client = ChatClient(endpoint, cache_dir=ROOT / "runs" / "llm_cache")

    def run(record: Record) -> dict:
        try:
            j = judge(client, record.state, record.to_question())
        except JudgeParseError as e:
            return {"id": record.id, "error": str(e)}
        return {"id": record.id, "probs": j.probs, "label": j.label, "reason": j.reason}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, records))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    data_out = args.out.with_suffix(".data.jsonl")
    with data_out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    u = client.usage
    failed = sum("error" in r for r in rows)
    print(
        f"{len(rows)} 件（失敗 {failed}、キャッシュ {u.cached_calls}）。"
        f"入力 {u.input_tokens:,} / 出力 {u.output_tokens:,} トークン、"
        f"約 {u.cost_usd(endpoint):.4f} ドル（ピーク価格で計算）"
    )
    print(f"予測 {args.out}、対応するデータ {data_out}")


if __name__ == "__main__":
    main()
