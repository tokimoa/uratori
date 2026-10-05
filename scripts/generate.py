"""合成データの候補を作る。検証は verify_and_build.py で行う。

  DEEPSEEK_API_KEY=... uv run python scripts/generate.py --name demo --docs 40 --budget-usd 2

生成に使う LLM は configs/endpoints.yaml の接続先から選ぶ（--generator）。API キーは接続先ごとの
環境変数（api_key_env）から読む。

原文をまとまりごとに処理し、まとまりが終わるたびに data/synth/<name>/chunks/ に書く。
途中で止めても、同じコマンドで続きから再開できる。費用（ピーク価格で計算）が上限を超えたら止まる。

外部の文書を原文に使うときは、先にファイルを置いておく。
  --external-docs  data/external/jagovfaqs/raw.jsonl（JaGovFaqs-22k の行。Question、Answer、copyright、url を使う）
  --wiki-docs      data/external/jawiki/docs.jsonl（1 行に doc_id、text、source、license）
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from uratori.data.rubrics import load_rubrics
from uratori.data.validate import load_records
from uratori.gen.family import make_family_candidates
from uratori.gen.pipeline import (
    SourceDoc,
    Stats,
    code_checks,
    make_candidates,
    make_docs,
    make_open_candidates,
)
from uratori.gen.plan import sample_doc_seeds
from uratori.llm.client import ChatClient, Endpoint

ROOT = Path(__file__).parent.parent


def external_docs(n: int, seed: int) -> list[SourceDoc]:
    """JaGovFaqs の回答文を原文にする。長さが 120〜1,100 字のものから無作為に選ぶ。"""
    if n == 0:
        return []
    path = ROOT / "data" / "external" / "jagovfaqs" / "raw.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [(i, r) for i, r in enumerate(rows) if 120 <= len(r["Answer"]) <= 1100]
    picked = random.Random(seed).sample(rows, n)
    return [
        SourceDoc(
            doc_id=f"govfaq-{i:05d}",
            text=f"{r['Question']}\n{r['Answer']}".strip(),
            genre=None,
            source=f"JaGovFaqs-22k #{i} {r['copyright']} {r['url']}",
            license="CC-BY-4.0",
        )
        for i, r in picked
    ]


def wiki_docs(n: int, seed: int) -> list[SourceDoc]:
    """日本語 Wikipedia の段落（CC BY-SA 4.0）。data/external/jawiki/docs.jsonl から選ぶ。"""
    if n == 0:
        return []
    path = ROOT / "data" / "external" / "jawiki" / "docs.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    return [SourceDoc(r["doc_id"], r["text"], None, r["source"], r["license"]) for r in picked]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--docs", type=int, default=40, help="教師に書かせる原文の数")
    p.add_argument("--external-docs", type=int, default=0, help="JaGovFaqs から取る原文の数")
    p.add_argument("--wiki-docs", type=int, default=0, help="日本語 Wikipedia から取る原文の数")
    p.add_argument("--rubrics-per-doc", type=int, default=2)
    p.add_argument(
        "--families", type=int, default=0, help="原文と問題をまとめて作らせる family の数"
    )
    p.add_argument("--trap", action="store_true", help="family 方式で、ひっかけ優先の指示を付ける")
    p.add_argument(
        "--open-per-doc", type=int, default=0, help="rubric にない質問を 1 原文あたり何問作るか"
    )
    p.add_argument("--chunks", type=int, default=10, help="原文を何回に分けて処理するか")
    p.add_argument("--budget-usd", type=float, default=5.0, help="ピーク価格で計算した費用の上限")
    p.add_argument("--generator", default="deepseek-flash")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--protected", nargs="*", type=Path, default=[],
        help="評価に使うデータの JSONL。文面が重なる候補を落とす",
    )  # fmt: skip
    args = p.parse_args()

    out = ROOT / "data" / "synth" / args.name / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    endpoint = Endpoint.load(args.generator, ROOT / "configs" / "endpoints.yaml")
    generator = ChatClient(endpoint, cache_dir=ROOT / "runs" / "llm_cache")
    rubrics = load_rubrics(ROOT / "rubrics")
    protected = [r for path in args.protected for r in load_records(path)]
    seeds = sample_doc_seeds(args.docs, args.seed, args.name)
    external = external_docs(args.external_docs, args.seed)
    wiki = wiki_docs(args.wiki_docs, args.seed)

    spent = 0.0
    for chunk in range(args.chunks):
        chunk_path = out / f"{chunk:04d}.jsonl"
        meta_path = out / f"{chunk:04d}.meta.json"
        if meta_path.exists():
            spent += json.loads(meta_path.read_text())["usd_peak"]
            continue
        if spent >= args.budget_usd:
            print(f"費用が上限に達したので止める（{spent:.2f} / {args.budget_usd} ドル）")
            break
        started = time.perf_counter()
        before = generator.usage.cost_usd(endpoint)
        tokens_before = (generator.usage.input_tokens, generator.usage.output_tokens)
        stats = Stats()
        docs = make_docs(generator, seeds[chunk :: args.chunks], stats, args.workers)
        docs += external[chunk :: args.chunks]
        docs += wiki[chunk :: args.chunks]
        candidates = make_candidates(
            generator, docs, rubrics, args.rubrics_per_doc, args.seed + chunk, stats,
            args.workers, args.name,
        )  # fmt: skip
        if args.families:
            candidates += make_family_candidates(
                generator, rubrics, args.families // args.chunks, args.seed * 1000 + chunk, stats,
                args.workers, args.name, trap=args.trap,
            )  # fmt: skip
        if args.open_per_doc:
            candidates += make_open_candidates(
                generator,
                docs,
                args.open_per_doc,
                args.seed + chunk,
                stats,
                args.workers,
                args.name,
            )
        code_checks(candidates, protected, stats)
        with chunk_path.open("w", encoding="utf-8") as f:
            for c in candidates:
                row = {"status": c.status, "reject_reason": c.reject_reason,
                       "record": json.loads(c.record.model_dump_json())}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        cost = generator.usage.cost_usd(endpoint) - before
        spent += cost
        meta = {
            "docs": len(docs),
            "candidates": len(candidates),
            "pending": sum(c.status == "pending" for c in candidates),
            "counts": dict(stats.counts),
            "usd_peak": round(cost, 4),
            "input_tokens": generator.usage.input_tokens - tokens_before[0],
            "output_tokens": generator.usage.output_tokens - tokens_before[1],
            "seconds": round(time.perf_counter() - started, 1),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"chunk {chunk}: {meta}", flush=True)

    metas = [json.loads(m.read_text()) for m in sorted(out.glob("*.meta.json"))]
    total = Counter()
    for m in metas:
        total.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
        total.update({f"count_{k}": v for k, v in m["counts"].items()})
    print(
        json.dumps(
            {"chunks": len(metas), **{k: round(v, 3) for k, v in total.items()}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
