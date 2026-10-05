"""生成した候補を別の判定者で検証し、学習用のデータを組み立てる。

  uv run python scripts/verify_and_build.py --name demo --verifier local-mlx-nothink

判定者は configs/endpoints.yaml の接続先から選ぶ。API を使う接続先は、キーを環境変数に入れておく。

判定は 1 件ごとに verify__<verifier>.jsonl に足していくので、途中で止めても続きから再開できる。
判定者が設計したラベルと一致した件を train.jsonl と dev.jsonl に、一致しなかった件を
disputed.jsonl に書く。disputed は捨てずに残す（判定者が弱くて難しい件を落としている恐れがあるため）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from uratori.data.schema import Record, Verification
from uratori.data.validate import validate_files
from uratori.llm.client import ChatClient, Endpoint
from uratori.llm.judge import JudgeParseError, judge

ROOT = Path(__file__).parent.parent


def load_candidates(base: Path) -> list[Record]:
    """コードの検査を通った候補を読む。まとまりをまたいだ重複はここで落とす。"""
    seen, out = set(), []
    for path in sorted((base / "chunks").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "rejected":
                continue
            record = Record.model_validate(row["record"])
            if record.content_hash() in seen:
                continue
            seen.add(record.content_hash())
            out.append(record)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--verifier", required=True)
    p.add_argument("--generator", default="deepseek-flash")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--build-only", action="store_true", help="判定をせず、済んでいる判定だけで組み立てる"
    )
    args = p.parse_args()

    base = ROOT / "data" / "synth" / args.name
    records = load_candidates(base)
    verify_path = base / f"verify__{args.verifier}.jsonl"
    done: dict[str, dict] = {}
    if verify_path.exists():
        for line in verify_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "label" in row:  # 読めなかった件は再開時にやり直す
                done[row["id"]] = row

    todo = [] if args.build_only else [r for r in records if r.id not in done]
    if todo:
        endpoint = Endpoint.load(args.verifier, ROOT / "configs" / "endpoints.yaml")
        client = ChatClient(endpoint, cache_dir=ROOT / "runs" / "llm_cache", timeout=600.0)
        lock = Lock()
        started = time.perf_counter()

        def run(record: Record) -> None:
            try:
                j = judge(client, record.state, record.to_question())
                row = {"id": record.id, "label": j.label, "probs": j.probs, "reason": j.reason}
            except JudgeParseError as e:
                row = {"id": record.id, "error": str(e)}
            except Exception as e:  # 接続の失敗などで全体を止めない。再開時にやり直す
                row = {"id": record.id, "error": f"{type(e).__name__}: {e}"}
            with lock:
                with verify_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if "label" in row:
                    done[row["id"]] = row
                if len(done) % 200 == 0:
                    rate = (time.perf_counter() - started) / max(1, len(done))
                    print(f"{len(done)}/{len(records)} 件", f"{rate:.2f} 秒/件", flush=True)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run, todo))
        print(f"判定 {len(todo)} 件に {time.perf_counter() - started:.0f} 秒")

    accepted, disputed, unjudged = [], [], 0
    for record in records:
        row = done.get(record.id)
        if row is None:
            unjudged += 1
            continue
        agreed = row["label"] == record.target_hard
        updated = record.model_copy(
            update={
                "teacher_reported_probs": row["probs"],
                "verification": Verification(
                    status="verified" if agreed else "disputed",
                    by=[args.generator, args.verifier] if agreed else [args.generator],
                    note=record.verification.note
                    if agreed
                    else f"{record.verification.note} / {args.verifier} は {row['label']}: {row['reason']}",
                ),
                "provenance": record.provenance.model_copy(
                    update={"teacher_model": args.generator}
                ),
            }
        )
        (accepted if agreed else disputed).append(updated)

    kept = {r.id for r in accepted}
    accepted = [
        r
        if r.provenance.contrast_of is None or r.provenance.contrast_of in kept
        else r.model_copy(
            update={"provenance": r.provenance.model_copy(update={"contrast_of": None})}
        )
        for r in accepted
    ]
    files = {
        "train.jsonl": [r for r in accepted if r.split.value == "train"],
        "dev.jsonl": [r for r in accepted if r.split.value == "dev"],
        "disputed.jsonl": disputed,
    }
    for name, rows in files.items():
        with (base / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.model_dump_json() + "\n")
    report = validate_files([base / "train.jsonl", base / "dev.jsonl"])

    def lengths(rows: list[Record]) -> dict:
        sizes = [sum(len(str(v)) for v in r.state.values()) for r in rows]
        return {"80字以下": sum(s <= 80 for s in sizes), "80〜400字": sum(80 < s <= 400 for s in sizes),
                "400字超": sum(s > 400 for s in sizes)}  # fmt: skip

    by_rubric = Counter(r.tags.rubric_id for r in records)
    ok_by_rubric = Counter(r.tags.rubric_id for r in accepted)
    summary = {
        "candidates": len(records),
        "judged": len(records) - unjudged,
        "accepted": len(accepted),
        "disputed": len(disputed),
        "acceptance_rate": round(len(accepted) / max(1, len(records) - unjudged), 4),
        "train": len(files["train.jsonl"]),
        "dev": len(files["dev.jsonl"]),
        "families": len({r.family_id for r in accepted}),
        "pairs": sum(r.provenance.contrast_of is not None for r in accepted),
        "type": dict(Counter(r.type.value for r in accepted)),
        "domain": dict(Counter(r.tags.domain for r in accepted)),
        "difficulty": dict(Counter(r.tags.difficulty for r in accepted)),
        "source": dict(Counter(r.provenance.source.split(":")[0].split(" ")[0] for r in accepted)),
        "length": lengths(accepted),
        "disputed_difficulty": dict(Counter(r.tags.difficulty for r in disputed)),
        "by_rubric": {k: f"{ok_by_rubric[k]}/{v}" for k, v in sorted(by_rubric.items())},
        "validator_ok": report.ok,
        "validator_errors": report.errors[:3],
    }
    (base / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
