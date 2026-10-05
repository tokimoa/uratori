"""検証前の候補から、学習用と確認用のファイルを作る（検証を待たずに学習するため）。

  uv run python scripts/build_unverified.py demo demo2

先に挙げた名前のデータと内容が重なる件は、後の名前のほうから除く。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify_and_build import load_candidates  # noqa: E402

ROOT = Path(__file__).parent.parent


def main() -> None:
    seen: set[str] = set()
    for name in sys.argv[1:]:
        base = ROOT / "data" / "synth" / name
        records = [r for r in load_candidates(base) if r.content_hash() not in seen]
        seen |= {r.content_hash() for r in records}
        for split in ("train", "dev"):
            rows = [r for r in records if r.split.value == split]
            (base / f"unverified_{split}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
            )
        counts = Counter(r.split.value for r in records)
        print(
            name, dict(counts), "family", len({r.family_id for r in records}),
            "型", dict(Counter(r.type.value for r in records)),
            "rubric なし", sum(r.tags.rubric_id is None for r in records),
            "対", sum(r.provenance.contrast_of is not None for r in records),
        )  # fmt: skip


if __name__ == "__main__":
    main()
