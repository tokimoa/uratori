"""合成データの設計。どの rubric を、どんな原文から、どのキーを書かせて作るか。

原文（根拠側）はコードが state に入れ、教師には残りのキーだけを書かせる。根拠側の文章が
原文と一字一句同じであることを、作り方で保証するため。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

# rubric ごとに、原文をそのまま入れるキー。空の rubric は原文を背景としてだけ使う
SOURCE_FIELD: dict[str, str | None] = {
    "gr-support-noul": "根拠",
    "gr-support-3": "根拠",
    "gr-contradict-noul": "根拠",
    "gr-number-match": "根拠",
    "gr-condition-met": "規定",
    "gr-attribution": "根拠",
    "gr-decided-vs-considered": "議事録",
    "gr-citation-supports": "引用箇所",
    "gr-support-degree": "根拠",
    "cmp-contradict-noul": "文書A",
    "cmp-relation-4": "文書A",
    "cmp-same-content": "文A",
    "cmp-added-claim": "文書A",
    "cmp-more-faithful": "根拠",
    "rag-relevance": "文書",
    "rag-answerable": "検索結果",
    "rag-sufficiency": "検索結果",
    "rag-need-more-search": "検索結果",
    "rag-answer-faithful": "検索結果",
    "rag-answer-complete": None,
    "wr-summary-faithful": "原文",
    "wr-requirement-met": None,
    "rt-inquiry-type": None,
}

# 原文の種類が限られる rubric
GENRE_ONLY = {
    "gr-decided-vs-considered": {"議事録"},
    "gr-attribution": {"議事録", "業務メール", "報告書"},
    "gr-condition-met": {"社内規程", "利用規約", "契約書の条項", "行政の案内"},
}
# 1〜2 文の抜粋を原文にする rubric
SENTENCE_SOURCE = {"gr-citation-supports": 2, "cmp-same-content": 1}

GENRES = [
    "社内規程", "利用規約", "契約書の条項", "議事録", "FAQ", "プレスリリース",
    "製品仕様書", "行政の案内", "業務メール", "報告書",
]  # fmt: skip
INDUSTRIES = [
    "製造業", "小売", "ソフトウェア開発", "医療機関", "物流", "学校", "地方銀行", "市役所",
    "建設", "飲食チェーン", "ホテル", "不動産管理", "人材派遣", "農業協同組合", "通信", "出版",
]  # fmt: skip
SUBJECTS = [
    "経費精算", "休暇と勤怠", "情報セキュリティ", "備品の貸出", "納期と検収", "料金改定",
    "障害対応", "採用と研修", "施設の利用", "個人情報の扱い", "保守と点検", "キャンペーン",
    "在庫と発注", "会員制度", "安全衛生", "イベントの運営", "補助金と申請", "品質検査",
]  # fmt: skip
LENGTHS = {"short": (80, 200), "medium": (250, 500), "long": (600, 1100)}
LENGTH_WEIGHTS = {"short": 0.3, "medium": 0.45, "long": 0.25}

PHENOMENA = {
    "主語省略", "否定の作用域", "二重否定", "敬語と間接依頼", "例外条件", "相対時点", "数値",
    "単位換算", "全角半角", "固有名の取り違え", "主体の取り違え", "決定と検討", "部分支持",
    "情報不足", "範囲の含意", "複数文の統合", "言い換え", "時点と版", "文体バイアス", "指示の混入",
}  # fmt: skip

_SENTENCE_END = re.compile(r"(?<=[。！？])")


@dataclass(frozen=True)
class DocSeed:
    doc_id: str
    genre: str
    industry: str
    subject: str
    length: str


def sample_doc_seeds(n: int, seed: int, prefix: str) -> list[DocSeed]:
    rng = random.Random(seed)
    lengths, weights = zip(*LENGTH_WEIGHTS.items(), strict=True)
    return [
        DocSeed(
            doc_id=f"{prefix}-{i:05d}",
            genre=GENRES[i % len(GENRES)],
            industry=rng.choice(INDUSTRIES),
            subject=rng.choice(SUBJECTS),
            length=rng.choices(lengths, weights)[0],
        )
        for i in range(n)
    ]


def rubrics_for(genre: str | None, rubric_ids: list[str], k: int, rng: random.Random) -> list[str]:
    """この原文に使う rubric を k 本選ぶ。genre が None の原文（外部データ）には制限つきの rubric を使わない。"""
    usable = [
        rid
        for rid in rubric_ids
        if rid not in GENRE_ONLY or (genre is not None and genre in GENRE_ONLY[rid])
    ]
    return rng.sample(usable, min(k, len(usable)))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.replace("\n", "")) if s.strip()]


def sentence_excerpt(text: str, n: int, rng: random.Random) -> str | None:
    """連続する n 文を抜き出す。数値や条件を含む文を優先したいので、短すぎる文は避ける。"""
    sentences = split_sentences(text)
    starts = [
        i for i in range(len(sentences) - n + 1) if all(len(s) >= 15 for s in sentences[i : i + n])
    ]
    if not starts:
        return None
    i = rng.choice(starts)
    return "".join(sentences[i : i + n])


# 「支持される」側のラベル。このラベルの件が原文の写しだと、単語の重なりだけで解けてしまう
POSITIVE_LABELS: dict[str, str] = {
    "gr-support-noul": "true",
    "gr-support-3": "支持",
    "gr-citation-supports": "true",
    "gr-support-degree": "2",
    "cmp-same-content": "true",
    "rag-answer-faithful": "忠実",
    "wr-summary-faithful": "4",
}
COPY_RATIO_LIMIT = 0.6


def is_dev_doc(doc_id: str, dev_fraction: float = 0.08) -> bool:
    """原文の単位で学習用と確認用に分ける。同じ原文の問題が両方に入らないようにする。"""
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < dev_fraction


# ---- rubric にない質問を教師に考えさせる経路 ------------------------------------------

# 質問の切り口の手がかり。呼び出しごとに 1 つ渡して、質問の種類を散らす
QUESTION_KINDS = [
    "手続きや作業の順番が正しいか", "期限や締め切りを守っているか", "費用や責任を誰が負うか",
    "数量や金額の合計、差、割合が合っているか", "条件がどれだけ厳しいか、緩いか", "例外や但し書きが当てはまるか",
    "前提が誤っている質問や主張かどうか", "承認や許可が必要か、誰の承認か", "対象に含まれるか、対象外か",
    "原因と結果の述べ方が原文と合っているか", "優先順位や重要度の述べ方が合っているか", "依頼か、報告か、提案か",
    "断っているのか、受けているのか", "必要な項目が全て書かれているか", "禁止されている行為に当たるか",
    "二つの記述のどちらが新しい、または詳しいか", "言い換えで意味が変わっていないか", "固有名や役職の対応が合っているか",
    "推測や見込みを事実として述べていないか", "回答が質問の意図に合っているか", "注意点やリスクに触れているか",
    "手順を省略していないか", "数値の単位や桁が合っているか", "要求の緊急度や深刻さ",
]  # fmt: skip
# 学習に使わない rubric と同じ切り口。汎化を測るために取っておくので、教師に作らせない
EXCLUDED_KINDS = [
    "ある基準日の時点で記述が有効かどうか（版、改定、施行日）",
    "旧版から新版への変更が内容にどの程度影響するか",
    "二つの文の含意の向き",
    "二つの検索結果どうしが食い違うか",
    "発言に担当者の決まったタスクが含まれるか",
]
SOURCE_KEY_NAMES = ["文書", "資料", "原文", "記録", "本文"]
# 弱かった現象を厚めに狙う（最初に学習したモデルで正答率が低かったもの）
PHENOMENA_WEIGHTS = {
    "数値": 3, "複数文の統合": 3, "範囲の含意": 3, "時点と版": 2, "相対時点": 3, "決定と検討": 3,
    "単位換算": 2, "否定の作用域": 2, "二重否定": 2, "主語省略": 2, "例外条件": 2,
}  # fmt: skip


def sample_phenomena(rng: random.Random, k: int = 2) -> list[str]:
    names = sorted(PHENOMENA - {"指示の混入"})
    weights = [PHENOMENA_WEIGHTS.get(n, 1) for n in names]
    out: list[str] = []
    while len(out) < k:
        pick = rng.choices(names, weights)[0]
        if pick not in out:
            out.append(pick)
    return out
