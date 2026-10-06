# uratori

[English README](README.md)

日本語の文章（state）と型つきの質問を渡すと、文章を生成せずに、真偽（noul）、選択（choice、2〜8 択）、段階評価（score、2〜10 段階）の確率分布を 1 回の forward で返すモデルです。与えた根拠に主張や回答が支持されるかの検証（根拠検証）、2 つの文書の比較（矛盾、意味の変更）、RAG の判断（検索結果の関連性と十分性、回答の忠実性）を対象に学習しています。入出力の形は TypeSafe の `/v1/systemone` API と同じにしてあるので、同じ質問定義をそのまま両方に送れます。TypeSafe の Jev に着想を得た独立の実装で、TypeSafe とは関係がありません。

この repository には、学習、評価、HTTP サーバのコードを置いています。モデルと評価セットは Hugging Face にあります。

## 公開しているモデルと評価セット

| 名前 | 内容 |
|------|------|
| [tokimoa/uratori-ja-310m](https://huggingface.co/tokimoa/uratori-ja-310m) v0.2 | ModernBERT-Ja-310M を土台にしたモデル。CPU で動く |
| [tokimoa/uratori-ja-2b](https://huggingface.co/tokimoa/uratori-ja-2b) | Qwen3.5-2B を土台にしたモデル（LoRA を統合済み） |
| [tokimoa/uratori-ja-4b](https://huggingface.co/tokimoa/uratori-ja-4b) | Qwen3.5-4B を土台にしたモデル（LoRA を統合済み） |
| [tokimoa/uratori-ja-9b](https://huggingface.co/tokimoa/uratori-ja-9b) | Qwen3.5-9B を土台にしたモデル（LoRA を統合済み）。最も精度が高い。4bit で読めば 8 GB の GPU で動く |
| [tokimoa/uratori-ja-2b-mlx-8bit](https://huggingface.co/tokimoa/uratori-ja-2b-mlx-8bit)、[-4bit](https://huggingface.co/tokimoa/uratori-ja-2b-mlx-4bit) | 2b の MLX 版（Apple silicon） |
| [tokimoa/uratori-ja-4b-mlx-8bit](https://huggingface.co/tokimoa/uratori-ja-4b-mlx-8bit)、[-4bit](https://huggingface.co/tokimoa/uratori-ja-4b-mlx-4bit) | 4b の MLX 版（Apple silicon） |
| [tokimoa/uratori-ja-eval](https://huggingface.co/datasets/tokimoa/uratori-ja-eval) | 評価セット |

uratori-ja-eval での Accuracy です（公開した重みで測った値）。参考として、TypeSafe の Jev 1.13.0 は同じ split で 0.903 と 0.900 です。評価セットの正解は人が付けたものではなく LLM の多数決です。作り方と測れないことは評価セットのカードに書いてあります。

| モデル | test（802 件） | challenge（300 件） |
|--------|----------------|---------------------|
| uratori-ja-9b | 0.887 | 0.903 |
| uratori-ja-4b | 0.867 | 0.863 |
| uratori-ja-2b | 0.766 | 0.800 |
| uratori-ja-310m v0.2 | 0.686 | 0.693 |
| 2b の MLX 8bit / 4bit | 0.768 / 0.754 | 0.807 / 0.783 |
| 4b の MLX 8bit / 4bit | 0.865 / 0.857 | 0.863 / 0.853 |

## インストール

Python 3.11 以上と [uv](https://docs.astral.sh/uv/) を使います。

```sh
git clone https://github.com/tokimoa/uratori.git
cd uratori
uv sync --group train --group serve
```

依存はグループに分けてあります。グループなしで入るのは、データの schema、評価指標、LLM のクライアントまでです。`train` は torch と transformers（学習と推論）、`serve` は HTTP サーバ、`export` は ONNX への書き出し、`mlx` はローカルの MLX サーバ（合成データの生成と検証に使う）を足します。

## クイックスタート

### Python から使う

公開しているモデルは transformers だけで読めます。モデルの repo に同梱されたコードを実行するので、`trust_remote_code=True` が必要です。

```python
from transformers import AutoModel, AutoTokenizer

repo = "tokimoa/uratori-ja-310m"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

state = {
    "根拠": "返品は商品到着後14日以内に限り受け付けます。開封済みの商品は返品できません。",
    "主張": "開封済みの商品でも、到着から7日以内なら返品できる。",
}
questions = {
    "support": {
        "type": "choice",
        "instructions": "`主張` は `根拠` から支持されるか",
        "criteria": {
            "支持": "根拠だけから主張の全体が成り立つ。",
            "矛盾": "根拠と両立しない部分がある。",
            "情報不足": "矛盾はないが、根拠からは成否を決められない部分がある。",
        },
    },
    "contradict": {
        "type": "noul",
        "instructions": "`主張` は `根拠` と矛盾するか",
        "criteria": {"true": "根拠と両立しない部分がある。", "false": "両立しない部分はない。"},
    },
}
answers = model.predict(tokenizer, state, questions)
print(answers["support"]["choice"], answers["support"]["probabilities"])
# 矛盾 {'支持': 0.12, '矛盾': 0.71, '情報不足': 0.17}
print(answers["contradict"]["noul"])
# 0.76
```

`state` は文字列か、名前つきの文章の dict です。dict のキーは、質問文から `` `キー名` `` で参照できます。1 回の呼び出しに質問をいくつでも入れられ、質問どうしは独立に判定されます。

質問の型は 3 つあります。

| type | criteria | 返る値 |
|------|----------|--------|
| `noul` | `{"true": "…", "false": "…"}`（省略可） | `noul`（はいの確率） |
| `choice` | `{"候補": "説明", …}`（2〜8 個） | `choice`、`probabilities`、`confidence` |
| `score` | `["段階 0 の説明", "段階 1 の説明", …]`（低い順、2〜10 段階） | `score`（期待値）、`probabilities`、`confidence` |

### サーバとして使う

TypeSafe の `/v1/systemone` と同じ形のリクエストを受けるサーバを立てます。

```sh
uv run --group train --group serve python -m uratori.serve.app --model tokimoa/uratori-ja-310m --port 8000
```

`--model` には Hub の repo id か、同じ形のローカルのフォルダを指定します。初回はモデルをダウンロードします。既定では CPU で動かし、`--device cuda` か `--device mps` で GPU を使います。`tokimoa/uratori-ja-2b` と `tokimoa/uratori-ja-4b` も同じ方法で指定できますが、GPU が要るので `--device cuda` を付けてください。

```sh
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tokimoa/uratori-ja-310m",
    "state": {
      "根拠": "返品は商品到着後14日以内に限り受け付けます。開封済みの商品は返品できません。",
      "主張": "開封済みの商品でも、到着から7日以内なら返品できる。"
    },
    "questions": {
      "support": {
        "type": "choice",
        "instructions": "`主張` は `根拠` から支持されるか",
        "criteria": {
          "支持": "根拠だけから主張の全体が成り立つ。",
          "矛盾": "根拠と両立しない部分がある。",
          "情報不足": "矛盾はないが、根拠からは成否を決められない部分がある。"
        }
      },
      "contradict": {
        "type": "noul",
        "instructions": "`主張` は `根拠` と矛盾するか",
        "criteria": {"true": "根拠と両立しない部分がある。", "false": "両立しない部分はない。"}
      }
    }
  }'
```

応答は次の形です（確率は丸めて示しています）。

```json
{
  "model": "tokimoa/uratori-ja-310m",
  "answers": {
    "support": {
      "type": "choice",
      "choice": "矛盾",
      "probabilities": {"支持": 0.12, "矛盾": 0.71, "情報不足": 0.17},
      "confidence": 0.57
    },
    "contradict": {"type": "noul", "noul": 0.76}
  },
  "usage": {"input_tokens": 194, "output_tokens": 0}
}
```

リクエストの `model` は必須ですが、サーバは起動時に読み込んだモデルで答えます。入力がモデルの上限（uratori-ja-310m は 1,024 トークン）を超えると、切り詰めずに 422 を返します。ほかに `GET /healthz` と `GET /v1/models` があります。認証はなく、既定では 127.0.0.1 だけで待ち受けます。

自分で学習した checkpoint で立てるときは、`--model` の代わりに `--ckpt <フォルダ>` を指定します。

## 評価

公開しているモデルを uratori-ja-eval で測るには、次を実行します。評価セットは Hub から取ってきます。

```sh
uv run --group train python scripts/eval_hub.py --model tokimoa/uratori-ja-310m --device cpu
```

既定では test と challenge を測り、split ごとに Accuracy などを 1 行の JSON で出力します。`runs/eval_hub/<モデル名>/` には、予測（`pred__<split>.jsonl`）、レポート（`report__<split>.md`）、この repository の形式に直した評価データ（`<split>.jsonl`）を書きます。`--split` で split を、`--ckpt` で自分の checkpoint を指定できます。

自分で学習した checkpoint を手元のデータで測るスクリプトもあります。どれも先頭のコメントに使い方を書いてあります。

| スクリプト | 内容 |
|------------|------|
| `scripts/eval_ckpt.py` | 複数の評価データで測り、予測とレポートを checkpoint のフォルダに書く |
| `scripts/calibrate_ckpt.py` | 校正用のデータで温度を合わせ、校正前後の指標を比べる |
| `scripts/position_probe.py` | 候補の順番を逆にしても答えが変わらないかを調べる |
| `scripts/noul_probe.py` | noul の criteria を外しても答えられるかを調べる |
| `scripts/predict.py` | 予測だけを書き出す。`uv run uratori-eval` で採点できる |

## 学習

学習データは 1 行 1 件の JSONL で、1 件が「state と 1 つの質問、その正解」です。形式の見本が `examples/sample_records.jsonl` にあり、schema は `src/uratori/data/schema.py` です。作ったデータは次のコマンドで検査できます。

```sh
uv run uratori-validate examples/sample_records.jsonl
```

エンコーダを全層学習する例です。

```sh
uv run --group train python scripts/train.py --model sbintuitions/modernbert-ja-310m \
    --train data/train.jsonl --dev data/dev.jsonl --out runs/mbj310 --device cuda
```

デコーダは LoRA で学習します。

```sh
uv run --group train python scripts/train.py --model Qwen/Qwen3.5-2B \
    --train data/train.jsonl --dev data/dev.jsonl --out runs/qwen35_2b \
    --lora 32 --model-dtype bf16 --autocast --device cuda
```

`scripts/` のスクリプトは `--device` の既定が `mps`（Apple silicon）です。ほかの環境では `--device cuda` か `--device cpu` を指定してください。

### 公開したモデルの学習手順

`scripts/train.py` を 2 回に分けて実行しています。

1. **第 1 段階**: JNLI（JGLUE）を `scripts/convert_jnli.py` で判定の形式に直したデータ 19,816 件（支持 / 矛盾 / 中立の選択と、はい / いいえ）。2〜3 epoch、256 トークン
2. **第 2 段階**: 架空の業務文書、官公庁の FAQ、Wikipedia の段落から LLM で生成した合成データ 49,668 件。第 1 段階の重みから `--init-from` で始める。1 epoch、1,024〜1,280 トークン。310m では、noul の 3 割を criteria なしで、choice の 5 割を候補を並べ替えて見せる（`--aug-noul-drop 0.3 --aug-choice-shuffle 0.5`）

細かい設定は各モデルカードにあります。合成の学習データそのものは、この repository に含めていません。作るためのコードは次のとおりです。

| スクリプト | 内容 |
|------------|------|
| `scripts/convert_jnli.py` | JNLI をこの repository の形式に直す |
| `scripts/generate.py` | LLM に原文と問題を書かせて、合成データの候補を作る |
| `scripts/verify_and_build.py` | 候補を別の LLM に判定させ、ラベルが一致した件で学習用のデータを組み立てる |
| `scripts/build_unverified.py` | 検証前の候補から学習用のデータを組み立てる |
| `scripts/judge_with_llm.py` | データを LLM に判定させる（比較用の基準値や、評価データのラベル付けに使う） |

生成と判定に使う LLM の接続先は `configs/endpoints.yaml` に書きます。API キーはファイルには書かず、接続先ごとの環境変数（`DEEPSEEK_API_KEY` など）から読みます。質問文と criteria の定義は `rubrics/` にあります。

学習の結果は実行ごとに変わります。310m で同じ設定の 2 段階をやり直したところ、最終的な test の Accuracy は 0.62〜0.71 の幅になりました。低く出た実行は、JNLI の検証データでの正答率は同じでも、第 1 段階の後の検証 NLL が高くなっていました（0.34。良かった実行は 0.24〜0.29）。学習し直す場合は、第 1 段階を複数回実行し、検証 NLL が最も低い重みから第 2 段階を始めてください。

## 構成

```text
src/uratori/
  types.py, serialize.py   質問の型と、モデル入力への直列化
  models/, train/          候補マーカーを採点するモデル、損失、学習のループ
  eval/                    指標、レポート、温度による校正
  serve/                   HTTP サーバ、ONNX への書き出し
  data/                    データの schema、validator、rubric の読み込み
  gen/, llm/               合成データの生成、LLM のクライアントと判定
scripts/                   学習、評価、データ作成のスクリプト
rubrics/                   質問文と criteria の定義
configs/                   LLM の接続先
examples/                  データ形式の見本
tests/
```

テストと lint は次のコマンドで実行します。

```sh
uv run --group train --group serve --group export pytest -q
uv run ruff check . && uv run ruff format --check .
```

## ライセンス

コードは Apache License 2.0 です。モデルと評価セットのライセンスは、それぞれの Hugging Face のページに書いてあります。

## 引用

```bibtex
@misc{uratori2026,
  title  = {uratori: Japanese decision models for grounding checks, document comparison and RAG judgments},
  author = {tokimoa},
  year   = {2026},
  url    = {https://github.com/tokimoa/uratori}
}
```
