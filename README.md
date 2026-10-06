# uratori

[日本語版 README](README_ja.md)

uratori is a family of Japanese *decision models*. Given a text state (for example a source document and a claim) and typed questions, a model returns a probability distribution over the allowed answers in one forward pass, without generating text. Three question types are supported: **noul** (yes/no), **choice** (one of 2 to 8 labelled options) and **score** (an ordinal scale).

The models are trained for three jobs:

- **Grounding**: is this claim or answer supported by the supplied text?
- **Document comparison**: do these two texts contradict each other, or did an edit change the meaning?
- **RAG judgments**: is this passage relevant to the question, is it sufficient, is the answer faithful to it?

The request and response format follows TypeSafe's `/v1/systemone` API, so the same question definitions can be sent to either. uratori is an independent implementation inspired by TypeSafe's Jev and is not affiliated with TypeSafe.

This repository contains the training, evaluation and serving code. The models and the evaluation set are on Hugging Face.

## Models and evaluation set

| | Base | Size | Runs on | test (802) | challenge (300) |
|---|---|---|---|---|---|
| [tokimoa/uratori-ja-4b](https://huggingface.co/tokimoa/uratori-ja-4b) | Qwen3.5-4B | 8.4 GB | GPU, about 12 GB | 0.867 | 0.863 |
| [tokimoa/uratori-ja-2b](https://huggingface.co/tokimoa/uratori-ja-2b) | Qwen3.5-2B | 3.8 GB | GPU, about 6 GB | 0.766 | 0.800 |
| [tokimoa/uratori-ja-310m](https://huggingface.co/tokimoa/uratori-ja-310m) v0.2 | ModernBERT-Ja-310M | 1.3 GB | CPU | 0.686 | 0.693 |
| [tokimoa/uratori-ja-eval](https://huggingface.co/datasets/tokimoa/uratori-ja-eval) | | 2,002 items | | | |

Accuracy on uratori-ja-eval, measured with the published weights. For reference, TypeSafe's Jev 1.13.0 scores 0.903 and 0.900 on the same splits. Labels in the evaluation set are LLM majority votes, not human annotations; see the dataset card for how it was built and what it does not measure.

## Install

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/tokimoa/uratori.git
cd uratori
uv sync --group train --group serve
```

Dependency groups: the base install has the data schema, metrics and LLM client; `train` adds torch and transformers (needed for inference too); `serve` adds the HTTP server; `export` adds ONNX export; `mlx` adds a local MLX server used for synthetic data generation on Apple silicon.

To use the published models you do not need this repository at all: `pip install torch transformers` is enough, as shown next.

## Quickstart

### Python

The published models carry their own code, so `trust_remote_code=True` is required.

```python
from transformers import AutoModel, AutoTokenizer

repo = "tokimoa/uratori-ja-310m"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()  # add .to("cuda") for the 2b and 4b models

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

`state` is a string or a dict of named texts; dict keys can be referenced from a question as `` `key` ``. Questions in one call share the state and are judged independently. Probabilities are temperature-scaled by default (`calibrate=False` for the raw softmax). Inputs longer than the model's limit (1,024 tokens for 310m, 1,280 for 2b and 4b) raise a `ValueError` instead of being truncated.

Question types:

| `type` | `criteria` | Returns |
|---|---|---|
| `noul` | `{"true": "...", "false": "..."}`, optional but recommended | `noul`: probability of yes |
| `choice` | `{label: description}`, 2 to 8 options | `choice`, `probabilities`, `confidence` |
| `score` | list of level descriptions, lowest first, 2 to 10 levels | `score` (expected level), `probabilities`, `legend`, `confidence` |

### HTTP server

The server accepts the same request shape as TypeSafe's `/v1/systemone`.

```sh
uv run --group train --group serve python -m uratori.serve.app --model tokimoa/uratori-ja-310m --port 8000
```

`--model` takes a Hub repo id or a local folder in the same format; the model is downloaded on first use. The default device is `cpu`; pass `--device cuda` (or `mps`) for the 2b and 4b models. To serve a checkpoint you trained yourself, pass `--ckpt <folder>` instead of `--model`.

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

Response (probabilities rounded here):

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

The `model` field is required by the schema but the server always answers with the model it loaded. Inputs over the model's token limit get a 422 instead of being truncated. `GET /healthz` and `GET /v1/models` are also available. There is no authentication and the server binds to 127.0.0.1 by default.

## Evaluate on uratori-ja-eval

```sh
uv run --group train python scripts/eval_hub.py --model tokimoa/uratori-ja-310m --device cpu
```

This downloads the evaluation set from the Hub, runs the `test` and `challenge` splits by default, prints one JSON line per split (accuracy, macro F1, ECE, bootstrap confidence intervals by document family) and writes predictions and Markdown reports under `runs/eval_hub/<model>/`. Use `--split` to choose splits and `--device cuda` for the larger models.

Scripts for checkpoints you trained yourself (usage is in each file's docstring):

| Script | Purpose |
|---|---|
| `scripts/eval_ckpt.py` | Evaluate a checkpoint on one or more data files; writes predictions and reports next to the checkpoint |
| `scripts/calibrate_ckpt.py` | Fit temperatures on a calibration file and compare metrics before and after |
| `scripts/position_probe.py` | Check whether answers change when option order is reversed |
| `scripts/noul_probe.py` | Check whether noul questions still work without `criteria` |
| `scripts/predict.py` | Write predictions only; score them with `uv run uratori-eval` |

## Train

Training data is JSONL, one record per line: a state, one question and its answer. `examples/sample_records.jsonl` shows the format and `src/uratori/data/schema.py` defines it. Validate a file with:

```sh
uv run uratori-validate examples/sample_records.jsonl
```

Full fine-tuning of an encoder:

```sh
uv run --group train python scripts/train.py --model sbintuitions/modernbert-ja-310m \
    --train data/train.jsonl --dev data/dev.jsonl --out runs/mbj310 --device cuda
```

LoRA on a decoder:

```sh
uv run --group train python scripts/train.py --model Qwen/Qwen3.5-2B \
    --train data/train.jsonl --dev data/dev.jsonl --out runs/qwen35_2b \
    --lora 32 --model-dtype bf16 --autocast --device cuda
```

Scripts default to `--device mps` (Apple silicon); pass `--device cuda` or `--device cpu` elsewhere.

### How the published models were trained

Two stages, each a run of `scripts/train.py`:

1. **Stage 1**: JNLI (JGLUE) converted to the decision format with `scripts/convert_jnli.py` (19,816 items; support/contradict/neutral choice questions and yes/no questions). 2 to 3 epochs, 256 tokens.
2. **Stage 2**: 49,668 synthetic items generated with an LLM from fictional business documents, government FAQs and Wikipedia paragraphs, started from the stage-1 weights with `--init-from`. 1 epoch, 1,024 to 1,280 tokens. For the 310m model, 30% of noul items are shown without criteria and 50% of choice items with shuffled options (`--aug-noul-drop 0.3 --aug-choice-shuffle 0.5`).

Exact hyperparameters are in each model card. The synthetic training data is not included in this repository; the code that produced it is:

| Script | Purpose |
|---|---|
| `scripts/convert_jnli.py` | Convert JNLI into this repository's format |
| `scripts/generate.py` | Have an LLM write documents and questions; produces candidate items |
| `scripts/verify_and_build.py` | Have a second LLM judge the candidates and keep the ones whose labels agree |
| `scripts/build_unverified.py` | Build training files from unverified candidates |
| `scripts/judge_with_llm.py` | Label data with an LLM (used for baselines and for evaluation labels) |

LLM endpoints are configured in `configs/endpoints.yaml`; API keys are read from environment variables (`DEEPSEEK_API_KEY` and so on), never from the file. Question templates and their criteria are in `rubrics/`.

Results vary between runs. For the 310m model, repeating both stages with the same settings gave final test accuracies between 0.62 and 0.71; the low runs had a visibly higher validation NLL after stage 1 (0.34 against 0.24 to 0.29) even though their accuracy on the JNLI validation set was the same. If you retrain, run stage 1 more than once and keep the checkpoint with the lowest validation NLL.

## Repository layout

```text
src/uratori/
  types.py, serialize.py   question types and serialisation into model input
  models/, train/          option-marker scorer, losses, training loop
  eval/                    metrics, reports, temperature calibration
  serve/                   HTTP server, ONNX export
  data/                    schema, validator, rubric loading, public-set conversion
  gen/, llm/               synthetic data generation, LLM client and judge
scripts/                   training, evaluation and data scripts
rubrics/                   question templates and criteria
configs/                   LLM endpoints
examples/                  sample data
tests/
```

Tests and lint:

```sh
uv run --group train --group serve --group export pytest -q
uv run ruff check . && uv run ruff format --check .
```

## License

Apache License 2.0. See each model card for the license of its base model.

## Citation

```bibtex
@misc{uratori2026,
  title  = {uratori: Japanese decision models for grounding checks, document comparison and RAG judgments},
  author = {tokimoa},
  year   = {2026},
  url    = {https://github.com/tokimoa/uratori}
}
```
