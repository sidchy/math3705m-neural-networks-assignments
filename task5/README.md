# Task 5: Wenzhou Dialect LLM Adaptation

This project implements a reproducible low-resource dialect experiment:

1. Clean Wenzhou dialect Excel data.
2. Build fixed train/val/test splits.
3. Train a character-level GPT model.
4. Fine-tune a small Qwen model with LoRA SFT.
5. Build DPO preference pairs and compare SFT vs DPO.

## Local Setup

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data Pipeline

```bash
python scripts/01_clean_data.py --raw_dir "/Users/sidneychai/Downloads/温州话语料" --out data/cleaned
python scripts/02_make_splits.py --cleaned data/cleaned --out data/splits
python scripts/03_build_pretrain.py --cleaned data/cleaned --splits data/splits --out data/final
python scripts/04_build_sft_data.py --cleaned data/cleaned --splits data/splits --out data/final
python scripts/06_validate_data.py --data_dir data/final
```

## Cloud GPU Run

First run a small GPU smoke test:

```bash
PYTHONPATH=. python sft/train.py --model unsloth/Qwen3.5-0.8B --data data/final/sft_train.jsonl --eval data/final/sft_val.jsonl --out runs/sft_smoke/adapter_model --smoke_steps 20 --batch_size 1 --grad_accum 4
```

Full SFT:

```bash
PYTHONPATH=. python sft/train.py --model unsloth/Qwen3.5-2B --data data/final/sft_train.jsonl --eval data/final/sft_val.jsonl --out runs/sft/adapter_model
PYTHONPATH=. python sft/inference.py --base_model unsloth/Qwen3.5-2B --model_or_adapter runs/sft/adapter_model --test data/final/sft_test.jsonl --out runs/sft/predictions.jsonl
```

DPO:

```bash
PYTHONPATH=. python scripts/05_build_dpo_data.py --sft_train data/final/sft_train.jsonl --out data/final --sample_size 1000
PYTHONPATH=. python dpo/train.py --model_or_adapter runs/sft/adapter_model --data data/final/dpo_train.jsonl --eval data/final/dpo_val.jsonl --out runs/dpo/adapter_model
PYTHONPATH=. python dpo/inference.py --base_model unsloth/Qwen3.5-2B --model_or_adapter runs/dpo/adapter_model --test data/final/sft_test.jsonl --out runs/dpo/predictions.jsonl
```

Fallback: if Qwen3.5-2B fails on T4, replace it with `unsloth/Qwen3.5-0.8B` or `Qwen/Qwen2.5-1.5B-Instruct`.
