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
