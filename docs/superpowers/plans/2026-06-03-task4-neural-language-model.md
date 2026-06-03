# Task 4 Neural Language Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete `task4` neural language modeling assignment for the provided `九章算经 2.txt`, then commit and push it to GitHub for cloud-server training.

**Architecture:** Create a standalone task4 project with preprocessing, a decoder-only Transformer language model, a PyTorch FastText-style embedding model, asset generation, and a LaTeX report. Local work should avoid installing dependencies; only run lightweight checks that use the standard library unless the environment already has the needed packages.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, NumPy 1.26.4, Matplotlib 3.8.4, scikit-learn 1.4.2, tqdm 4.66.4, uv, LaTeX/ctexart.

---

## Current Repo Facts

- Repo root: `/Users/sidneychai/Documents/神经网络课程作业/math3705m-neural-networks-assignments`
- Branch: `main`
- Remote: `origin git@github.com:sidchy/math3705m-neural-networks-assignments.git`
- Source file: `task4/九章算经 2.txt`
- Source encoding: `gb18030`; UTF-8 decoding fails.
- Answer marker in the corpus: `荅曰`; code must also support `答曰`.
- Approximate corpus stats from inspection:
  - 72,051 decoded characters
  - 2,043 lines
  - 506 `荅曰` occurrences
  - 480 `今有` occurrences
  - about 488 extractable problem-answer blocks
- Existing dirty files must not be modified, reverted, staged, or committed:
  - `task3/report/main.tex`
  - `task3/report/课程报告3_柴昊阳.pdf`

## File Structure To Create

- `task4/requirements.txt`: uv-compatible pinned dependencies.
- `task4/README.md`: cloud-server setup and run instructions.
- `task4/src/data.py`: corpus decoding, normalization, QA extraction, vocab, dataset creation.
- `task4/src/transformer_lm.py`: decoder-only Transformer model.
- `task4/src/train_lm.py`: language-model training, evaluation, checkpointing, generation.
- `task4/src/fasttext_embed.py`: FastText-style subword skip-gram training and nearest-neighbor export.
- `task4/src/evaluate.py`: reusable metric and generation helpers if needed by training/report scripts.
- `task4/src/generate_report_assets.py`: produce figures/tables for the report from run outputs.
- `task4/report/main.tex`: Chinese course report matching task2/task3 two-column `ctexart` style.
- `task4/report/figures/.gitkeep`: keep figures folder in git.
- `task4/runs/.gitkeep`: keep runs folder in git without committing heavy outputs.
- `task4/.gitignore`: ignore checkpoints, generated run outputs, LaTeX build artifacts, and caches.

## Dependency Requirements

- [ ] Create `task4/requirements.txt` with conservative pins:

```txt
torch==2.3.1
numpy==1.26.4
matplotlib==3.8.4
scikit-learn==1.4.2
tqdm==4.66.4
```

- [ ] Do not install dependencies locally unless the user explicitly asks.
- [ ] Document cloud setup with uv:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Implementation Tasks

### Task 1: Data Pipeline

**Files:**
- Create: `task4/src/data.py`

- [ ] Implement `read_corpus(path: str | Path) -> str`.
  - Read bytes from `path`.
  - Decode with `gb18030`.
  - Normalize `\r\n` and `\r` to `\n`.
  - Preserve traditional Chinese characters and punctuation.

- [ ] Implement `normalize_text(text: str) -> str`.
  - Replace runs of spaces/tabs with a single full-width-compatible normal space only inside lines.
  - Collapse more than two blank lines to two blank lines.
  - Do not convert traditional characters to simplified.

- [ ] Implement `extract_qa_blocks(text: str) -> list[dict]`.
  - Split on bracketed problem markers matching `〔...〕`.
  - Keep blocks containing a question phrase such as `今有` or `又有`.
  - Support answer markers `荅曰：`, `荅曰:`, `答曰：`, `答曰:`.
  - Return dictionaries with `id`, `question`, `answer`, and `raw`.
  - Expected extraction count should be near 488.

- [ ] Implement `CharVocab`.
  - Special tokens in this exact order: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
  - Provide `encode`, `decode`, `save`, and `load`.

- [ ] Implement character language-model dataset helpers.
  - Build one long stream: `<bos>` + corpus chars + `<eos>`.
  - Create fixed-length next-token examples with `seq_len`.
  - Split train/val/test using seed `42`, default ratios `0.9/0.05/0.05`.

- [ ] Add a CLI self-check:

```bash
python src/data.py --data "九章算经 2.txt"
```

Expected output should include decoded character count, vocabulary size, and extracted QA count.

### Task 2: Transformer Language Model

**Files:**
- Create: `task4/src/transformer_lm.py`
- Create: `task4/src/train_lm.py`

- [ ] Implement a decoder-only Transformer in `transformer_lm.py`.
  - Token embedding + learned positional embedding.
  - Causal self-attention through `torch.nn.TransformerEncoder` with a causal mask.
  - Output projection to vocabulary size.
  - Config fields: `vocab_size`, `d_model`, `n_layers`, `n_heads`, `ffn_dim`, `seq_len`, `dropout`.

- [ ] Implement training presets in `train_lm.py`.
  - `gpu`: `d_model=256`, `n_layers=4`, `n_heads=4`, `ffn_dim=1024`, `seq_len=128`, `dropout=0.1`, `batch_size=64`, `epochs=30`, `lr=3e-4`.
  - `smoke`: `d_model=64`, `n_layers=1`, `n_heads=2`, `ffn_dim=128`, `seq_len=64`, `dropout=0.1`, `batch_size=8`, `epochs=1`, `lr=5e-4`.

- [ ] Implement training behavior.
  - Use `AdamW`.
  - Use cross-entropy loss ignoring `<pad>`.
  - Track train loss, val loss, and val perplexity.
  - Save `checkpoint.pt`, `vocab.json`, `config.json`, `metrics.json`, and `samples.txt`.
  - Use deterministic seed `42`.
  - Automatically choose CUDA if available, otherwise CPU.

- [ ] Implement generation.
  - Prompts: `今有田廣`, `荅曰`, `方田術曰`, `句股`.
  - Support `temperature`, `top_k`, and `max_new_tokens`.
  - Save generated examples into `samples.txt` and `samples.json`.

- [ ] Document commands:

```bash
python src/train_lm.py --data "九章算经 2.txt" --preset smoke --out runs/smoke
python src/train_lm.py --data "九章算经 2.txt" --preset gpu --out runs/transformer
```

### Task 3: FastText-Style Embedding Model

**Files:**
- Create: `task4/src/fasttext_embed.py`

- [ ] Implement token extraction.
  - Use character-level tokens.
  - Add subword features as character unigram, bigram, and trigram strings.
  - Include selected multi-character terms directly if they occur in the corpus:
    `方田`, `粟米`, `少廣`, `方程`, `句股`, `畝`, `步`, `分`, `實`, `法`.

- [ ] Implement skip-gram with negative sampling in PyTorch.
  - Embedding dim: `100`.
  - Context window: `4`.
  - Negative samples: `5`.
  - Epochs: `20`.
  - Batch size: `512`.
  - Seed: `42`.

- [ ] Save embedding outputs.
  - `embeddings.pt`
  - `token_to_id.json`
  - `nearest_neighbors.json`
  - `nearest_neighbors.tex`

- [ ] Compute nearest neighbors for:
  - `方田`, `粟米`, `少廣`, `方程`, `句股`, `畝`, `步`, `分`, `實`, `法`

- [ ] Document command:

```bash
python src/fasttext_embed.py --data "九章算经 2.txt" --out runs/fasttext
```

### Task 4: Report Asset Generation

**Files:**
- Create: `task4/src/generate_report_assets.py`
- Create: `task4/report/figures/.gitkeep`

- [ ] Load `runs/transformer/metrics.json` and generate `report/figures/lm_loss.png`.
- [ ] Load `runs/fasttext/nearest_neighbors.json` and write a LaTeX table fragment if useful.
- [ ] Load embeddings and selected terms, then generate a PCA scatter plot at `report/figures/embedding_pca.png`.
- [ ] Generate `report/figures/corpus_stats.tex` with corpus statistics.
- [ ] Fail with a clear message if expected run outputs are missing.

Document command:

```bash
python src/generate_report_assets.py --lm runs/transformer --embed runs/fasttext --out report/figures
```

### Task 5: Report

**Files:**
- Create: `task4/report/main.tex`

- [ ] Use the same compact two-column `ctexart` style as task2/task3.
- [ ] Title: `课程报告四：基于Transformer与FastText式嵌入的《九章算术》语言建模实验`
- [ ] Author: `柴昊阳 22542013`
- [ ] Include these sections:
  - `研究目标`
  - `数据集与预处理`
  - `Transformer序列生成模型`
  - `FastText式文本嵌入模型`
  - `训练设置`
  - `结果展示与分析`
  - `结论`
  - `参考文献`
- [ ] Include placeholders that are safe before cloud training.
  - Use `\IfFileExists` wrappers for figures.
  - Mention that final numeric results come from `runs/transformer/metrics.json` and `runs/fasttext/nearest_neighbors.json`.
- [ ] Include generated text examples and embedding neighbor table once assets exist.
- [ ] Keep wording in Chinese and directly address the assignment requirement: whether the model learned knowledge/patterns from 《九章算术》.

### Task 6: README and Ignore Rules

**Files:**
- Create: `task4/README.md`
- Create: `task4/.gitignore`
- Create: `task4/runs/.gitkeep`

- [ ] README must include:
  - project purpose
  - dependency installation with uv
  - data decoding note: `gb18030`
  - smoke command
  - cloud GPU training command
  - embedding training command
  - report asset command
  - report compile command
  - expected outputs

- [ ] `.gitignore` must ignore:

```gitignore
__pycache__/
*.pyc
.DS_Store
runs/*
!runs/.gitkeep
report/*.aux
report/*.log
report/*.out
report/*.toc
report/*.pdf
report/figures/*
!report/figures/.gitkeep
```

### Task 7: Local Verification Without Installing Dependencies

- [ ] Do not run `uv pip install`, `pip install`, or any dependency install command locally.
- [ ] Run only standard-library checks first:

```bash
python3 src/data.py --data "九章算经 2.txt"
```

Expected:
- decoded characters around `72051`
- QA blocks around `488`
- vocabulary size around `1000`

- [ ] If PyTorch is already available, optionally run smoke training:

```bash
python src/train_lm.py --data "九章算经 2.txt" --preset smoke --out runs/smoke
```

- [ ] If PyTorch is not available, skip smoke training and document that local dependency installation was intentionally not performed.

### Task 8: Git Commit and Push

- [ ] Check status:

```bash
git status --short
```

- [ ] Ensure task3 dirty files remain unstaged.
- [ ] Stage only the task4 project and this plan:

```bash
git add task4 docs/superpowers/plans/2026-06-03-task4-neural-language-model.md
```

- [ ] Confirm staged files:

```bash
git diff --cached --name-only
```

Expected staged paths must not include `task3/`.

- [ ] Commit:

```bash
git commit -m "feat: add task4 neural language modeling assignment"
```

- [ ] Push:

```bash
git push origin main
```

## Cloud Server Runbook

After pulling on the cloud server:

```bash
cd math3705m-neural-networks-assignments/task4
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
python src/train_lm.py --data "九章算经 2.txt" --preset gpu --out runs/transformer
python src/fasttext_embed.py --data "九章算经 2.txt" --out runs/fasttext
python src/generate_report_assets.py --lm runs/transformer --embed runs/fasttext --out report/figures
tectonic report/main.tex
```

## Acceptance Criteria

- `task4/README.md` lets the user run the project on a cloud server with uv.
- `task4/requirements.txt` uses conservative pinned dependency versions.
- Data parsing handles the actual `gb18030` corpus and `荅曰` marker.
- Transformer training code can produce checkpoint, metrics, and samples.
- FastText-style embedding code can produce nearest-neighbor outputs and PCA-ready embeddings.
- Report compiles once generated assets exist.
- Git commit excludes existing task3 dirty changes.
- No local dependency installation is performed during implementation unless the user later explicitly requests it.
