# Task 5 GPU Runbook

完整操作流程：Mac 端准备 → 服务器环境 → 数据 → 训练 → 评估 → 报告。

---

## 1. Mac 端：打包数据（可选，跳过则服务器重新生成）

```bash
cd /Users/sidneychai/Documents/神经网络课程作业/math3705m-neural-networks-assignments
tar czf task5-data.tar.gz task5/data/
# scp task5-data.tar.gz <server>:/path/to/math3705m-neural-networks-assignments/task5/
```

服务器解压：

```bash
cd math3705m-neural-networks-assignments/task5
tar xzf task5-data.tar.gz
```

原始 Excel 文件也需要上传到服务器的某个目录，或者重新走数据流水线（见第 3 步）。

---

## 2. 服务器环境

```bash
# Python 3.10
uv venv --python 3.10
source .venv/bin/activate

# 依赖
uv pip install -r requirements.txt

# 验证
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

预期：显示 `True`（有 GPU）。

---

## 3. 数据流水线（如果没有从 Mac 传数据）

把原始语料 Excel 放到服务器的某个目录，假设是 `~/wenzhou-data/`。

```bash
# 清理
PYTHONPATH=. python scripts/01_clean_data.py \
  --raw_dir ~/wenzhou-data \
  --out data/cleaned

# 划分
PYTHONPATH=. python scripts/02_make_splits.py \
  --cleaned data/cleaned \
  --out data/splits

# 预训练语料
PYTHONPATH=. python scripts/03_build_pretrain.py \
  --cleaned data/cleaned \
  --splits data/splits \
  --out data/final

# SFT 数据
PYTHONPATH=. python scripts/04_build_sft_data.py \
  --cleaned data/cleaned \
  --splits data/splits \
  --out data/final

# 校验
PYTHONPATH=. python scripts/06_validate_data.py --data_dir data/final
```

预期输出：

```
{'path': 'data/final/sft_train.jsonl', 'rows': 44574, 'unique_ids': 44574}
{'path': 'data/final/sft_val.jsonl', 'rows': 2965, 'unique_ids': 2965}
{'path': 'data/final/sft_test.jsonl', 'rows': 2865, 'unique_ids': 2865}
```

### 3.1 跑测试

```bash
PYTHONPATH=. pytest -q
```

预期：11 passed。

---

## 4. CPU 预训练 smoke（可选，仅验证代码）

```bash
PYTHONPATH=. python pretrain/train.py --config configs/pretrain_smoke.yaml
```

预期 1 分钟内完成，输出 `epoch 1: train_loss=~8.5 val_loss=~8.4`。

确认产出：

```bash
ls runs/pretrain_smoke/
# checkpoint.pt  metrics.json  vocab.json  config.json
```

---

## 5. GPU 预训练（T4，获得实验一结果）

```bash
PYTHONPATH=. python pretrain/train.py --config configs/pretrain_t4.yaml
```

预期 100 轮、val loss 低于 unigram baseline（5.93）。早停生效意味着通常更少轮。

确认产出：

```bash
ls runs/pretrain/
# checkpoint.pt  metrics.json  vocab.json  config.json
```

---

## 6. GPU SFT smoke（用小模型快速验证）

```bash
PYTHONPATH=. python sft/train.py \
  --model unsloth/Qwen3.5-0.8B \
  --data data/final/sft_train.jsonl \
  --eval data/final/sft_val.jsonl \
  --out runs/sft_smoke/adapter_model \
  --smoke_steps 20 \
  --batch_size 1 \
  --grad_accum 4
```

预期 20 步内完成，3-5 分钟。

---

## 7. Full SFT 训练

```bash
PYTHONPATH=. python sft/train.py \
  --model unsloth/Qwen3.5-2B \
  --data data/final/sft_train.jsonl \
  --eval data/final/sft_val.jsonl \
  --out runs/sft/adapter_model
```

如果 T4 内存不够 Qwen3.5-2B，降级为：

```bash
  --model unsloth/Qwen3.5-0.8B
```

---

## 8. Base 模型推理（对照组）

无 LoRA adapter 直接跑基座模型：

```bash
PYTHONPATH=. python sft/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --test data/final/sft_test.jsonl \
  --out runs/base/predictions.jsonl
```

---

## 9. SFT 推理

```bash
PYTHONPATH=. python sft/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --model_or_adapter runs/sft/adapter_model \
  --test data/final/sft_test.jsonl \
  --out runs/sft/predictions.jsonl
```

---

## 10. 构建 DPO 数据（可复现）

```bash
PYTHONPATH=. python scripts/05_build_dpo_data.py \
  --sft_train data/final/sft_train.jsonl \
  --out data/final \
  --sample_size 1000
```

确认产出：

```bash
ls data/final/dpo_train.jsonl data/final/dpo_val.jsonl data/final/dpo_stats.json
# 预期 train ~900, val ~100
```

---

## 11. DPO 训练

```bash
PYTHONPATH=. python dpo/train.py \
  --model_or_adapter runs/sft/adapter_model \
  --data data/final/dpo_train.jsonl \
  --eval data/final/dpo_val.jsonl \
  --out runs/dpo/adapter_model
```

---

## 12. DPO 推理

```bash
PYTHONPATH=. python dpo/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --model_or_adapter runs/dpo/adapter_model \
  --test data/final/sft_test.jsonl \
  --out runs/dpo/predictions.jsonl
```

---

## 13. 评估：生成指标表和图表

```bash
PYTHONPATH=. python eval/evaluate.py \
  --data data/final \
  --runs runs \
  --out report/figures
```

产出：

| 文件 | 内容 |
|------|------|
| `report/figures/auto_metrics.json` | Base/SFT/DPO 的 char-BLEU + chrF |
| `report/figures/auto_metrics_table.tex` | LaTeX 表格 |
| `report/figures/pretrain_loss.png` | 预训练 loss 曲线图 |

---

## 14. 编译报告

```bash
# 如果服务器有 tectonic
tectonic report/main.tex

# 或者 Overleaf 上传 task5/report/ 目录
```

---

## 15. 产物清单

跑完后确认以下文件存在：

```
data/
├── final/
│   ├── pretrain_train.txt / val.txt / test.txt
│   ├── sft_train.jsonl / val.jsonl / test.jsonl
│   ├── dpo_train.jsonl / val.jsonl
│   └── sft_stats.json / dpo_stats.json
└── splits/
    ├── train_ids.json / val_ids.json / test_ids.json
    └── split_report.json

runs/
├── pretrain/
│   ├── checkpoint.pt
│   ├── metrics.json
│   └── vocab.json
├── base/predictions.jsonl
├── sft/
│   ├── adapter_model/
│   ├── metrics.json
│   └── predictions.jsonl
└── dpo/
    ├── adapter_model/
    ├── metrics.json
    └── predictions.jsonl

report/
├── main.tex
└── figures/
    ├── auto_metrics_table.tex
    ├── auto_metrics.json
    └── pretrain_loss.png
```

---

## 故障排除

| 问题 | 处理 |
|------|------|
| T4 OOM (Qwen3.5-2B) | 换 `unsloth/Qwen3.5-0.8B` 或 `Qwen/Qwen2.5-1.5B-Instruct` |
| `ModuleNotFoundError: task5lib` | 确认设置了 `PYTHONPATH=.` |
| pytest segfault (旧 Python) | 必须在 Python 3.10 venv 中运行 |
| `sacrebleu` 短字符串 BLEU 为 0 | 已内建 `use_effective_order=True` |
| DPO 两次跑结果不同 | 已修复，使用 seeded RNG |
