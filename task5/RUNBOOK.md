# Task 5 GPU Runbook

从零开始的完整操作流程：裸服务器 → 环境 → 数据 → 训练 → 评估 → 报告。

---

## 1. Mac 端：打包数据上传到服务器

### 1.1 打包已产出的 data/（避免服务器重新生成）

```bash
cd /Users/sidneychai/Documents/神经网络课程作业/math3705m-neural-networks-assignments
tar czf task5-data.tar.gz task5/data/
```

### 1.2 打包原始语料 Excel（上传到服务器重新跑数据流水线时需要）

```bash
tar czf wenzhou-raw.tar.gz -C "/Users/sidneychai/Downloads/温州话语料" .
```

### 1.3 上传到服务器

```bash
# 替换 <server-ip> 为实际服务器地址
scp task5-data.tar.gz wenzhou-raw.tar.gz root@<server-ip>:~/
```

如果服务器没有运行 sshd，用别的传输方式（网盘、跳板机 scp 等）。

---

## 2. 服务器：从零搭建环境

SSH 登录服务器后，按顺序执行。

### 2.1 安装基础工具

```bash
# git（一般已有）
git --version || apt-get install -y git

# 如果需要 Python 3.10（一般已有，先检查）
python3.10 --version 2>/dev/null || python3 --version
```

如果没有 Python 3.10：

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y python3.10 python3.10-venv

# 或者用 conda
# conda create -n task5 python=3.10 -y && conda activate task5
```

### 2.2 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 重新加载 shell
source ~/.bashrc  # 或 source ~/.zshrc
uv --version
```

### 2.3 Clone 代码

```bash
git clone https://github.com/sidchy/math3705m-neural-networks-assignments.git
cd math3705m-neural-networks-assignments/task5
```

### 2.4 创建 Python 虚拟环境

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 2.5 安装依赖

```bash
uv pip install -r requirements.txt
```

这一步耗时最长（下载 PyTorch/triton/unsloth 等），预计 5-15 分钟。

### 2.6 验证 CUDA 可用

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果输出 `cuda: False`，检查服务器是否有 GPU 且 CUDA 版本匹配。

---

## 3. 解压数据

### 方案 A：用 Mac 打包的 data/（推荐，已含所有产出）

```bash
# 仍在 task5/ 目录下
tar xzf ~/task5-data.tar.gz
# 验证
ls data/cleaned/ data/splits/ data/final/
```

**跳过第 4 步，直接到第 5 步。**

### 方案 B：服务器重新生成（如果没传 data 打包）

先解压原始语料：

```bash
mkdir -p ~/wenzhou-data
tar xzf ~/wenzhou-raw.tar.gz -C ~/wenzhou-data
ls ~/wenzhou-data/
# 应看到 01_...xlsx, 02_...xlsx, ..., 有误词条.xlsx, 新事物名词.xlsx, 温州地名300个.xlsx
```

然后按第 4 步执行。

---

## 4. 数据流水线（仅方案 B 需要）

```bash
# 确保在 task5/ 目录，虚拟环境已激活

# 清理
PYTHONPATH=. python scripts/01_clean_data.py \
  --raw_dir ~/wenzhou-data \
  --out data/cleaned

# 划分（group-level，防泄漏，seed=42 可复现）
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
```

### 4.1 验证数据

```bash
PYTHONPATH=. python scripts/06_validate_data.py --data_dir data/final
```

预期输出：

```
{'path': 'data/final/sft_train.jsonl', 'rows': 44574, 'unique_ids': 44574}
{'path': 'data/final/sft_val.jsonl', 'rows': 2965, 'unique_ids': 2965}
{'path': 'data/final/sft_test.jsonl', 'rows': 2865, 'unique_ids': 2865}
```

### 4.2 跑单元测试

```bash
PYTHONPATH=. pytest -q
```

预期：`11 passed`。

---

## 5. 实验一：字符级 GPT 预训练

### 5.1 CPU smoke（可选，1 分钟验证代码）

```bash
PYTHONPATH=. python pretrain/train.py --config configs/pretrain_smoke.yaml
```

预期：`epoch 1: train_loss=~8.5 val_loss=~8.4`（比 unigram baseline 5.93 差是正常的，这只是 1 epoch smoke）。

### 5.2 生成样例验证

```bash
PYTHONPATH=. python pretrain/generate.py \
  --checkpoint runs/pretrain_smoke/checkpoint.pt \
  --vocab runs/pretrain_smoke/vocab.json \
  --out runs/pretrain_smoke/samples.json
```

### 5.3 GPU 完整训练（用于实验一结果）

```bash
PYTHONPATH=. python pretrain/train.py --config configs/pretrain_t4.yaml
```

配置：block_size=256, 6 层, 384 维, 100 轮, 早停 patience=10。
预期：val loss 低于 unigram baseline (5.93)，早停在 ~10-50 轮。

确认产出：

```bash
ls runs/pretrain/
# checkpoint.pt  metrics.json  vocab.json  config.json
```

---

## 6. 实验二：SFT 翻译微调

### 6.1 GPU smoke（3-5 分钟，验证 Unsloth+TRL 正常）

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

预期：20 步完成，`runs/sft_smoke/adapter_model/` 有 adapter 权重文件。

### 6.2 Full SFT

```bash
PYTHONPATH=. python sft/train.py \
  --model unsloth/Qwen3.5-2B \
  --data data/final/sft_train.jsonl \
  --eval data/final/sft_val.jsonl \
  --out runs/sft/adapter_model
```

T4 15GB 内存一般够。如果 OOM，降到 0.8B：

```bash
  --model unsloth/Qwen3.5-0.8B
```

---

## 7. 推理：Base / SFT / DPO（三路对比）

DPO 训练还没跑，先跑 Base + SFT；DPO 推理在第 10 步。

### 7.1 Base 模型推理（无微调对照组）

```bash
PYTHONPATH=. python sft/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --test data/final/sft_test.jsonl \
  --out runs/base/predictions.jsonl
```

### 7.2 SFT 模型推理

```bash
PYTHONPATH=. python sft/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --model_or_adapter runs/sft/adapter_model \
  --test data/final/sft_test.jsonl \
  --out runs/sft/predictions.jsonl
```

---

## 8. 构建 DPO 偏好数据

```bash
PYTHONPATH=. python scripts/05_build_dpo_data.py \
  --sft_train data/final/sft_train.jsonl \
  --out data/final \
  --sample_size 1000
```

确认产出（seed=42 固定，每次产出相同）：

```bash
ls -la data/final/dpo_train.jsonl data/final/dpo_val.jsonl
# train ~900 条, val ~100 条
```

---

## 9. 实验三：DPO 偏好优化

### 9.1 DPO 训练

```bash
PYTHONPATH=. python dpo/train.py \
  --model_or_adapter runs/sft/adapter_model \
  --data data/final/dpo_train.jsonl \
  --eval data/final/dpo_val.jsonl \
  --out runs/dpo/adapter_model
```

### 9.2 DPO 推理

```bash
PYTHONPATH=. python dpo/inference.py \
  --base_model unsloth/Qwen3.5-2B \
  --model_or_adapter runs/dpo/adapter_model \
  --test data/final/sft_test.jsonl \
  --out runs/dpo/predictions.jsonl
```

---

## 10. 评估：三路对比 + 图表

```bash
PYTHONPATH=. python eval/evaluate.py \
  --data data/final \
  --runs runs \
  --out report/figures
```

产出：

| 文件 | 内容 |
|------|------|
| `report/figures/auto_metrics.json` | Base / SFT / DPO 的 char-BLEU + chrF |
| `report/figures/auto_metrics_table.tex` | LaTeX 三行对比表 |
| `report/figures/pretrain_loss.png` | 预训练 loss 曲线 |

查看结果：

```bash
cat report/figures/auto_metrics.json
```

---

## 11. 编译报告

### 选项 A：服务器编译

```bash
# 安装 tectonic（如果没有）
curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-$(uname -m)-unknown-linux-gnu.tar.gz | tar xz -C /usr/local/bin
tectonic report/main.tex
```

### 选项 B：Overleaf

把 `task5/report/` 目录打包上传：

```bash
tar czf report.tar.gz -C report main.tex figures/
# 下载到本地，上传 Overleaf
```

---

## 12. 下载结果到 Mac

```bash
# 在 Mac 端运行
scp -r root@<server-ip>:~/math3705m-neural-networks-assignments/task5/runs ~/Downloads/task5-runs/
scp -r root@<server-ip>:~/math3705m-neural-networks-assignments/task5/report ~/Downloads/task5-report/
```

---

## 13. 产物清单（全部跑完后确认）

```
task5/
├── data/
│   └── final/
│       ├── dpo_train.jsonl / dpo_val.jsonl        ← DPO 数据
│       ├── pretrain_train.txt / val.txt / test.txt   ← 预训练语料
│       ├── sft_train.jsonl / val.jsonl / test.jsonl  ← SFT 数据
│       └── sft_stats.json / dpo_stats.json / pretrain_stats.json
│
├── runs/
│   ├── pretrain/
│   │   ├── checkpoint.pt       ← 字符 GPT 权重
│   │   ├── metrics.json        ← 训练曲线数据
│   │   └── vocab.json          ← 字符词表
│   ├── base/
│   │   └── predictions.jsonl   ← 基座推理结果
│   ├── sft/
│   │   ├── adapter_model/      ← LoRA 权重
│   │   ├── metrics.json        ← SFT 训练指标
│   │   └── predictions.jsonl   ← SFT 推理结果
│   └── dpo/
│       ├── adapter_model/      ← DPO LoRA 权重
│       ├── metrics.json        ← DPO 训练指标
│       └── predictions.jsonl   ← DPO 推理结果
│
└── report/
    ├── main.tex                ← 报告源文件
    ├── main.pdf                ← 编译后 PDF
    └── figures/
        ├── auto_metrics.json      ← Base/SFT/DPO 自动指标
        ├── auto_metrics_table.tex ← LaTeX 指标表
        └── pretrain_loss.png      ← 预训练 loss 曲线
```

---

## 常见问题

| 症状 | 原因 | 处理 |
|------|------|------|
| T4 OOM | Qwen3.5-2B 超出 15GB | 换 `unsloth/Qwen3.5-0.8B` 或 `Qwen/Qwen2.5-1.5B-Instruct` |
| `RuntimeError: CUDA out of memory` | batch size 太大 | 减小 `--batch_size 1 --grad_accum 4` |
| `ModuleNotFoundError: task5lib` | PYTHONPATH 没设 | 所有 Python 命令前加 `PYTHONPATH=.` |
| pytest 在 Python 3.13 崩溃 | Python 版本不兼容 | 必须用 3.10 venv |
| `tokenizers` 安装失败 | 需要 Rust 编译器 | `apt-get install -y cargo` 或 `pip install tokenizers --no-build-isolation` |
| `unsloth` 安装失败 / triton 冲突 | CUDA 版本问题 | 检查 CUDA 版本 `nvcc --version`，参考 unsloth 官方安装指南 |
| SSH 连不上服务器 | 安全组/防火墙 | 检查云平台安全组是否放行 22 端口 |
| DPO 两次跑结果不同 | 旧代码 bug | 已修复（#2a90f8e），拉最新代码 |
| pretrain train_loss > baseline | smoke 仅 1 epoch 正常 | 跑完整 GPU config 后应低于 baseline |
