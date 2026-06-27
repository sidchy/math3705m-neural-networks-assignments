# 温州话低资源大语言模型适配实验

本项目是 MATH3705M 神经网络课程作业五，围绕**温州话**这一低资源汉语方言，构建了从字符级语言建模、指令微调到偏好优化的完整实验流程。

## 项目背景

温州话在词汇、语法和书写习惯上均与普通话存在显著差异。现有大语言模型虽然具备较强中文能力，但对温州话这类低资源方言的翻译与解释常出现过度解释、幻觉或输出风格不受控等问题。本实验的目标不是训练一个可部署的通用温州话大模型，而是验证一个课程规模内可复现的低资源方言适配流程。

具体关注三个研究问题：

1. 小规模温州话文本是否足以支撑字符级语言模型学习其表层分布；
2. LoRA SFT 是否能显著改善基座模型的温州话到普通话翻译；
3. 基于人工翻译与偏好对构造的 DPO 是否能在 SFT 基础上进一步改善输出质量。

## 数据集

原始语料来自词典、词语考释、对话、论文语料、短视频字幕、新事物名词和温州地名等 13 个 Excel 文件。清洗阶段统一列名、删除空列、应用 231 条勘误表修正、去除空翻译对，并按任务相关字段去重。最终保留 34,842 行多来源数据。

为防止数据泄漏，划分在 SFT 样本构造之前完成，词条类样本按全局词条分组，确保同一词条的变体不会跨 train/val/test 出现。

## 实验方法

### 实验一：字符级 GPT 预训练

采用小型 decoder-only Transformer，以字符为 token 对温州话文本进行语言建模。字符级建模避免了分词器对方言异体字和罕见字处理不稳定的问题。模型配置为 6 层、6 个 attention head、embedding 维度 384，训练目标为标准 next-character prediction。

### 实验二：LoRA SFT 指令微调

使用 Qwen3.5-2B 作为基座模型，训练 LoRA adapter。SFT 数据涵盖温州话到普通话翻译、普通话到温州话反向翻译、词语解释和例句翻译四种任务。训练与推理均使用统一的指令模板，避免 prompt 格式不一致。

### 实验三：DPO 偏好优化

在 SFT adapter 基础上进行 Direct Preference Optimization。chosen 为人工普通话翻译，rejected 由规则扰动构造（反义替换、删减内容、删除字符、重复字符等）。

工程上发现 Unsloth patched Qwen3.5 backward 在 T4 GPU 上对 DPO 反传产生非有限 LoRA 梯度。排查后确定非有限值出现在 backward 路径上（reference log-prob、forward 和 loss 本身均正常）。最终 DPO 使用 native Transformers + PEFT 完成，SFT 阶段仍使用 Unsloth。

## 主要结果

| 指标 | Base | SFT | DPO |
|------|------|-----|-----|
| char-BLEU | 4.33 | 27.36 | 31.38 |
| chrF | 12.08 | 26.08 | 28.10 |

- **字符级预训练**：validation loss 从 unigram baseline 5.931 降至最低 4.157（改善 29.9%），之后快速过拟合，说明小语料上模型主要学习表层分布。
- **SFT**：Base 模型平均输出 128.4 字符，且 848/1076 条含 thinking 标记泄漏；SFT 后平均输出长度降至 17.5 字符，格式稳定、基本消除了模板泄漏。
- **DPO**：平均指标进一步提升，但对俗语、双关或专名仍可能产生语义偏离。DPO 结果应理解为平均稳定性改善，而非所有样例上语义准确率的提高。

## 结论与局限

课程规模下最稳健的路线是：严格数据清洗和防泄漏划分 → SFT 建立基本翻译能力 → 偏好优化作为探索性增强。

主要局限：
- 温州话文本书写缺少统一标准，自动指标会低估可接受译文；
- DPO 偏好对规模小（450 条训练），rejected 由规则扰动生成，偏好信号不如真实人工偏好丰富；
- BLEU 和 chrF 只能衡量字符重合，不能充分评价方言俗语和文化词汇；
- 预训练语料规模有限，字符级模型不能期待生成高质量连贯文本。

## 快速开始

```bash
# 环境
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# 数据流水线
python scripts/01_clean_data.py --raw_dir "<原始Excel目录>" --out data/cleaned
python scripts/02_make_splits.py --cleaned data/cleaned --out data/splits
python scripts/03_build_pretrain.py --cleaned data/cleaned --splits data/splits --out data/final
python scripts/04_build_sft_data.py --cleaned data/cleaned --splits data/splits --out data/final
python scripts/06_validate_data.py --data_dir data/final

# SFT 训练
PYTHONPATH=. python sft/train.py --model unsloth/Qwen3.5-2B --data data/final/sft_train.jsonl --eval data/final/sft_val.jsonl --out runs/sft/adapter_model

# DPO 训练
PYTHONPATH=. python scripts/05_build_dpo_data.py --sft_train data/final/sft_train.jsonl --out data/final --sample_size 1000
PYTHONPATH=. python dpo/train.py --model_or_adapter runs/sft/adapter_model --data data/final/dpo_train.jsonl --eval data/final/dpo_val.jsonl --out runs/dpo/adapter_model
```

详细的 GPU 环境搭建、训练参数和故障排除请参考 [RUNBOOK.md](RUNBOOK.md)。

## 项目结构

```
task5/
├── configs/           # 训练配置文件
├── scripts/           # 数据流水线脚本（清洗、划分、构造、验证）
├── pretrain/          # 字符级 GPT 模型定义与训练
├── sft/               # LoRA SFT 训练与推理
├── dpo/               # DPO 偏好训练与推理
├── eval/              # 自动评估与图表生成
├── report/            # LaTeX 课程报告
├── task5lib/          # 共享工具库
└── tests/             # 单元测试
```

## 参考

- Vaswani et al. Attention Is All You Need. NeurIPS, 2017.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.
- Rafailov et al. Direct Preference Optimization. NeurIPS, 2023.
- Qwen Team. Qwen technical reports and model cards, 2025--2026.
- Papineni et al. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL, 2002.
