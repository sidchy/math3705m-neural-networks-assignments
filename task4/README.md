# Task 4: 《九章算经》Neural Language Modeling

基于 Transformer 和 FastText 式嵌入对《九章算术》古汉语数学文本进行语言建模和语义表示学习。

## 项目结构

```
task4/
├── 九章算经 2.txt          # 原始语料 (GB18030 编码)
├── requirements.txt        # Python 依赖
├── README.md
├── .gitignore
├── src/
│   ├── data.py             # 数据预处理、词表构建、数据集
│   ├── transformer_lm.py   # Decoder-only Transformer 模型
│   ├── train_lm.py         # 语言模型训练与生成
│   ├── answer_probe.py     # 答案判别知识探针
│   ├── fasttext_embed.py   # FastText 式 skip-gram 嵌入训练
│   ├── evaluate.py         # 评估与生成辅助函数
│   └── generate_report_assets.py  # 报告图表生成
├── report/
│   ├── main.tex            # LaTeX 课程报告
│   └── figures/            # 生成的图表
└── runs/                   # 训练输出 (被 .gitignore 忽略)
```

## 环境搭建

```bash
cd task4
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 数据解码说明

原始语料 `九章算经 2.txt` 使用 **GB18030** 编码，Python 脚本已内置自动解码。

```bash
# 数据自检
python src/data.py --data "九章算经 2.txt"
```

预期输出：约 70,000 解码字符、约 1,048 词表大小、488 个 QA 块。

## 训练命令

### Transformer 语言模型

快速冒烟测试：
```bash
python src/train_lm.py --data "九章算经 2.txt" --preset smoke --out runs/smoke
```

GPU 完整训练：
```bash
python src/train_lm.py --data "九章算经 2.txt" --preset gpu --out runs/transformer
```

输出文件：
- `checkpoint.pt` — 模型权重
- `vocab.json` — 字符词表
- `config.json` — 训练配置
- `metrics.json` — epoch 级训练指标
- `samples.txt` / `samples.json` — 生成样例

### 答案判别知识探针

训练完 Transformer 后运行：
```bash
python src/answer_probe.py --data "九章算经 2.txt" --lm runs/transformer --out runs/probe
```

输出文件：
- `answer_probe.json` — 真实答案 vs. 随机错误答案的条件 loss、判别准确率和样例

该探针用于检验模型是否对“题干 + 荅曰”条件下的真实答案赋予更高概率，比自由生成更直接地体现模型是否学到了题目-答案关系。

### FastText 嵌入训练

```bash
python src/fasttext_embed.py --data "九章算经 2.txt" --out runs/fasttext
```

输出文件：
- `embeddings.pt` — 嵌入矩阵
- `token_to_id.json` — token 映射
- `nearest_neighbors.json` — 最近邻结果
- `nearest_neighbors.tex` — LaTeX 表格

### 报告图表生成

```bash
python src/generate_report_assets.py --lm runs/transformer --embed runs/fasttext --probe runs/probe --out report/figures
```

### 编译报告

```bash
# 需要 LaTeX 环境 (tectonic 或 xelatex)
tectonic report/main.tex
```

## 预期输出

| 指标 | 值 |
|------|-----|
| 解码字符数 | ~70,009 |
| QA 块数 | 488 |
| 词表大小 | ~1,048 |
| Transformer 参数 | ~3.73M |
| 嵌入维度 | 100 |

## 模型权重与结果

训练好的权重和报告结果已上传到 GitHub Release `v4.0`：

- [Transformer checkpoint.pt](https://github.com/sidchy/math3705m-neural-networks-assignments/releases/download/v4.0/checkpoint.pt)
- [FastText embeddings.pt](https://github.com/sidchy/math3705m-neural-networks-assignments/releases/download/v4.0/embeddings.pt)
- [Report artifacts zip](https://github.com/sidchy/math3705m-neural-networks-assignments/releases/download/v4.0/task4_report_artifacts.zip)

Release 页面：
https://github.com/sidchy/math3705m-neural-networks-assignments/releases/tag/v4.0
