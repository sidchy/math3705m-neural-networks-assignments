# Task 2: Scratch vs Fine-tune on Oxford-IIIT Pet

本目录对应课程作业二，主题是比较卷积神经网络的两种训练方式：

- 从头训练（from scratch）
- 微调（fine-tune）

实验数据集为 `Oxford-IIIT Pet`，对比模型为：

- `DenseNet121`
- `ResNeXt50_32x4d`

## 目录说明

- `src/`：数据读取、模型构建、训练循环、评价指标和绘图逻辑
- `train.py`：单组实验入口
- `run_all.py`：顺序执行四组主实验，并在累计训练时间不足 2 小时时继续追加 scratch 组训练
- `summarize_runs.py`：汇总 `runs/` 中的结果，生成 `summary.json`、`summary.csv` 和图表
- `colab_notebook.ipynb`：早期实验 notebook，保留作参考

## 复现实验

建议在有 GPU 的 Linux 环境中运行。

先跑单组实验：

```bash
python train.py \
  --preset densenet121_finetune \
  --data-root data \
  --output-root runs \
  --device cuda \
  --num-workers 2
```

再跑四组正式实验：

```bash
python -u run_all.py \
  --data-root data \
  --output-root runs \
  --device cuda \
  --num-workers 2
```

汇总结果：

```bash
python summarize_runs.py \
  --runs-root runs \
  --report-dir report
```

## 输出文件

每次实验会生成：

```text
runs/<model>_<mode>_<timestamp>/
```

其中包含：

- `history.csv`
- `results.json`
- `best.pt`
- `last.pt`
- `curves.png`
- `predictions_grid.png`
- `top_confusions.png`

## 说明

- 训练数据目录 `data/`、实验结果目录 `runs/` 和 `report/` 中的产物默认不纳入版本控制
- 仓库主要保留实验代码、配置和结果汇总流程
