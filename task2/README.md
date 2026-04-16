# Task 2: Scratch vs Fine-tune on CIFAR-10

这个目录是课程作业二的完整实现，目标是在 `CIFAR-10` 上比较两类 CNN 模型的 `from scratch` 与 `fine-tune`：

- `DenseNet121`
- `ResNeXt50_32x4d`

核心交付包括：

- Colab 实验 notebook：`colab_notebook.ipynb`
- 训练脚本与汇总代码：`src/`、`train.py`、`run_all.py`、`summarize_runs.py`
- LaTeX 报告模板与生成脚本：`report/main.tex`、`make_report.py`

## 目录结构

- `colab_notebook.ipynb`：Colab 端到端运行入口
- `src/`：数据、模型、训练、绘图等逻辑
- `train.py`：单次实验 CLI
- `run_all.py`：顺序跑 4 组主实验，并在总训练时间不足 2 小时时继续扩展 scratch 组
- `summarize_runs.py`：读取 `runs/*/results.json`，生成汇总图和 `summary.json`
- `make_report.py`：根据汇总结果输出 `report/main.tex`
- `report/`：PDF 报告目录

## Colab 推荐流程

1. 打开 `colab_notebook.ipynb`
2. 挂载 Google Drive
3. 克隆你的 GitHub 仓库到 Drive 或 `/content`
4. 安装轻量依赖：

```bash
pip install -q -r task2/requirements-colab.txt
```

5. 先执行四组 `1 epoch smoke test`
6. 再执行正式训练：

```bash
python task2/run_all.py \
  --data-root /content/drive/MyDrive/task2_pet/data \
  --output-root /content/drive/MyDrive/task2_pet/runs \
  --device cuda \
  --num-workers 2
```

7. 训练结束后汇总结果：

```bash
python task2/summarize_runs.py \
  --runs-root /content/drive/MyDrive/task2_pet/runs \
  --report-dir /content/drive/MyDrive/task2_pet/report
```

8. 将 `runs/` 和 `report/` 同步回本地后，在本地生成报告源码：

```bash
python3 task2/make_report.py \
  --summary-json task2/report/summary.json \
  --outdir task2/report \
  --repo-url https://github.com/<your-username>/<your-repo>
```

9. 使用本地 `tectonic` 编译 PDF：

```bash
./tools/tectonic/tectonic -X compile task2/report/main.tex --outdir task2/report
```

## 单次实验命令

```bash
python task2/train.py \
  --preset densenet121_finetune \
  --data-root /content/drive/MyDrive/task2_pet/data \
  --output-root /content/drive/MyDrive/task2_pet/runs \
  --device cuda
```

烟雾测试：

```bash
python task2/run_all.py \
  --data-root /content/drive/MyDrive/task2_pet/data \
  --output-root /content/drive/MyDrive/task2_pet/runs_smoke \
  --device cuda \
  --smoke-test
```

## 输出约定

每次运行产物目录固定为：

```text
runs/<model>_<mode>_<timestamp>/
```

目录内至少包含：

- `history.csv`
- `results.json`
- `best.pt`
- `last.pt`
- `curves.png`
- `predictions_grid.png`
- `top_confusions.png`

## 注意事项

- 若 Colab 免费 GPU 显存不足，优先把 `ResNeXt50_32x4d` 的 batch size 从 `48` 改到 `32`
- `run_all.py` 会在总训练时长不足 `7200` 秒时，自动从 scratch checkpoint 继续追加训练
- 正式提交 PDF 时不要附代码，只放仓库地址
