from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "TBD"
    return f"{value * 100:.2f}\\%"


def _fmt_min(seconds: float | None) -> str:
    if seconds is None:
        return "TBD"
    return f"{seconds / 60.0:.1f} min"


def _fmt_hour(seconds: float | None) -> str:
    if seconds is None:
        return "TBD"
    return f"{seconds / 3600.0:.2f} h"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"rows": [], "best_run": None, "total_train_seconds": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


def _row_map(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {f"{row['model']}_{row['init_mode']}": row for row in rows}


def _table_row(row: Dict[str, object] | None, fallback_label: str) -> str:
    if row is None:
        return f"{_latex_escape(fallback_label)} & TBD & TBD & TBD & TBD \\\\"
    return (
        f"{_latex_escape(str(row['display_name']))} & {_fmt_pct(float(row['test_acc']))} & {_fmt_pct(float(row['macro_f1']))} & "
        f"{int(row['best_epoch'])} & {_fmt_min(float(row['train_seconds']))} \\\\"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the LaTeX report source for task2.")
    parser.add_argument("--summary-json", default="task2/report/summary.json")
    parser.add_argument("--outdir", default="task2/report")
    parser.add_argument("--repo-url", default="https://github.com/<your-username>/task2-cnn-transfer")
    parser.add_argument("--student-name", default="柴昊阳")
    parser.add_argument("--student-id", default="22542013")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(Path(args.summary_json))
    rows = list(summary.get("rows", []))
    row_by_key = _row_map(rows)
    total_train_seconds = float(summary.get("total_train_seconds", 0.0))
    best_run = summary.get("best_run")

    densenet_scratch = row_by_key.get("densenet121_scratch")
    densenet_finetune = row_by_key.get("densenet121_finetune")
    resnext_scratch = row_by_key.get("resnext50_32x4d_scratch")
    resnext_finetune = row_by_key.get("resnext50_32x4d_finetune")

    if densenet_scratch and densenet_finetune:
        densenet_gain = float(densenet_finetune["test_acc"]) - float(densenet_scratch["test_acc"])
    else:
        densenet_gain = 0.0
    if resnext_scratch and resnext_finetune:
        resnext_gain = float(resnext_finetune["test_acc"]) - float(resnext_scratch["test_acc"])
    else:
        resnext_gain = 0.0

    if best_run:
        best_text = (
            f"最佳实验为 {_latex_escape(str(best_run['display_name']))}，测试准确率 {_fmt_pct(float(best_run['test_acc']))}，"
            f"Macro-F1 为 {_fmt_pct(float(best_run['macro_f1']))}。"
        )
    else:
        best_text = "当前报告模板尚未加载正式实验结果，图表位置与正文结构已经就绪。"

    content = rf"""
\documentclass[UTF8,a4paper,twocolumn]{{ctexart}}

\usepackage{{geometry}}
\geometry{{left=1.25cm,right=1.25cm,top=1.3cm,bottom=1.35cm,columnsep=0.42cm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{enumitem}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\textfloatsep}}{{5pt}}
\setlength{{\floatsep}}{{4pt}}
\setlength{{\intextsep}}{{4pt}}
\setlength{{\parskip}}{{2pt}}
\setlist[itemize]{{leftmargin=*,topsep=2pt,itemsep=1pt,parsep=0pt}}

\newcommand{{\safeincludegraphics}}[2]{{%
  \IfFileExists{{#2}}{{\includegraphics[width=#1]{{#2}}}}{{%
    \fbox{{\parbox[c][0.26\textheight][c]{{#1}}{{\centering Missing figure:\\\texttt{{\detokenize{{#2}}}}}}}}%
  }}%
}}

\title{{课程报告二：从头训练与微调的卷积神经网络对比实验}}
\author{{{args.student_name} {args.student_id}}}
\date{{\today}}

\begin{{document}}
\maketitle
\small

\section{{研究目标}}
本次任务比较两类典型卷积网络在细粒度图像分类任务上的两种训练策略：\textbf{{从头训练（from scratch）}}与\textbf{{微调（fine-tune）}}。实验统一采用 Oxford-IIIT Pet 37 类宠物品种数据集，并选取 DenseNet121 与 ResNeXt50\_32x4d 作为代表模型。全部训练在 Colab GPU 环境中完成，本地仅负责整理结果与编译 PDF。

仓库地址：\url{{{args.repo_url}}}

\section{{模型与数据集}}
Oxford-IIIT Pet 含有 37 个猫狗品种类别，图像分辨率较高，既适合展示迁移学习的优势，也比 CIFAR-10 更适合报告中的可视化分析。实验将官方 trainval 划分为 80\% 训练集与 20\% 验证集，测试集保持官方 split 不变。输入尺寸统一为 $224\times224$，训练增强为 RandomResizedCrop、RandomHorizontalFlip、轻量 ColorJitter 与 ImageNet 归一化。

两组模型均来自 torchvision 官方实现。DenseNet121 通过密集连接缓解梯度传播问题；ResNeXt50\_32x4d 在残差结构中引入 grouped convolution，可在不显著增大复杂度的前提下增强表示能力。微调策略统一为：先冻结 backbone 仅训练分类头 3 个 epoch，再解冻全网络继续训练。

\section{{实验设置}}
优化器统一使用 AdamW，weight decay 设为 $10^{{-4}}$。从头训练的学习率固定为 $3\times10^{{-4}}$；微调阶段则采用头部 $10^{{-3}}$、骨干 $10^{{-4}}$ 的分组学习率。DenseNet121 使用 batch size 64，ResNeXt50\_32x4d 使用 batch size 48。四组主实验分别为：
\begin{{itemize}}
\item DenseNet121 from scratch
\item DenseNet121 fine-tune
\item ResNeXt50\_32x4d from scratch
\item ResNeXt50\_32x4d fine-tune
\end{{itemize}}

累计训练时长约为 {_fmt_hour(total_train_seconds)}。该设置的目的不是做大规模超参数搜索，而是控制变量后直接观察预训练初始化对收敛速度、最终精度和训练效率的影响。

\section{{结果分析}}
\begin{{table}}[!ht]
\centering
\scriptsize
\begin{{tabular}}{{lrrrr}}
\toprule
Run & Test Acc & Macro-F1 & Best Ep & Time \\
\midrule
{_table_row(densenet_scratch, "DenseNet121 (Scratch)")}
{_table_row(densenet_finetune, "DenseNet121 (Fine-tune)")}
{_table_row(resnext_scratch, "ResNeXt50_32x4d (Scratch)")}
{_table_row(resnext_finetune, "ResNeXt50_32x4d (Fine-tune)")}
\bottomrule
\end{{tabular}}
\caption{{四组主实验的总体结果。}}
\end{{table}}

{best_text}

若聚焦于同架构对比，DenseNet121 的微调相对从头训练带来约 {densenet_gain * 100:.2f} 个百分点的测试精度提升；ResNeXt50\_32x4d 的对应提升约为 {resnext_gain * 100:.2f} 个百分点。更重要的是，微调曲线通常在前几个 epoch 就进入高精度区间，说明 ImageNet 预训练确实提供了对边缘、纹理和局部结构的有效初始化。

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/densenet_curves.png}}
\caption{{DenseNet121: scratch vs fine-tune 的验证曲线。}}
\end{{figure}}

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/resnext_curves.png}}
\caption{{ResNeXt50\_32x4d: scratch vs fine-tune 的验证曲线。}}
\end{{figure}}

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/accuracy_comparison.png}}
\caption{{四组实验的最终测试精度对比。}}
\end{{figure}}

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/efficiency_tradeoff.png}}
\caption{{训练时间与测试精度的权衡。}}
\end{{figure}}

从结构层面看，DenseNet121 参数利用率较高，往往更适合中等规模细粒度分类；ResNeXt50\_32x4d 的容量更强，但也更依赖充足训练预算。如果 Colab GPU 较弱，后者的单 epoch 训练时间更长，因而在同等预算下不一定总能取得绝对优势。

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/best_predictions_grid.png}}
\caption{{最佳模型在测试集上的代表性预测样例。}}
\end{{figure}}

\begin{{figure}}[!ht]
\centering
\safeincludegraphics{{0.96\columnwidth}}{{figs/best_top_confusions.png}}
\caption{{最佳模型中最常见的混淆类别对。}}
\end{{figure}}

从易混淆样例可以看到，主要错误集中在毛色、脸部轮廓和耳朵形态相近的品种之间。这符合细粒度图像分类的一般规律：模型并非无法识别“猫”或“狗”，而是在相似品种之间仍然会受姿态、背景和局部遮挡影响。

\section{{结论}}
本次实验表明，在 Oxford-IIIT Pet 这类中等规模、细粒度的图像分类任务上，\textbf{{微调通常显著优于从头训练}}。预训练模型不仅在前期收敛更快，也往往能在最终测试指标上取得更高结果。对于课程作业而言，这一结论足以支撑“从头训练 V.S. 微调”的核心对比：预训练不是简单的加速技巧，而是改变优化起点与泛化性能的关键因素。

\section{{参考文献}}
[1] M. Parkhi, A. Vedaldi, A. Zisserman, C. Jawahar. Cats and Dogs.\par
[2] G. Huang, Z. Liu, L. van der Maaten, K. Weinberger. Densely Connected Convolutional Networks.\par
[3] S. Xie, R. Girshick, P. Dollár, Z. Tu, K. He. Aggregated Residual Transformations for Deep Neural Networks.\par
[4] TorchVision Documentation and Pre-trained Model Zoo.

\end{{document}}
"""

    tex_path = outdir / "main.tex"
    tex_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote report template: {tex_path}")


if __name__ == "__main__":
    main()
