"""Generate report figures and LaTeX fragments from run outputs.

Usage::

    python src/generate_report_assets.py --lm runs/transformer --embed runs/fasttext --probe runs/probe --out report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _require(path: Path, desc: str):
    if not path.exists():
        print(f"ERROR: {desc} not found at {path}", file=sys.stderr)
        print("Run training and embedding scripts first, then re-run this script.",
              file=sys.stderr)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# 1. Training loss curve
# ──────────────────────────────────────────────────────────────────────


def plot_lm_loss(metrics_path: Path, out_path: Path):
    _require(metrics_path, "LM metrics.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(metrics_path) as f:
        metrics = json.load(f)

    if isinstance(metrics, dict):
        metrics = metrics["epochs"]

    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    val_ppl = [m["val_perplexity"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.2)
    ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_ppl, "g-", label="Val Perplexity", linewidth=1.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LM loss plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 2. Embedding PCA scatter plot
# ──────────────────────────────────────────────────────────────────────


def plot_embedding_pca(
    embeddings_path: Path,
    token_to_id_path: Path,
    out_path: Path,
    highlight_terms: List[str] | None = None,
):
    _require(embeddings_path, "Embeddings .pt file")
    _require(token_to_id_path, "token_to_id.json")

    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if highlight_terms is None:
        highlight_terms = ["方田", "粟米", "少廣", "方程", "句股", "畝", "步", "分", "實", "法"]

    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True).numpy()
    with open(token_to_id_path) as f:
        token_to_id = json.load(f)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.3, c="gray")

    # Highlight target terms — use <term:X> lookup first, then plain char fallback
    for term in highlight_terms:
        lookup = f"<term:{term}>"
        if lookup in token_to_id:
            tid = token_to_id[lookup]
        elif term in token_to_id:
            tid = token_to_id[term]
        else:
            continue
        x, y = coords[tid]
        ax.scatter(x, y, s=60, c="red", edgecolors="darkred", linewidth=0.5, zorder=5)
        ax.annotate(
            term,
            (x, y),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    ax.set_title(
        f"PCA of FastText-style Embeddings\n"
        f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding PCA plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 3. Corpus statistics LaTeX fragment
# ──────────────────────────────────────────────────────────────────────


def generate_corpus_stats(data_path: str, out_path: Path):
    """Write a small LaTeX snippet with corpus statistics."""
    from data import extract_qa_blocks, normalize_text, read_corpus

    _require(Path(data_path), f"Corpus file '{data_path}'")

    text = read_corpus(data_path)
    norm = normalize_text(text)
    blocks = extract_qa_blocks(norm)
    chars = set(norm)

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        rf"Total characters (decoded) & {len(text):,} \\",
        rf"Characters after normalization & {len(norm):,} \\",
        rf"Unique characters & {len(chars):,} \\",
        rf"Lines & {norm.count(chr(10)) + 1:,} \\",
        rf"“荅曰” occurrences & {norm.count('荅曰'):,} \\",
        rf"“今有” occurrences & {norm.count('今有'):,} \\",
        rf"QA blocks extracted & {len(blocks):,} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved corpus stats → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 4. Generated samples LaTeX snippet
# ──────────────────────────────────────────────────────────────────────


def _latex_escape(s: str) -> str:
    for char, repl in [
        ("\\", r"\textbackslash "),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde "),
        ("^", r"\textasciicircum "),
    ]:
        s = s.replace(char, repl)
    return s


def generate_samples_tex(lm_dir: Path, out_path: Path):
    """Read samples.json and write a small LaTeX snippet of generated text."""
    samples_path = lm_dir / "samples.json"
    _require(samples_path, "LM samples.json")

    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)

    lines: List[str] = []
    for prompt, text in samples.items():
        safe_prompt = _latex_escape(prompt)
        safe_text = _latex_escape(text)
        # Truncate long generated text for the two-column layout
        display_text = safe_text if len(safe_text) <= 300 else safe_text[:300] + "..."
        lines.append(r"\noindent\textbf{Prompt:} “" + safe_prompt + r"”")
        lines.append(r"\begin{quote}\small " + display_text + r"\end{quote}")
        lines.append(r"\vspace{4pt}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved generated samples → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 5. Answer probe LaTeX snippet
# ──────────────────────────────────────────────────────────────────────


def generate_answer_probe_tex(probe_dir: Path, out_path: Path):
    """Read answer_probe.json and write a concise LaTeX result table."""
    probe_path = probe_dir / "answer_probe.json"
    _require(probe_path, "answer_probe.json")

    with open(probe_path, encoding="utf-8") as f:
        payload = json.load(f)

    summary = payload["summary"]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        rf"Probe examples & {summary['num_examples']} \\",
        rf"Accuracy & {summary['accuracy'] * 100:.2f}\% \\",
        rf"Avg. true-answer loss & {summary['avg_true_loss']:.3f} \\",
        rf"Avg. wrong-answer loss & {summary['avg_wrong_loss']:.3f} \\",
        rf"Avg. margin & {summary['avg_margin']:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{答案判别探针结果。Margin = wrong loss $-$ true loss，越大表示模型越偏好真实答案。}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved answer probe table → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Generate report figures and tables from run outputs"
    )
    ap.add_argument("--lm", required=True, help="Path to LM run directory")
    ap.add_argument("--embed", required=True, help="Path to embedding run directory")
    ap.add_argument("--probe", default=None, help="Path to answer-probe run directory")
    ap.add_argument("--data", default="九章算经 2.txt", help="Path to corpus .txt file")
    ap.add_argument("--out", required=True, help="Output directory for figures")
    args = ap.parse_args()

    lm_dir = Path(args.lm)
    embed_dir = Path(args.embed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LM loss curve
    try:
        plot_lm_loss(lm_dir / "metrics.json", out_dir / "lm_loss.png")
    except SystemExit:
        print("(Skipping LM loss plot — metrics.json not found)")

    # PCA scatter
    try:
        plot_embedding_pca(
            embed_dir / "embeddings.pt",
            embed_dir / "token_to_id.json",
            out_dir / "embedding_pca.png",
        )
    except SystemExit:
        print("(Skipping PCA plot — embeddings not found)")

    # Generated samples
    try:
        generate_samples_tex(lm_dir, out_dir / "generated_samples.tex")
    except SystemExit:
        print("(Skipping generated samples — samples.json not found)")

    # Answer probe
    if args.probe is not None:
        try:
            generate_answer_probe_tex(Path(args.probe), out_dir / "answer_probe.tex")
        except SystemExit:
            print("(Skipping answer probe — answer_probe.json not found)")

    # Corpus stats
    generate_corpus_stats(args.data, out_dir / "corpus_stats.tex")

    print(f"\nReport assets generated in {out_dir}")


if __name__ == "__main__":
    main()
