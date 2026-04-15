from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return np.clip(image * std + mean, 0.0, 1.0)


def plot_history(history: Sequence[Dict[str, float]], out_png: Path, title: str) -> None:
    _ensure_parent(out_png)
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_acc"]) * 100.0 for row in history]
    val_acc = [float(row["val_acc"]) * 100.0 for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(epochs, train_loss, label="train", color="#1f77b4")
    axes[0].plot(epochs, val_loss, label="val", color="#d62728")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train", color="#1f77b4")
    axes[1].plot(epochs, val_acc, label="val", color="#d62728")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_predictions_grid(
    samples: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    out_png: Path,
    title: str,
) -> None:
    _ensure_parent(out_png)
    if not samples:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No prediction samples available", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return

    limit = min(len(samples), 16)
    cols = 4
    rows = int(math.ceil(limit / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.0, 3.1 * rows))
    axes = np.array(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, sample in zip(axes, samples[:limit]):
        image = _denormalize(np.asarray(sample["image"], dtype=np.float32))
        truth = int(sample["true"])
        pred = int(sample["pred"])
        axis.imshow(np.transpose(image, (1, 2, 0)))
        axis.set_title(
            f"T: {class_names[truth]}\nP: {class_names[pred]}",
            color="#2ca02c" if truth == pred else "#d62728",
            fontsize=8,
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_top_confusions(confusions: Sequence[tuple[str, str, int]], out_png: Path, title: str) -> None:
    _ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if not confusions:
        ax.axis("off")
        ax.text(0.5, 0.5, "No off-diagonal confusions found", ha="center", va="center")
    else:
        labels = [f"{truth} -> {pred}" for truth, pred, _ in confusions][::-1]
        counts = [count for _, _, count in confusions][::-1]
        ax.barh(labels, counts, color="#4c72b0")
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_accuracy_comparison(rows: Sequence[Dict[str, object]], out_png: Path) -> None:
    _ensure_parent(out_png)
    labels = [str(row["display_name"]) for row in rows]
    scores = [float(row["test_acc"]) * 100.0 for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]
    ax.bar(labels, scores, color=colors[: len(labels)])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Final Test Accuracy Comparison")
    ax.set_ylim(0.0, max(scores + [1.0]) + 5.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_efficiency_tradeoff(rows: Sequence[Dict[str, object]], out_png: Path) -> None:
    _ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = {"scratch": "#d62728", "finetune": "#2ca02c"}
    for row in rows:
        seconds = float(row["train_seconds"])
        accuracy = float(row["test_acc"]) * 100.0
        init_mode = str(row["init_mode"])
        label = str(row["display_name"])
        ax.scatter(seconds / 60.0, accuracy, s=90, color=colors.get(init_mode, "#4c72b0"))
        ax.annotate(label, (seconds / 60.0, accuracy), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Training Time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_pair_curves(rows: Sequence[Dict[str, object]], out_png: Path, title: str) -> None:
    _ensure_parent(out_png)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    palette = {"scratch": "#d62728", "finetune": "#2ca02c"}
    for row in rows:
        history = row["history"]
        epochs = [int(item["epoch"]) for item in history]
        val_acc = [float(item["val_acc"]) * 100.0 for item in history]
        val_loss = [float(item["val_loss"]) for item in history]
        init_mode = str(row["init_mode"])
        axes[0].plot(epochs, val_loss, label=init_mode, color=palette[init_mode], linewidth=1.8)
        axes[1].plot(epochs, val_acc, label=init_mode, color=palette[init_mode], linewidth=1.8)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

