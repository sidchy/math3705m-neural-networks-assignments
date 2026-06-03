"""Training script for the decoder-only Transformer language model.

Supports two presets:

* ``smoke`` — tiny config for quick local smoke testing.
* ``gpu`` — production config for cloud-server GPU training.

Usage::

    python src/train_lm.py --data "九章算经 2.txt" --preset smoke --out runs/smoke
    python src/train_lm.py --data "九章算经 2.txt" --preset gpu --out runs/transformer
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import (
    build_datasets,
    build_char_stream,
    create_lm_examples,
    normalize_text,
    read_corpus,
    CharVocab,
)
from evaluate import compute_loss, evaluate_model, generate_samples, save_samples
from transformer_lm import DecoderOnlyTransformer, TransformerConfig

# ──────────────────────────────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────────────────────────────

PRESETS = {
    "gpu": dict(
        d_model=256,
        n_layers=4,
        n_heads=4,
        ffn_dim=1024,
        seq_len=128,
        dropout=0.1,
        batch_size=64,
        epochs=30,
        lr=3e-4,
    ),
    "smoke": dict(
        d_model=64,
        n_layers=1,
        n_heads=2,
        ffn_dim=128,
        seq_len=64,
        dropout=0.1,
        batch_size=8,
        epochs=1,
        lr=5e-4,
    ),
}

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    device: torch.device,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    indices = list(range(len(data)))
    random.shuffle(indices)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        x = torch.tensor(data[batch_idx], dtype=torch.long, device=device)
        logits = model(x)
        preds = logits[:, :-1, :].contiguous()
        targets = x[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            preds.view(-1, preds.size(-1)),
            targets.view(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    vocab: CharVocab,
    config: dict,
    out_dir: Path,
):
    """Full training loop with checkpointing and generation."""
    device = get_device()
    model = model.to(device)

    print(f"Device: {device}")
    print(f"Train examples: {len(train_data)}, Val examples: {len(val_data)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    metrics: List[Dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(
            model, train_data, config["batch_size"], optimizer, vocab.pad_id, device
        )

        val_result = evaluate_model(model, torch.tensor(val_data, dtype=torch.long), vocab.pad_id)
        val_loss = val_result["loss"]
        val_ppl = val_result["perplexity"]

        scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_ppl, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        metrics.append(entry)
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_ppl: {val_ppl:.2f} | "
            f"lr: {entry['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "vocab_size": len(vocab),
            }
            torch.save(checkpoint, out_dir / "checkpoint.pt")

    # Save final artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    vocab.save(out_dir / "vocab.json")

    # Generate samples
    prompts = ["今有田廣", "荅曰", "方田術曰", "句股"]
    samples = generate_samples(
        model, vocab, prompts, max_new_tokens=80, temperature=0.8, top_k=40
    )
    save_samples(samples, out_dir / "samples.txt", out_dir / "samples.json")

    # Evaluate on test set with the best checkpoint
    checkpoint = torch.load(out_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_result = evaluate_model(model, torch.tensor(test_data, dtype=torch.long), vocab.pad_id)
    test_metrics = {
        "test_loss": round(test_result["loss"], 6),
        "test_perplexity": round(test_result["perplexity"], 4),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"epochs": metrics, "test": test_metrics}, f, indent=2)

    print(f"\nSaved outputs to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['test_loss']:.4f}  |  Test perplexity: {test_metrics['test_perplexity']:.2f}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Train a decoder-only Transformer LM on 《九章算经》"
    )
    ap.add_argument("--data", required=True, help="Path to corpus .txt file")
    ap.add_argument("--preset", choices=["gpu", "smoke"], default="gpu")
    ap.add_argument("--out", required=True, help="Output directory for run artifacts")
    args = ap.parse_args()

    set_seed(42)

    # Load and prepare data
    text = read_corpus(args.data)
    text = normalize_text(text)

    chars = sorted(set(text))
    vocab = CharVocab(chars)

    preset = PRESETS[args.preset]
    train_data, val_data, test_data = build_datasets(
        text, vocab, preset["seq_len"], seed=42
    )

    cfg = TransformerConfig(
        vocab_size=len(vocab),
        d_model=preset["d_model"],
        n_layers=preset["n_layers"],
        n_heads=preset["n_heads"],
        ffn_dim=preset["ffn_dim"],
        seq_len=preset["seq_len"],
        dropout=preset["dropout"],
    )

    model = DecoderOnlyTransformer(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Config: d_model={cfg.d_model}, layers={cfg.n_layers}, "
          f"heads={cfg.n_heads}, ffn={cfg.ffn_dim}, seq_len={cfg.seq_len}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train(model, train_data, val_data, test_data, vocab, preset, out_dir)


if __name__ == "__main__":
    main()
