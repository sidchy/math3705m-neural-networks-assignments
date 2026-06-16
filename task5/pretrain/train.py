from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pretrain.data import CharBlockDataset, CharVocab, unigram_baseline_loss
from pretrain.model import GPT, GPTConfig
from task5lib.io import ensure_dir, write_json


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config_path: str) -> dict:
    config = load_config(config_path)
    device = config.get("device", "cpu")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    train_text = Path(config["train_text"]).read_text(encoding="utf-8")
    val_text = Path(config["val_text"]).read_text(encoding="utf-8")
    out_dir = ensure_dir(config["out_dir"])

    vocab = CharVocab.build(train_text)
    train_ds = CharBlockDataset(train_text, vocab, config["block_size"])
    val_ds = CharBlockDataset(val_text, vocab, config["block_size"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    baseline = unigram_baseline_loss(val_text, vocab)
    print(f"vocab_size={len(vocab.stoi)} baseline_loss={baseline:.4f}")

    model_config = GPTConfig(
        vocab_size=len(vocab.stoi),
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.01))

    metrics = []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 0)
    global_step = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_losses = []
        for step, (x, y) in enumerate(train_loader):
            if config.get("steps_per_epoch") and step >= config["steps_per_epoch"]:
                break
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            global_step += 1

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = evaluate(model, val_loader, device)
        metrics.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "config": model_config.__dict__}, out_dir / "checkpoint.pt")
        elif early_stopping_patience > 0:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    write_json(out_dir / "vocab.json", {"stoi": vocab.stoi, "itos": vocab.itos})
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", {"baseline_loss": baseline, "epochs": metrics})
    return {"baseline_loss": baseline, "final_epochs": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = train(args.config)
    print(json.dumps({"baseline_loss": result["baseline_loss"], "epochs": len(result["final_epochs"])}))


if __name__ == "__main__":
    main()
