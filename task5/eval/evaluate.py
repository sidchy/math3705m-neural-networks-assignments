from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from task5lib.io import ensure_dir, read_json, read_jsonl, write_json
from task5lib.metrics import char_bleu, chrf


def load_predictions(path: Path) -> tuple[list[str], list[str]]:
    rows = read_jsonl(path)
    return [row["prediction"] for row in rows], [row["reference"] for row in rows]


def latex_metrics_table(rows: list[dict]) -> str:
    body = "\n".join(f"{row['model']} & {row['bleu']:.2f} & {row['chrf']:.2f} \\\\" for row in rows)
    return "\\begin{tabular}{lrr}\nModel & char-BLEU & chrF \\\\\n\\hline\n" + body + "\n\\end{tabular}\n"


def plot_pretrain(metrics_path: Path, out_path: Path) -> None:
    metrics = read_json(metrics_path)
    epochs = metrics.get("epochs", [])
    if not epochs:
        return
    plt.figure()
    plt.plot([row["epoch"] for row in epochs], [row["train_loss"] for row in epochs], label="train")
    plt.plot([row["epoch"] for row in epochs], [row["val_loss"] for row in epochs], label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate(data_dir: str, runs_dir: str, out_dir: str) -> dict:
    out = ensure_dir(out_dir)
    rows = []
    for model_name, rel in [("SFT", "sft/predictions.jsonl"), ("DPO", "dpo/predictions.jsonl")]:
        path = Path(runs_dir) / rel
        if path.exists():
            preds, refs = load_predictions(path)
            rows.append({"model": model_name, "bleu": char_bleu(preds, refs), "chrf": chrf(preds, refs)})
    (out / "auto_metrics_table.tex").write_text(latex_metrics_table(rows), encoding="utf-8")
    write_json(out / "auto_metrics.json", rows)
    pretrain_metrics = Path(runs_dir) / "pretrain" / "metrics.json"
    if pretrain_metrics.exists():
        plot_pretrain(pretrain_metrics, out / "pretrain_loss.png")
    return {"models": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/final")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()
    print(evaluate(args.data, args.runs, args.out))


if __name__ == "__main__":
    main()
