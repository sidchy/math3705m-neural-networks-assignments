from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.config import DATASET_NAME, PRESET_ORDER
from src.plotting import plot_accuracy_comparison, plot_efficiency_tradeoff, plot_pair_curves


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "phase": row["phase"],
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "lr_backbone": float(row["lr_backbone"]),
                    "lr_head": float(row["lr_head"]),
                    "epoch_seconds": float(row["epoch_seconds"]),
                    "total_train_seconds": float(row["total_train_seconds"]),
                }
            )
    return rows


def _canonical_key(row: Dict[str, object]) -> str:
    return f"{row['model']}_{row['init_mode']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all completed task2 runs into report figures.")
    parser.add_argument("--runs-root", default="task2/runs")
    parser.add_argument("--report-dir", default="task2/report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    report_dir = Path(args.report_dir)
    figs_dir = report_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    discovered: Dict[str, Dict[str, object]] = {}
    for result_path in sorted(runs_root.glob("*/results.json")):
        payload = _load_json(result_path)
        key = _canonical_key(payload)
        history = _load_history(Path(payload["history_path"]))
        payload["history"] = history
        payload["result_path"] = str(result_path)
        discovered[key] = payload

    rows = [discovered[key] for key in PRESET_ORDER if key in discovered]
    plot_accuracy_comparison(rows, figs_dir / "accuracy_comparison.png")
    plot_efficiency_tradeoff(rows, figs_dir / "efficiency_tradeoff.png")

    densenet_rows = [row for row in rows if row["model"] == "densenet121"]
    resnext_rows = [row for row in rows if row["model"] == "resnext50_32x4d"]
    if densenet_rows:
        plot_pair_curves(densenet_rows, figs_dir / "densenet_curves.png", "DenseNet121: Scratch vs Fine-tune")
    if resnext_rows:
        plot_pair_curves(resnext_rows, figs_dir / "resnext_curves.png", "ResNeXt50_32x4d: Scratch vs Fine-tune")

    best_row = max(rows, key=lambda item: float(item["test_acc"])) if rows else None
    if best_row is not None:
        best_run_dir = Path(best_row["run_dir"])
        shutil.copy2(best_run_dir / "predictions_grid.png", figs_dir / "best_predictions_grid.png")
        shutil.copy2(best_run_dir / "top_confusions.png", figs_dir / "best_top_confusions.png")

    summary_rows: List[Dict[str, object]] = []
    for row in rows:
        summary_rows.append(
            {
                "model": row["model"],
                "init_mode": row["init_mode"],
                "display_name": row["display_name"],
                "best_val_acc": float(row["best_val_acc"]),
                "test_acc": float(row["test_acc"]),
                "macro_f1": float(row["macro_f1"]),
                "train_seconds": float(row["train_seconds"]),
                "best_epoch": int(row["best_epoch"]),
                "epochs": int(row["epochs"]),
                "batch_size": int(row["batch_size"]),
                "run_dir": row["run_dir"],
            }
        )

    summary_csv = report_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["display_name"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    summary_json = report_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "dataset_name": DATASET_NAME,
                "rows": summary_rows,
                "best_run": best_row,
                "total_train_seconds": sum(float(row["train_seconds"]) for row in summary_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()

