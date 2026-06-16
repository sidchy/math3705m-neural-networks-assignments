from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from task5lib.io import ensure_dir, read_json, write_json
from task5lib.splits import item_id


TEXT_COLUMNS = ["dialect", "entry"]


def load_split_ids(splits_dir: str | Path) -> dict[str, set[str]]:
    return {
        name: set(read_json(Path(splits_dir) / f"{name}_ids.json"))
        for name in ["train", "val", "test"]
    }


def collect_lines(cleaned_dir: str | Path, ids: set[str]) -> list[str]:
    lines: list[str] = []
    for path in sorted(Path(cleaned_dir).glob("*_clean.csv")):
        source_code = path.name.split("_", 1)[0]
        df = pd.read_csv(path, keep_default_na=False)
        for _, row in df.iterrows():
            if item_id(source_code, int(row["row_id"])) not in ids:
                continue
            for column in TEXT_COLUMNS:
                if column in row and str(row[column]).strip():
                    value = str(row[column]).strip()
                    if len(value) <= 200:
                        lines.append(value)
    return lines


def write_corpus(cleaned_dir: str | Path, splits_dir: str | Path, out_dir: str | Path) -> dict:
    split_ids = load_split_ids(splits_dir)
    out_dir = ensure_dir(out_dir)
    stats = {}
    for split, ids in split_ids.items():
        lines = collect_lines(cleaned_dir, ids)
        text = "\n".join(lines) + ("\n" if lines else "")
        (out_dir / f"pretrain_{split}.txt").write_text(text, encoding="utf-8")
        chars = [ch for line in lines for ch in line]
        stats[split] = {
            "lines": len(lines),
            "chars": len(chars),
            "unique_chars": len(set(chars)),
            "avg_len": (sum(len(line) for line in lines) / len(lines)) if lines else 0.0,
        }
    write_json(out_dir / "pretrain_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.cleaned, args.splits, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
