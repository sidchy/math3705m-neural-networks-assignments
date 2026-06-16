from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task5lib.io import ensure_dir, read_json, write_json, write_jsonl
from task5lib.sft import build_records_for_row
from task5lib.splits import item_id


def load_split_ids(splits_dir: str | Path) -> dict[str, set[str]]:
    return {split: set(read_json(Path(splits_dir) / f"{split}_ids.json")) for split in ["train", "val", "test"]}


def build_sft(cleaned_dir: str | Path, splits_dir: str | Path, out_dir: str | Path) -> dict:
    split_ids = load_split_ids(splits_dir)
    buckets = {split: [] for split in split_ids}
    for path in sorted(Path(cleaned_dir).glob("*_clean.csv")):
        source_code = path.name.split("_", 1)[0]
        df = pd.read_csv(path, keep_default_na=False)
        for _, row in df.iterrows():
            iid = item_id(source_code, int(row["row_id"]))
            split = next((name for name, ids in split_ids.items() if iid in ids), None)
            if split is None:
                continue
            for record in build_records_for_row(row, split=split):
                if split in {"val", "test"} and record.source != "human":
                    continue
                buckets[split].append(record.to_dict())
    out_dir = ensure_dir(out_dir)
    stats = {}
    for split, rows in buckets.items():
        write_jsonl(out_dir / f"sft_{split}.jsonl", rows)
        by_task: dict[str, int] = {}
        for row in rows:
            by_task[row["task_type"]] = by_task.get(row["task_type"], 0) + 1
        stats[split] = {"rows": len(rows), "by_task": by_task}
    write_json(out_dir / "sft_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(build_sft(args.cleaned, args.splits, args.out))


if __name__ == "__main__":
    main()
