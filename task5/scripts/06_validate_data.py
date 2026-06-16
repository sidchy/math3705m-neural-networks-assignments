from __future__ import annotations

import argparse
from pathlib import Path

from task5lib.io import read_jsonl


REQUIRED_SFT = {"id", "instruction", "input", "output", "task_type", "source", "source_file", "row_id", "group_id"}


def validate_sft_file(path: Path) -> dict:
    rows = read_jsonl(path)
    ids = set()
    for idx, row in enumerate(rows):
        missing = REQUIRED_SFT - set(row)
        if missing:
            raise ValueError(f"{path}:{idx} missing {sorted(missing)}")
        if not row["input"] or not row["output"]:
            raise ValueError(f"{path}:{idx} has empty input/output")
        if row["id"] in ids:
            raise ValueError(f"{path}:{idx} duplicate id {row['id']}")
        ids.add(row["id"])
    return {"path": str(path), "rows": len(rows), "unique_ids": len(ids)}


def validate_data_dir(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    reports = []
    for split in ["train", "val", "test"]:
        reports.append(validate_sft_file(data_dir / f"sft_{split}.jsonl"))
    for split in ["train", "val", "test"]:
        path = data_dir / f"pretrain_{split}.txt"
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"{path} missing or empty")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    for report in validate_data_dir(args.data_dir):
        print(report)


if __name__ == "__main__":
    main()
