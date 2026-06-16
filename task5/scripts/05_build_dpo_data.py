from __future__ import annotations

import argparse
import random
from pathlib import Path

from task5lib.dpo import make_dpo_record
from task5lib.io import read_jsonl, write_json, write_jsonl


def build_dpo(sft_train: str | Path, out_dir: str | Path, sample_size: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rows = [row for row in read_jsonl(sft_train) if row["task_type"] in {"wz_to_zh", "example_translate"}]
    rng.shuffle(rows)
    records = [make_dpo_record(row, rng=rng).to_dict() for row in rows[:sample_size]]
    cut = max(1, int(len(records) * 0.9))
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "dpo_train.jsonl", records[:cut])
    write_jsonl(out_dir / "dpo_val.jsonl", records[cut:])
    stats = {"sample_size": len(records), "train": cut, "val": len(records) - cut}
    write_json(out_dir / "dpo_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", default="data/final/sft_train.jsonl")
    parser.add_argument("--out", default="data/final")
    parser.add_argument("--sample_size", type=int, default=1000)
    args = parser.parse_args()
    print(build_dpo(args.sft_train, args.out, args.sample_size))


if __name__ == "__main__":
    main()
