from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from task5lib.io import ensure_dir, write_json


def item_id(source_code: str, row_id: int) -> str:
    return f"{source_code}:{int(row_id)}"


def group_id_for_row(row: pd.Series) -> str:
    source = str(row["source_code"])
    row_id = int(row["row_id"])
    if "entry" in row and str(row.get("entry", "")).strip():
        return f"entry:{row['entry']}"
    if source == "07" and str(row.get("source_doc", "")).strip():
        return f"{source}:doc:{row['source_doc']}"
    return f"{source}:row:{row_id}"


def collect_groups(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for source_code, df in frames.items():
        for _, row in df.iterrows():
            gid = group_id_for_row(row)
            groups.setdefault(gid, []).append(item_id(source_code, int(row["row_id"])))
    return groups


def assign_splits(groups: dict[str, list[str]], seed: int = 42) -> dict[str, list[str]]:
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    n = len(names)
    train_cut = int(n * 0.85)
    val_cut = int(n * 0.925)
    mapping = {
        "train": names[:train_cut],
        "val": names[train_cut:val_cut],
        "test": names[val_cut:],
    }
    return {split: [item for group in group_names for item in groups[group]] for split, group_names in mapping.items()}


def load_cleaned(cleaned_dir: str | Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(cleaned_dir).glob("*_clean.csv")):
        source_code = path.name.split("_", 1)[0]
        frames[source_code] = pd.read_csv(path, keep_default_na=False)
    return frames


def make_splits(cleaned_dir: str | Path, out_dir: str | Path, seed: int = 42) -> dict:
    frames = load_cleaned(cleaned_dir)
    groups = collect_groups(frames)
    splits = assign_splits(groups, seed=seed)
    out_dir = ensure_dir(out_dir)
    for split, ids in splits.items():
        write_json(out_dir / f"{split}_ids.json", ids)
    report = {
        "seed": seed,
        "groups": len(groups),
        "counts": {split: len(ids) for split, ids in splits.items()},
    }
    write_json(out_dir / "split_report.json", report)
    return report
