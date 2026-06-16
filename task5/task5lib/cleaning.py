from __future__ import annotations

from pathlib import Path

import pandas as pd

from task5lib.excel_sources import SOURCES, ExcelSource
from task5lib.io import ensure_dir, write_json
from task5lib.text import apply_corrections, normalize_text


def read_corrections(raw_dir: str | Path) -> dict[str, str]:
    path = Path(raw_dir) / "有误词条.xlsx"
    df = pd.read_excel(path, header=None)
    corrections: dict[str, str] = {}
    for _, row in df.iterrows():
        wrong = normalize_text(row.iloc[0] if len(row) > 0 else "")
        right = normalize_text(row.iloc[1] if len(row) > 1 else "")
        status = normalize_text(row.iloc[2] if len(row) > 2 else "1")
        if wrong and right and status not in {"0", "未修正"}:
            corrections[wrong] = right
    return corrections


def _rename_columns(df: pd.DataFrame, source: ExcelSource) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").rename(columns=source.rename)
    for column in source.text_columns:
        if column not in df.columns:
            df[column] = ""
    return df


def clean_source(raw_dir: str | Path, source: ExcelSource, corrections: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    path = Path(raw_dir) / source.filename
    df = pd.read_excel(path, **source.read_kwargs)
    original_rows = len(df)
    df = _rename_columns(df, source)
    keep_columns = list(dict.fromkeys(["source_code", "source_file", "row_id", *source.text_columns]))
    df["source_code"] = source.code
    df["source_file"] = source.filename
    df["row_id"] = range(original_rows)
    for column in source.text_columns:
        df[column] = df[column].map(normalize_text)
        if source.kind == "parallel" or source.code in {"06", "07", "08", "09"}:
            df[column] = df[column].map(lambda value: apply_corrections(value, corrections))
    df = df[keep_columns]
    if source.kind == "parallel":
        before = len(df)
        df = df[(df["dialect"] != "") & (df["translation"] != "")]
        dropped_empty = before - len(df)
    elif "entry" in df.columns:
        before = len(df)
        df = df[df["entry"] != ""]
        dropped_empty = before - len(df)
    else:
        dropped_empty = 0
    before_dedup = len(df)
    key_columns = [column for column in source.key_columns if column in df.columns]
    df = df.drop_duplicates(subset=key_columns).reset_index(drop=True)
    report = {
        "source_code": source.code,
        "source_file": source.filename,
        "original_rows": original_rows,
        "clean_rows": len(df),
        "dropped_empty": dropped_empty,
        "dropped_duplicates": before_dedup - len(df),
    }
    return df, report


def clean_all(raw_dir: str | Path, out_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = ensure_dir(out_dir)
    corrections = read_corrections(raw_dir)
    reports = []
    for source in SOURCES:
        df, report = clean_source(raw_dir, source, corrections)
        df.to_csv(out_dir / f"{source.code}_clean.csv", index=False, encoding="utf-8")
        reports.append(report)
    summary = {
        "corrections_loaded": len(corrections),
        "sources": reports,
        "total_clean_rows": sum(item["clean_rows"] for item in reports),
    }
    write_json(out_dir / "cleaning_report.json", summary)
    return summary
