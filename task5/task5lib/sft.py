from __future__ import annotations

import pandas as pd

from task5lib.schema import SftRecord
from task5lib.text import stable_id


PROMPT_TEMPLATE = """### 指令
{instruction}

### 输入
{input}

### 回答
"""


def render_prompt(instruction: str, input_text: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)


def _record(row: pd.Series, instruction: str, input_text: str, output_text: str, task_type: str, source: str = "human") -> SftRecord:
    source_code = str(row["source_code"])
    row_id = int(row["row_id"])
    group_id = f"{source_code}:{row_id}"
    return SftRecord(
        id=f"{group_id}:{task_type}:{stable_id(input_text, output_text)}",
        instruction=instruction,
        input=input_text,
        output=output_text,
        task_type=task_type,
        source=source,
        source_file=str(row["source_file"]),
        row_id=row_id,
        group_id=group_id,
    )


def build_records_for_row(row: pd.Series, split: str) -> list[SftRecord]:
    records: list[SftRecord] = []
    dialect = str(row.get("dialect", "")).strip()
    translation = str(row.get("translation", "")).strip()
    entry = str(row.get("entry", "")).strip()
    definition = str(row.get("definition", "")).strip()
    if dialect and translation:
        records.append(_record(row, "请将以下温州话翻译成普通话。", dialect, translation, "wz_to_zh"))
        if split == "train":
            records.append(_record(row, "请将以下普通话翻译成温州话。", translation, dialect, "zh_to_wz"))
        if entry:
            records.append(_record(row, "请将以下温州话例句翻译成普通话。", dialect, translation, "example_translate"))
    if entry and definition:
        records.append(_record(row, "解释以下温州话词语的意思，并给出普通话说明。", entry, definition, "lexicon_explain"))
    return records
