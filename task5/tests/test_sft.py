import pandas as pd

from task5lib.sft import build_records_for_row, render_prompt


def test_build_parallel_records_has_forward_and_reverse():
    row = pd.Series({
        "source_code": "06",
        "source_file": "06.xlsx",
        "row_id": 3,
        "dialect": "热显热",
        "translation": "很热",
    })
    records = build_records_for_row(row, split="train")
    assert {record.task_type for record in records} == {"wz_to_zh", "zh_to_wz"}
    assert records[0].group_id == "06:3"


def test_render_prompt_uses_single_template():
    prompt = render_prompt("请将以下温州话翻译成普通话。", "热显热")
    assert prompt.startswith("### 指令")
    assert prompt.endswith("### 回答\n")
