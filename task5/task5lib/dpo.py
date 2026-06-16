from __future__ import annotations

from task5lib.schema import DpoRecord


ANTONYM_SWAPS = {
    "热": "冷",
    "冷": "热",
    "多": "少",
    "少": "多",
    "好": "坏",
    "有": "没有",
}


def make_dpo_prompt(input_text: str) -> str:
    return f"请将以下温州话翻译成普通话：{input_text}"


def fallback_rejected(chosen: str) -> tuple[str, str]:
    for src, dst in ANTONYM_SWAPS.items():
        if src in chosen:
            return chosen.replace(src, dst, 1), "antonym_swap"
    pieces = chosen.split("，")
    if len(pieces) > 1:
        return pieces[0], "drop_content"
    return chosen[::-1], "reverse_chars"


def make_dpo_record(sft_row: dict, rejected: str | None = None, tag: str | None = None) -> DpoRecord:
    if rejected is None or tag is None:
        rejected, tag = fallback_rejected(sft_row["output"])
    return DpoRecord(
        prompt=make_dpo_prompt(sft_row["input"]),
        chosen=sft_row["output"],
        rejected=rejected,
        source="human_vs_fallback",
        quality_tag=tag,
        source_id=sft_row["id"],
    )
