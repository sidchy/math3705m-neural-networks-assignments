from __future__ import annotations

from task5lib.schema import DpoRecord
from task5lib.sft import render_prompt


ANTONYM_SWAPS = {
    "热": "冷",
    "冷": "热",
    "多": "少",
    "少": "多",
    "好": "坏",
    "有": "没有",
    "大": "小",
    "上": "下",
    "来": "去",
    "前": "后",
}


def make_dpo_prompt(input_text: str) -> str:
    return render_prompt("请将以下温州话翻译成普通话。", input_text)


def fallback_rejected(chosen: str) -> tuple[str, str]:
    strategies = [
        ("antonym_swap", _antonym_swap),
        ("drop_content", _drop_content),
        ("reverse_chars", lambda s: s[::-1]),
        ("delete_random_char", _delete_random_char),
        ("repeat_first_char", _repeat_first_char),
    ]
    import random
    weighted = []
    for tag, fn in strategies:
        weight = {"antonym_swap": 3, "drop_content": 3, "reverse_chars": 1, "delete_random_char": 2, "repeat_first_char": 1}.get(tag, 1)
        weighted.extend([(tag, fn)] * weight)
    tag, fn = random.choice(weighted)
    return fn(chosen), tag


def _antonym_swap(text: str) -> str:
    for src, dst in ANTONYM_SWAPS.items():
        if src in text:
            return text.replace(src, dst, 1)
    return text[::-1]


def _drop_content(text: str) -> str:
    pieces = text.split("，")
    if len(pieces) > 1:
        return pieces[0]
    return text[:max(1, len(text) // 2)]


def _delete_random_char(text: str) -> str:
    if len(text) <= 2:
        return text[::-1]
    import random
    i = random.randint(0, len(text) - 1)
    return text[:i] + text[i + 1:]


def _repeat_first_char(text: str) -> str:
    if not text:
        return text
    return text[0] * min(4, len(text)) + text


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
