from __future__ import annotations

import hashlib
import re
from typing import Mapping


FEATURE_TERMS = ["显", "爻", "逮", "匄", "覅", "冇"]


def _to_halfwidth(text: str) -> str:
    """Convert fullwidth ASCII letters (A-Z, a-z) and digits (0-9) to halfwidth.

    Preserves fullwidth CJK punctuation (parentheses, brackets, etc.).
    """
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0xFF21 <= cp <= 0xFF3A:  # A-Z
            result.append(chr(cp - 0xFEE0))
        elif 0xFF41 <= cp <= 0xFF5A:  # a-z
            result.append(chr(cp - 0xFEE0))
        elif 0xFF10 <= cp <= 0xFF19:  # 0-9
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(ch)
    return "".join(result)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    text = _to_halfwidth(text)
    return text.strip()


def apply_corrections(text: str, corrections: Mapping[str, str]) -> str:
    fixed = text
    for wrong, right in corrections.items():
        if wrong:
            fixed = fixed.replace(wrong, right)
    return fixed


def stable_id(*parts: object) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def contains_feature_term(text: str) -> bool:
    return any(term in text for term in FEATURE_TERMS)
