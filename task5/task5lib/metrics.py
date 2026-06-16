from __future__ import annotations

import sacrebleu

from task5lib.text import FEATURE_TERMS


def _chars(text: str) -> str:
    return " ".join(list(text.strip()))


def char_bleu(predictions: list[str], references: list[str]) -> float:
    preds = [_chars(text) for text in predictions]
    refs = [[_chars(text) for text in references]]
    return float(sacrebleu.corpus_bleu(preds, refs, use_effective_order=True).score)


def chrf(predictions: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_chrf(predictions, [references]).score)


def keyword_hits(text: str) -> dict[str, int]:
    return {term: text.count(term) for term in FEATURE_TERMS}
