"""Data pipeline for 《九章算经》 neural language modeling.

Handles gb18030 decoding, normalization, QA extraction, character vocabulary,
and character-level language-model dataset construction.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 1. Corpus I/O and normalization
# ──────────────────────────────────────────────────────────────────────


def read_corpus(path: str | Path) -> str:
    """Read corpus bytes, decode as gb18030, and normalise line endings."""
    raw = Path(path).read_bytes()
    text = raw.decode("gb18030")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def normalize_text(text: str) -> str:
    """Normalize the decoded corpus text.

    * Replace runs of spaces / tabs inside each line with a single normal
      space, but preserve leading / trailing whitespace on lines.
    * Collapse more than two consecutive blank lines into two blank lines.
    """
    lines = text.split("\n")
    out: List[str] = []
    blank_run = 0

    for line in lines:
        stripped = line.strip(" \t")
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            leading = line[: len(line) - len(line.lstrip(" \t"))]
            trailing = line[len(line.rstrip(" \t")) :]
            inner = " ".join(stripped.split())
            out.append(f"{leading}{inner}{trailing}")

    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────
# 2. QA block extraction
# ──────────────────────────────────────────────────────────────────────


def extract_qa_blocks(text: str) -> List[dict]:
    """Extract problem-answer blocks from the corpus.

    Splits on ``〔...〕`` bracketed problem markers.  A block is kept when
    it contains ``今有`` or ``又有`` (the question phrase).  The answer is
    identified by one of ``荅曰：``, ``荅曰:``, ``答曰：``, ``答曰:``.

    Returns
    -------
    list[dict]
        Each dict has keys ``id`` (int), ``question`` (str), ``answer``
        (str or None if no answer found), and ``raw`` (str – the full
        block text).
    """
    import re

    answer_pattern = re.compile(r"[荅答]曰[：:]")

    parts = re.split(r"〔[^〕]+〕", text)
    # The first part is everything before the first marker (preamble).
    # Subsequent parts follow their corresponding markers.
    blocks: List[dict] = []
    qid = 0

    for part in parts[1:]:  # skip preamble
        part = part.strip()
        if not part:
            continue

        has_question = "今有" in part or "又有" in part
        if not has_question:
            continue

        answer: str | None = None
        m = answer_pattern.search(part)
        if m:
            ans_start = m.end()
            # Try to take the answer up to the next punctuation or newline
            ans_text = part[ans_start:].strip()
            # Take first "sentence" of the answer — up to the first
            # sentence-ending punctuation or newline, whichever comes first.
            end_m = re.search(r"[。\n]", ans_text)
            if end_m:
                ans_text = ans_text[: end_m.start()]
            answer = ans_text.strip()

        question = part.split("荅曰")[0].split("答曰")[0].strip()

        qid += 1
        blocks.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "raw": part,
            }
        )

    return blocks


# ──────────────────────────────────────────────────────────────────────
# 3. Character vocabulary
# ──────────────────────────────────────────────────────────────────────


class CharVocab:
    """Character-level vocabulary with special tokens.

    Special tokens (fixed order): ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, chars: List[str] | None = None):
        self._specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        self._stoi: dict[str, int] = {}
        self._itos: List[str] = []

        if chars is not None:
            self.build(chars)

    def build(self, chars: List[str]):
        """Build the vocab from a list of characters (specials are prepended)."""
        self._stoi = {tok: i for i, tok in enumerate(self._specials)}
        for ch in sorted(set(chars)):
            if ch not in self._stoi:
                self._stoi[ch] = len(self._stoi)
        self._itos = [""] * len(self._stoi)
        for tok, idx in self._stoi.items():
            self._itos[idx] = tok

    def __len__(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> List[int]:
        return [self._stoi.get(ch, self._stoi[self.UNK]) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._itos[i] for i in ids if i < len(self._itos))

    def save(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"specials": self._specials, "itos": self._itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab._specials = data["specials"]
        vocab._itos = data["itos"]
        vocab._stoi = {tok: i for i, tok in enumerate(vocab._itos)}
        return vocab

    @property
    def pad_id(self) -> int:
        return self._stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self._stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self._stoi[self.EOS]

    @property
    def unk_id(self) -> int:
        return self._stoi[self.UNK]


# ──────────────────────────────────────────────────────────────────────
# 4. Character-level language-model datasets
# ──────────────────────────────────────────────────────────────────────


def build_char_stream(text: str, vocab: CharVocab) -> List[int]:
    """Build one long token stream: ``<bos> + chars + <eos>``."""
    tokens = vocab.encode(text)
    return [vocab.bos_id] + tokens + [vocab.eos_id]


def create_lm_examples(
    stream: List[int],
    seq_len: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-length input/target pairs and split into train/val/test.

    Returns three (num_examples, seq_len) uint16 arrays of input token ids.
    Targets are offset by 1, so the last input column's target is the next
    token in the stream — callers should use ``x[:, 1:]`` as target
    (or equivalently construct targets as ``stream[i+1 : i+1+seq_len]``).

    The returned arrays contain *input* sequences only; the corresponding
    targets are the inputs shifted right by one position.
    """
    rng = random.Random(seed)
    total_tokens = len(stream)
    num_examples = (total_tokens - 1) // seq_len
    usable = num_examples * seq_len

    indices = list(range(num_examples))
    rng.shuffle(indices)

    n_train = int(num_examples * train_ratio)
    n_val = int(num_examples * val_ratio)

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train : n_train + n_val])
    test_idx = sorted(indices[n_train + n_val :])

    def _build(subset_indices):
        arr = np.zeros((len(subset_indices), seq_len), dtype=np.uint16)
        for i, ex in enumerate(subset_indices):
            start = ex * seq_len
            arr[i] = stream[start : start + seq_len]
        return arr

    return _build(train_idx), _build(val_idx), _build(test_idx)


def build_datasets(
    text: str,
    vocab: CharVocab,
    seq_len: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: build char stream, then create train/val/test arrays."""
    stream = build_char_stream(text, vocab)
    return create_lm_examples(stream, seq_len, train_ratio, val_ratio, seed)


# ──────────────────────────────────────────────────────────────────────
# 5. CLI self-check
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Data pipeline self-check")
    ap.add_argument("--data", required=True, help="Path to the corpus .txt file")
    args = ap.parse_args()

    text = read_corpus(args.data)
    print(f"Decoded characters: {len(text)}")
    print(f"Lines: {text.count(chr(10)) + 1}")

    norm = normalize_text(text)
    print(f"Normalized characters: {len(norm)}")

    blocks = extract_qa_blocks(norm)
    print(f"QA blocks extracted: {len(blocks)}")
    if blocks:
        print(f"  First block id: {blocks[0]['id']}")
        print(f"  First block question (first 80 chars): {blocks[0]['question'][:80]}")
        print(f"  First block answer: {blocks[0]['answer']}")

    # Build vocabulary
    chars = sorted(set(norm))
    vocab = CharVocab(chars)
    print(f"Vocabulary size: {len(vocab)}")

    # Quick dataset check
    stream = build_char_stream(norm, vocab)
    print(f"Token stream length: {len(stream)}")
    train, val, test = create_lm_examples(stream, seq_len=64, seed=42)
    print(f"Train examples: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 荅曰 / 答曰 counts
    print(f"'荅曰' occurrences: {norm.count('荅曰')}")
    print(f"'今有' occurrences: {norm.count('今有')}")


if __name__ == "__main__":
    main()
