from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class CharVocab:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharVocab":
        chars = sorted(set(text))
        stoi = {"<unk>": 0}
        for i, ch in enumerate(chars):
            stoi[ch] = i + 1
        return cls(stoi=stoi, itos={i: ch for ch, i in stoi.items()})

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CharBlockDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocab, block_size: int):
        self.ids = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx: int):
        chunk = self.ids[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def unigram_baseline_loss(text: str, vocab: CharVocab) -> float:
    ids = vocab.encode(text)
    counts = torch.bincount(torch.tensor(ids), minlength=len(vocab.stoi)).float()
    probs = counts / counts.sum().clamp_min(1)
    return float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
