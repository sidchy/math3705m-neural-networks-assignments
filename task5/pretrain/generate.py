from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pretrain.data import CharVocab
from pretrain.model import GPT, GPTConfig
from task5lib.io import read_json, write_json


def load_vocab(path: str) -> CharVocab:
    data = read_json(path)
    stoi = {str(k): int(v) for k, v in data["stoi"].items()}
    return CharVocab(stoi=stoi, itos={int(v): str(k) for k, v in stoi.items()})


def generate_samples(checkpoint: str, vocab_path: str, prompts: list[str], out: str) -> list[dict]:
    vocab = load_vocab(vocab_path)
    payload = torch.load(checkpoint, map_location="cpu")
    cfg_dict = {**payload["config"]}
    cfg_dict.pop("vocab_size", None)
    cfg = GPTConfig(vocab_size=len(vocab.stoi), **cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    rows = []
    for prompt in prompts:
        ids = vocab.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long)
        generated = model.generate(idx, max_new_tokens=80)
        rows.append({"prompt": prompt, "generated": vocab.decode(generated[0].tolist())})
    write_json(out, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", default="samples.json")
    args = parser.parse_args()
    prompts = ["", "热", "覅", "阿"]
    rows = generate_samples(args.checkpoint, args.vocab, prompts, args.out)
    for row in rows:
        print(row["prompt"], "->", row["generated"][:60])


if __name__ == "__main__":
    main()
