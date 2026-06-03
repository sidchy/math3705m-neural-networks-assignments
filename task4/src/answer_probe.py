"""Answer-discrimination probe for the 《九章算经》 language model.

The probe asks whether the trained LM assigns lower conditional loss to the
true answer than to a randomly selected wrong answer:

    loss(true answer | question + 荅曰) < loss(wrong answer | question + 荅曰)

This is not a full arithmetic-reasoning benchmark.  It is a lightweight
knowledge probe showing whether the model learned some association between
problem statements and answer forms from the corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch

from data import CharVocab, extract_qa_blocks, normalize_text, read_corpus
from transformer_lm import DecoderOnlyTransformer, TransformerConfig


def _load_lm(lm_dir: Path, device: torch.device):
    vocab = CharVocab.load(lm_dir / "vocab.json")
    with open(lm_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = TransformerConfig(
        vocab_size=len(vocab),
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ffn_dim=cfg["ffn_dim"],
        seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
    )
    model = DecoderOnlyTransformer(model_cfg).to(device)
    checkpoint = torch.load(lm_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


@torch.no_grad()
def continuation_loss(
    model: DecoderOnlyTransformer,
    vocab: CharVocab,
    prompt: str,
    answer: str,
    device: torch.device,
) -> float:
    """Average NLL of answer characters conditioned on prompt."""
    prompt_ids = [vocab.bos_id] + vocab.encode(prompt)
    answer_ids = vocab.encode(answer)
    if not answer_ids:
        return float("inf")

    full_ids = prompt_ids + answer_ids
    answer_start = len(prompt_ids)
    losses: List[float] = []

    for target_pos in range(answer_start, len(full_ids)):
        start = max(0, target_pos - model.config.seq_len)
        context = full_ids[start:target_pos]
        if not context:
            continue
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        log_probs = torch.log_softmax(logits, dim=-1)
        losses.append(float(-log_probs[full_ids[target_pos]].item()))

    return sum(losses) / max(len(losses), 1)


def build_probe_examples(blocks: List[Dict], max_examples: int, seed: int) -> List[Dict]:
    usable = [b for b in blocks if b.get("question") and b.get("answer")]
    rng = random.Random(seed)
    rng.shuffle(usable)
    selected = usable[:max_examples]
    answers = [b["answer"] for b in usable]

    examples: List[Dict] = []
    for block in selected:
        true_answer = block["answer"]
        wrong_answer = rng.choice(answers)
        while wrong_answer == true_answer and len(answers) > 1:
            wrong_answer = rng.choice(answers)

        examples.append(
            {
                "id": block["id"],
                "question": block["question"],
                "prompt": block["question"].strip() + "\n荅曰：",
                "true_answer": true_answer.rstrip("。") + "。",
                "wrong_answer": wrong_answer.rstrip("。") + "。",
            }
        )

    return examples


def run_probe(
    data_path: Path,
    lm_dir: Path,
    out_dir: Path,
    max_examples: int = 200,
    seed: int = 42,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, vocab = _load_lm(lm_dir, device)
    text = normalize_text(read_corpus(data_path))
    blocks = extract_qa_blocks(text)
    examples = build_probe_examples(blocks, max_examples=max_examples, seed=seed)

    results: List[Dict] = []
    correct = 0
    true_losses: List[float] = []
    wrong_losses: List[float] = []

    for ex in examples:
        true_loss = continuation_loss(model, vocab, ex["prompt"], ex["true_answer"], device)
        wrong_loss = continuation_loss(model, vocab, ex["prompt"], ex["wrong_answer"], device)
        is_correct = true_loss < wrong_loss
        correct += int(is_correct)
        true_losses.append(true_loss)
        wrong_losses.append(wrong_loss)
        results.append(
            {
                "id": ex["id"],
                "true_answer": ex["true_answer"],
                "wrong_answer": ex["wrong_answer"],
                "true_loss": round(true_loss, 6),
                "wrong_loss": round(wrong_loss, 6),
                "margin": round(wrong_loss - true_loss, 6),
                "correct": is_correct,
                "question_preview": ex["question"][:80],
            }
        )

    n = len(results)
    summary = {
        "num_examples": n,
        "accuracy": round(correct / max(n, 1), 4),
        "avg_true_loss": round(sum(true_losses) / max(n, 1), 6),
        "avg_wrong_loss": round(sum(wrong_losses) / max(n, 1), 6),
        "avg_margin": round(
            (sum(wrong_losses) - sum(true_losses)) / max(n, 1), 6
        ),
        "seed": seed,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "examples": results}
    with open(out_dir / "answer_probe.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved answer probe → {out_dir / 'answer_probe.json'}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Run answer-discrimination probe")
    ap.add_argument("--data", required=True, help="Path to corpus .txt file")
    ap.add_argument("--lm", required=True, help="Path to LM run directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_probe(
        data_path=Path(args.data),
        lm_dir=Path(args.lm),
        out_dir=Path(args.out),
        max_examples=args.max_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
