"""Shared helpers for evaluation, generation, and metric computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch


def compute_loss(
    model,
    x: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute average cross-entropy loss over a batch, ignoring pad tokens.

    ``x`` is ``(batch, seq_len)`` input ids.  Targets are ``x[:, 1:]`` and
    predictions are taken from the model's output on ``x[:, :-1]``.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    logits = model(x)  # (B, S, V)
    # Predict position i from input position i, target is position i+1
    preds = logits[:, :-1, :].contiguous()  # (B, S-1, V)
    targets = x[:, 1:].contiguous()  # (B, S-1)

    loss = torch.nn.functional.cross_entropy(
        preds.view(-1, preds.size(-1)),
        targets.view(-1),
        ignore_index=pad_id,
    )
    return loss.item()


@torch.no_grad()
def evaluate_model(
    model,
    x: torch.Tensor,
    pad_id: int,
) -> Dict[str, float]:
    """Evaluate loss and perplexity on a dataset.

    Args:
        model: The language model in eval mode.
        x: ``(num_examples, seq_len)`` token ids.
        pad_id: Padding token id to ignore in loss.

    Returns:
        Dict with ``loss`` and ``perplexity``.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    # Process in chunks to avoid OOM on large validation sets
    batch_size = 64
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]
        batch = batch.to(device)
        logits = model(batch)
        preds = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, preds.size(-1)),
            targets.view(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        # Count non-pad target tokens
        n_tokens = (targets != pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = 2.0 ** avg_loss  # perplexity in base-2 (matching cross-entropy base-e / ln 2)
    # Actually use exp for natural base
    import math

    ppl = math.exp(avg_loss)

    return {"loss": avg_loss, "perplexity": ppl}


def generate_samples(
    model,
    vocab,
    prompts: List[str],
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
) -> Dict[str, str]:
    """Generate text continuations for a list of prompts.

    Returns:
        Dict mapping prompt string to generated continuation.
    """
    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, str] = {}

    for prompt in prompts:
        prompt_ids = torch.tensor(
            [vocab.encode(prompt)], device=device
        )
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=vocab.eos_id,
        )
        full_text = vocab.decode(out.squeeze(0).tolist())
        results[prompt] = full_text

    return results


def save_samples(
    samples: Dict[str, str],
    txt_path: str | Path,
    json_path: str | Path | None = None,
):
    """Save generated samples to a text file and optionally a JSON file."""
    with open(txt_path, "w", encoding="utf-8") as f:
        for prompt, text in samples.items():
            f.write(f"=== Prompt: {prompt} ===\n")
            f.write(text)
            f.write("\n\n")

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
