"""FastText-style subword skip-gram embedding training for 《九章算经》.

Produces character+subword embeddings, nearest-neighbor tables, and
LaTeX exports for the course report.

Usage::

    python src/fasttext_embed.py --data "九章算经 2.txt" --out runs/fasttext
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import normalize_text, read_corpus

# ──────────────────────────────────────────────────────────────────────
# Token extraction
# ──────────────────────────────────────────────────────────────────────

MULTI_CHAR_TERMS = ["方田", "粟米", "少廣", "方程", "句股", "畝", "步", "分", "實", "法"]


def extract_tokens(text: str) -> List[str]:
    """Extract character + subword tokens from the corpus.

    Returns a flat list of token strings representing the corpus sequence,
    where each character position gets:
    * the character itself
    * its unigram, bigram, and trigram subword features
    * matching multi-character terms starting at that position.
    """
    chars = list(text)
    tokens: List[str] = []

    for i, ch in enumerate(chars):
        position_tokens = [ch]  # character-level token

        # Unigram (same as char), bigram, trigram context features
        for n in [2, 3]:
            if i + n <= len(chars):
                position_tokens.append(f"<{n}gram:{''.join(chars[i:i+n])}>")

        # Multi-character terms
        for term in MULTI_CHAR_TERMS:
            tlen = len(term)
            if i + tlen <= len(chars) and "".join(chars[i : i + tlen]) == term:
                position_tokens.append(f"<term:{term}>")

        tokens.extend(position_tokens)

    return tokens


def build_vocab_from_tokens(
    tokens: List[str], min_count: int = 1
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Build token-to-id, id-to-token, and token frequency maps."""
    freq: Dict[str, int] = defaultdict(int)
    for tok in tokens:
        freq[tok] += 1

    # Filter by min_count and sort by frequency (desc), then alphabetically
    kept = [(t, c) for t, c in freq.items() if c >= min_count]
    kept.sort(key=lambda x: (-x[1], x[0]))

    token_to_id = {t: i for i, (t, _) in enumerate(kept)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    return token_to_id, id_to_token, dict(freq)


# ──────────────────────────────────────────────────────────────────────
# Skip-gram with negative sampling
# ──────────────────────────────────────────────────────────────────────


def build_skipgram_pairs(
    token_ids: List[int],
    window: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Build (center, context) pairs with dynamic window size."""
    pairs: List[Tuple[int, int]] = []
    n = len(token_ids)

    for i, center in enumerate(token_ids):
        win = rng.randint(1, window)
        start = max(0, i - win)
        end = min(n, i + win + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))

    return pairs


class SkipGramNeg(nn.Module):
    """Skip-gram model with negative sampling."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)

    def forward(self, center: torch.Tensor, pos_context: torch.Tensor, neg_context: torch.Tensor):
        """Compute skip-gram loss with negative sampling.

        Args:
            center: ``(batch,)`` center word ids.
            pos_context: ``(batch,)`` positive context word ids.
            neg_context: ``(batch, n_neg)`` negative sample ids.

        Returns:
            Scalar loss.
        """
        v_c = self.in_embeddings(center)   # (B, D)
        u_pos = self.out_embeddings(pos_context)  # (B, D)

        pos_score = torch.sum(v_c * u_pos, dim=-1)  # (B,)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score).mean()

        u_neg = self.out_embeddings(neg_context)     # (B, K, D)
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=-1).mean()

        return pos_loss + neg_loss


def train_skipgram(
    pairs: List[Tuple[int, int]],
    vocab_size: int,
    embed_dim: int = 100,
    n_neg: int = 5,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> nn.Module:
    """Train skip-gram with negative sampling."""
    torch.manual_seed(seed)
    model = SkipGramNeg(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Build noise distribution (unigram ^ 0.75)
    freq = defaultdict(int)
    for c, ctx in pairs:
        freq[c] += 1
        freq[ctx] += 1
    freq_list = [freq.get(i, 1) for i in range(vocab_size)]
    freq_tensor = torch.tensor(freq_list, dtype=torch.float) ** 0.75
    noise_dist = freq_tensor / freq_tensor.sum()

    rng = random.Random(seed)

    pair_array = np.array(pairs, dtype=np.int64)
    n_pairs = len(pair_array)

    for epoch in range(1, epochs + 1):
        indices = list(range(n_pairs))
        rng.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(0, n_pairs, batch_size), desc=f"Epoch {epoch}/{epochs}")
        for start in pbar:
            batch_idx = indices[start : start + batch_size]
            batch = pair_array[batch_idx]
            center = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            pos_ctx = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            # Negative samples
            neg_samples = torch.multinomial(
                noise_dist, len(batch_idx) * n_neg, replacement=True
            ).view(len(batch_idx), n_neg)
            neg_samples = neg_samples.to(device)

            loss = model(center, pos_ctx, neg_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs} - avg loss: {avg_loss:.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────
# Nearest neighbors
# ──────────────────────────────────────────────────────────────────────


def compute_nearest_neighbors(
    model: SkipGramNeg,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    query_terms: List[str],
    k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Compute k nearest neighbors for query terms by cosine similarity."""
    embeddings = model.in_embeddings.weight.detach().cpu().numpy()
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    results: Dict[str, List[Tuple[str, float]]] = {}

    for term in query_terms:
        if term not in token_to_id:
            results[term] = []
            continue

        tid = token_to_id[term]
        vec = embeddings[tid]
        sims = np.dot(embeddings, vec)  # cosine similarity

        # Get top k+1 (include self), skip self
        top_indices = np.argsort(-sims)[: k + 1]
        neighbors: List[Tuple[str, float]] = []
        for idx in top_indices:
            if idx == tid:
                continue
            neighbors.append((id_to_token.get(idx, f"<{idx}>"), float(sims[idx])))
            if len(neighbors) >= k:
                break

        results[term] = neighbors

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Train FastText-style embeddings on 《九章算经》"
    )
    ap.add_argument("--data", required=True, help="Path to corpus .txt file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--embed-dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load corpus
    text = read_corpus(args.data)
    text = normalize_text(text)
    print(f"Corpus: {len(text)} characters")

    # Extract tokens
    tokens = extract_tokens(text)
    print(f"Tokens extracted: {len(tokens)}")

    # Build vocab
    token_to_id, id_to_token, freq = build_vocab_from_tokens(tokens)
    print(f"Vocabulary size: {len(token_to_id)}")

    # Save token mapping
    with open(out_dir / "token_to_id.json", "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)

    # Convert tokens to ids
    token_ids = [token_to_id[t] for t in tokens if t in token_to_id]
    print(f"Token ids (filtered): {len(token_ids)}")

    # Build skip-gram pairs
    rng = random.Random(42)
    pairs = build_skipgram_pairs(token_ids, args.window, rng)
    print(f"Skip-gram pairs: {len(pairs)}")

    # Train
    model = train_skipgram(
        pairs,
        vocab_size=len(token_to_id),
        embed_dim=args.embed_dim,
        n_neg=args.n_neg,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        seed=42,
    )

    # Save embeddings
    torch.save(model.in_embeddings.weight.detach().cpu(), out_dir / "embeddings.pt")

    # Nearest neighbors
    query_terms = ["方田", "粟米", "少廣", "方程", "句股", "畝", "步", "分", "實", "法"]
    neighbors = compute_nearest_neighbors(
        model, token_to_id, id_to_token, query_terms, k=10
    )

    with open(out_dir / "nearest_neighbors.json", "w", encoding="utf-8") as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)

    # LaTeX table
    latex_lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Term & Nearest Neighbors (cosine similarity) \\",
        r"\midrule",
    ]
    for term in query_terms:
        nbrs = neighbors.get(term, [])
        nbr_str = "; ".join(
            f"{t} ({s:.3f})" for t, s in nbrs[:5]
        )
        nbr_str = nbr_str.replace("_", r"\_").replace("&", r"\&")
        latex_lines.append(f"{term} & {nbr_str} \\\\")
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    with open(out_dir / "nearest_neighbors.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
