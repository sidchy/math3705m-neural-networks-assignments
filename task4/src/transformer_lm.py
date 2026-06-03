"""Decoder-only Transformer language model for 《九章算经》.

Uses ``nn.TransformerEncoder`` with a causal mask to implement a
decoder-only architecture suitable for autoregressive character-level
language modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 1024
    seq_len: int = 128
    dropout: float = 0.1


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer for character-level language modeling.

    Uses a learned token embedding plus learned positional embedding,
    followed by a stack of ``TransformerEncoderLayer`` blocks with a
    causal (upper-triangular) attention mask, and a final linear projection
    to vocabulary size.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len)`` tensor of token ids.

        Returns:
            ``(batch, seq_len, vocab_size)`` logits.
        """
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)

        tok_emb = self.token_embedding(x)  # (B, S, d_model)
        pos_emb = self.pos_embedding(positions)  # (1, S, d_model)
        h = self.dropout(tok_emb + pos_emb)

        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device), diagonal=1
        )

        h = self.transformer(h, mask=causal_mask)  # (B, S, d_model)
        logits = self.lm_head(h)  # (B, S, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation from a prompt.

        Args:
            prompt_ids: ``(1, prompt_len)`` tensor.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = no change).
            top_k: If > 0, restrict sampling to top-k logits.
            eos_id: Optional EOS token id to stop generation.

        Returns:
            ``(1, prompt_len + generated_len)`` tensor.
        """
        self.eval()
        generated = list(prompt_ids.squeeze(0).tolist())

        for _ in range(max_new_tokens):
            # Take up to seq_len context from the right
            ctx = generated[-self.config.seq_len :]
            x = torch.tensor([ctx], device=prompt_ids.device)

            logits = self(x)  # (1, ctx_len, vocab_size)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, top_k)
                probs = torch.full_like(next_logits, float("-inf"))
                probs.scatter_(0, topk_idx, topk_vals)
            else:
                probs = next_logits

            probs = torch.softmax(probs, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

            if eos_id is not None and next_token == eos_id:
                break

        return torch.tensor([generated], device=prompt_ids.device)
