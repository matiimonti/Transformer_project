"""
attention.py — Multi-Head Attention variants from scratch.

Implements:
  - Vanilla Multi-Head Self-Attention (MHA)
  - Multi-Head Attention with Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - Sparse / Sliding-Window Attention

No nn.MultiheadAttention used anywhere — everything is explicit.

B — Batch size. How many sequences you're processing at once. e.g. 64 games simultaneously
T — Time steps, i.e. sequence length. How many tokens in each sequence. e.g. 128 moves
C — Channels, i.e. d_model. The embedding dimension of each token. e.g. 128

Q (Query) — what you're looking for. "What information do I need at this position?"
K (Key) — what each position is advertising. "What information do I contain?"
V (Value) — the actual content. "What information do I actually pass along if selected?"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Utility: scaled dot-product attention (shared by all variants)

def scaled_dot_product_attention(
    q: torch.Tensor,          # (B, heads, T, head_dim)
    k: torch.Tensor,          # (B, heads, T, head_dim)
    v: torch.Tensor,          # (B, heads, T, head_dim)
    mask: Optional[torch.Tensor] = None,  # (T, T) or (B, 1, T, T)
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Core attention operation.
    Dividing by sqrt(head_dim) keeps gradients stable — without this,
    dot products grow large in magnitude, pushing softmax into flat regions.
    """
    head_dim = q.size(-1)
    scale = math.sqrt(head_dim)

    # (B, heads, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        # mask = True/1 where we want to BLOCK attention
        scores = scores.masked_fill(mask == 1, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    return torch.matmul(attn_weights, v)

## Casual mask
# Verify by hand that position 0 can only attend to position 0, position 1 to positions 0-1, etc
def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask — prevents position i from attending to j > i."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

######################################################
#### Variant 1: Vanilla Multi-Head Self-Attention ####
######################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model) 

        # Reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)  

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )

        # Merge heads: (B, heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

