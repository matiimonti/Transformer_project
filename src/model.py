"""
model.py — Full Transformer (decoder-only, GPT-style) from scratch.

Building blocks:
  - Sinusoidal Positional Encoding
  - Transformer Block (Attention + FFN + LayerNorm + residuals)
  - Full ChessTransformer model with configurable attention variant

All architectural choices are explicit and documented.
"""

import torch
import torch.nn as nn

##############################
#### Feed-Forward Network ####
##############################

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)





###########################
#### Transformer Block ####
###########################

class TransformerBlock(nn.Module):
    """
    Pre-LN transformer block: normalize before attention and FFN.
    Residual connections ensure gradients flow directly through the network.
    """
    def __init__(self, d_model: int, attention: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = attention
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask=None):
        # Pre-LN attention with residual
        x = x + self.attention(self.ln1(x), mask=mask)
        # Pre-LN FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x





