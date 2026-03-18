"""
model.py — Full Transformer (decoder-only, GPT-style) from scratch.

Building blocks:
  - Sinusoidal Positional Encoding
  - Transformer Block (Attention + FFN + LayerNorm + residuals)
  - Full ChessTransformer model with configurable attention variant

All architectural choices are explicit and documented.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from attention import causal_mask

##############################
#### Feed-Forward Network ####
##############################

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation
    """
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
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


#############################
#### Positional Encoding ####
#############################

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed  (non-learned) positional encoding — adds a unique sine/cosine pattern
    to each position so the model knows where each token is.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)



########################################
#### Full Model: Chess Transformeer ####
########################################

class ChessTransformer(nn.Module):
    """
    Decoder-only transformer for chess move sequence modelling.
    """

    def __init__(
            self, 
            vocab_size: int,
            attention_factory: Callable[[], nn.Module],
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            use_sinusoidal_pe: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_sinusoidal_pe = use_sinusoidal_pe

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # Positional encoding
        # RoPE variants encode position inside attention — use plain dropout instead
        if use_sinusoidal_pe:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.emb_dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        # attention_factory() is called once per layer — each block gets its own instance
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_factory(), dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm + projection to vocab
        self.ln_final = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)
 
        # Weight tying: share token embedding and output projection weights
        # Reduces parameters and typically improves perplexity
        self.head.weight = self.token_emb.weight
 
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
 
        x = self.token_emb(idx)
 
        if self.use_sinusoidal_pe:
            x = self.pos_enc(x)
        else:
            x = self.emb_dropout(x)
 
        mask = causal_mask(T, idx.device)
 
        for block in self.blocks:
            x = block(x, mask=mask)
 
        x      = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)
 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
 
        return logits, loss
 
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature scaling and top-k sampling.
 
        temperature < 1.0 -> sharper, more confident outputs
        temperature > 1.0 -> flatter, more random outputs
        top_k             -> only sample from the top-k most likely tokens
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.max_seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float('-inf')

                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx        = torch.cat([idx, next_token], dim=1)
        finally:
            if was_training:
                self.train()

        return idx
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 

