from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttention2D(nn.Module):
    """Multi-head self-attention applied over 2-D spatial feature maps.

    The spatial dimensions are flattened into a token sequence, multi-head
    self-attention is computed, and the result is reshaped back to the original
    spatial layout with a residual connection and layer normalisation.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.view(batch, channels, height * width).transpose(1, 2)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        return out.transpose(1, 2).view(batch, channels, height, width)
