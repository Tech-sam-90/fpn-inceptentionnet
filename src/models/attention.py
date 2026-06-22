from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttention2D(nn.Module):
    """Multi-head self-attention applied to a 2-D feature map.

    The spatial positions are treated as tokens; the channel dimension is
    the embedding dimension.  A residual connection and layer-norm are
    applied after attention.
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
