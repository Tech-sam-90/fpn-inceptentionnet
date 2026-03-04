from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModifiedInceptionBlock(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 64) -> None:
        super().__init__()
        self.branch_1x1 = ConvBnRelu(in_channels, branch_channels, kernel_size=1)
        self.branch_3x3 = ConvBnRelu(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch_5x5 = ConvBnRelu(in_channels, branch_channels, kernel_size=5, padding=2)
        self.branch_downsample = ConvBnRelu(in_channels, branch_channels, kernel_size=3, stride=2, padding=1)
        self.downsample_after_concat = nn.Sequential(
            nn.Conv2d(branch_channels * 4, branch_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(x)
        b2 = self.branch_3x3(x)
        b3 = self.branch_5x5(x)
        b4 = self.branch_downsample(x)

        target_h = min(b1.shape[-2], b2.shape[-2], b3.shape[-2], b4.shape[-2])
        target_w = min(b1.shape[-1], b2.shape[-1], b3.shape[-1], b4.shape[-1])
        if b1.shape[-2:] != (target_h, target_w):
            b1 = b1[:, :, :target_h, :target_w]
        if b2.shape[-2:] != (target_h, target_w):
            b2 = b2[:, :, :target_h, :target_w]
        if b3.shape[-2:] != (target_h, target_w):
            b3 = b3[:, :, :target_h, :target_w]
        if b4.shape[-2:] != (target_h, target_w):
            b4 = b4[:, :, :target_h, :target_w]

        merged = torch.cat([b1, b2, b3, b4], dim=1)
        return self.downsample_after_concat(merged)


class SelfAttention2D(nn.Module):
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


class InceptentionNet(nn.Module):
    def __init__(self, stem_channels: int = 64, branch_channels: int = 64, num_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = ConvBnRelu(3, stem_channels, kernel_size=3, stride=1, padding=1)
        self.inception = ModifiedInceptionBlock(stem_channels, branch_channels=branch_channels)
        embed_dim = branch_channels * 4
        self.attention = SelfAttention2D(embed_dim=embed_dim, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception(x)
        x = self.attention(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits.squeeze(1)
