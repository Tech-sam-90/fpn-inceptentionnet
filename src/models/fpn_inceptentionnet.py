from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention2D


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


class _InceptionStage(nn.Module):
    """Modified Inception block with a single 2× spatial downsampling.

    Produces four parallel branches (1×1, 3×3, 5×5 at full resolution and a
    stride-2 branch), aligns them to the stride-2 spatial size, concatenates
    and applies a 3×3 smoothing convolution.  The overall stride is 2.
    """

    def __init__(self, in_channels: int, branch_channels: int = 64) -> None:
        super().__init__()
        self.branch_1x1 = ConvBnRelu(in_channels, branch_channels, kernel_size=1)
        self.branch_3x3 = ConvBnRelu(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch_5x5 = ConvBnRelu(in_channels, branch_channels, kernel_size=5, padding=2)
        self.branch_downsample = ConvBnRelu(in_channels, branch_channels, kernel_size=3, stride=2, padding=1)
        out_channels = branch_channels * 4
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(x)
        b2 = self.branch_3x3(x)
        b3 = self.branch_5x5(x)
        b4 = self.branch_downsample(x)

        target_h, target_w = b4.shape[-2], b4.shape[-1]
        b1 = F.adaptive_avg_pool2d(b1, (target_h, target_w))
        b2 = F.adaptive_avg_pool2d(b2, (target_h, target_w))
        b3 = F.adaptive_avg_pool2d(b3, (target_h, target_w))

        merged = torch.cat([b1, b2, b3, b4], dim=1)
        return self.smooth(merged)


class _FPNMerge(nn.Module):
    """Lateral connection + top-down merge for one FPN level."""

    def __init__(self, in_channels: int, fpn_channels: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, fpn_channels, kernel_size=1, bias=False)
        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bottom_up: torch.Tensor, top_down: torch.Tensor) -> torch.Tensor:
        lateral = self.lateral(bottom_up)
        top_down_up = F.interpolate(top_down, size=lateral.shape[-2:], mode="nearest")
        return self.smooth(lateral + top_down_up)


class FPNInceptentionNet(nn.Module):
    """FPN-InceptentionNet: multi-scale variant of InceptentionNet.

    Architecture overview:
    - Stem: 3×3 conv (stride 1).
    - Bottom-up backbone: three Inception stages each with 2× downsampling,
      producing C3, C4, C5.
    - Top-down FPN with lateral connections: C5 → P5, merge with C4 → P4,
      merge with C3 → P3.
    - Scale-aware self-attention at each pyramid level after pooling to a
      fixed spatial size (``attn_spatial × attn_spatial``).
    - Classification: GAP per scale → concatenate → BN + ReLU → Dropout →
      Linear → scalar logit.

    Args:
        stem_channels: Number of output channels for the stem convolution.
        branch_channels: Channels per branch inside each Inception stage.
            The stage output has ``branch_channels * 4`` channels.
        fpn_channels: Uniform channel width used throughout the FPN.
        attn_spatial: Spatial size (H = W) to which each FPN feature map is
            pooled before self-attention.  Defaults to 7 (49 tokens per level).
        num_heads: Number of attention heads.
        dropout: Dropout probability in the classification head.
    """

    def __init__(
        self,
        stem_channels: int = 64,
        branch_channels: int = 64,
        fpn_channels: int = 256,
        attn_spatial: int = 7,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        backbone_out = branch_channels * 4

        # Stem
        self.stem = ConvBnRelu(3, stem_channels, kernel_size=3, stride=1, padding=1)

        # Bottom-up backbone: three 2× stages
        self.stage1 = _InceptionStage(stem_channels, branch_channels)   # → C3
        self.stage2 = _InceptionStage(backbone_out, branch_channels)     # → C4
        self.stage3 = _InceptionStage(backbone_out, branch_channels)     # → C5

        # Top-down FPN
        self.lateral5 = nn.Sequential(
            nn.Conv2d(backbone_out, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_merge4 = _FPNMerge(backbone_out, fpn_channels)  # C4 + ↑P5 → P4
        self.fpn_merge3 = _FPNMerge(backbone_out, fpn_channels)  # C3 + ↑P4 → P3

        # Pool each pyramid level to a fixed spatial size before attention
        self.pool_spatial = nn.AdaptiveAvgPool2d((attn_spatial, attn_spatial))

        # Scale-aware self-attention (shared weights across levels)
        self.attention3 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)
        self.attention4 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)
        self.attention5 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        fused_dim = fpn_channels * 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        # Top-down FPN
        p5 = self.lateral5(c5)
        p4 = self.fpn_merge4(c4, p5)
        p3 = self.fpn_merge3(c3, p4)

        # Pool to fixed spatial size, apply self-attention, then GAP
        p3_attn = self.gap(self.attention3(self.pool_spatial(p3)))
        p4_attn = self.gap(self.attention4(self.pool_spatial(p4)))
        p5_attn = self.gap(self.attention5(self.pool_spatial(p5)))

        # Fuse across scales
        fused = torch.cat([p3_attn, p4_attn, p5_attn], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(1)
