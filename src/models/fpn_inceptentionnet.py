from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention2D
from models.inceptentionnet import ConvBnRelu, ModifiedInceptionBlock


class LateralConnection(nn.Module):
    """FPN lateral connection: 1×1 projection plus optional top-down addition and 3×3 smoothing."""

    def __init__(self, in_channels: int, fpn_channels: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, fpn_channels, kernel_size=1, bias=False)
        self.smooth = ConvBnRelu(fpn_channels, fpn_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, top_down: torch.Tensor | None = None) -> torch.Tensor:
        x = self.lateral(x)
        if top_down is not None:
            top_down_up = F.interpolate(top_down, size=x.shape[-2:], mode="nearest")
            x = x + top_down_up
        return self.smooth(x)


class FPNInceptentionNet(nn.Module):
    """FPN-InceptentionNet: multi-scale extension of InceptentionNet.

    Architecture overview:
    1. Bottom-up backbone: stem + three Inception blocks producing C3, C4, C5 at
       progressively coarser spatial resolutions.
    2. Top-down FPN with lateral connections: C5 → P5; upsample(P5)+C4 → P4;
       upsample(P4)+C3 → P3.  Each merge is followed by a 3×3 smoothing conv.
    3. Scale-aware self-attention applied to each pyramid level P3, P4, P5.
       P3 is spatially pooled to match P4 before attention to keep the token
       sequence tractable (image-level classification does not require full
       spatial resolution at P3).
    4. Global average pooling per scale, concatenation, dense head with batch
       normalisation, ReLU, dropout, and a single sigmoid output.
    """

    def __init__(
        self,
        stem_channels: int = 64,
        branch_channels: int = 64,
        fpn_channels: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        backbone_channels = branch_channels * 4  # 256 with default settings

        # Bottom-up backbone
        self.stem = ConvBnRelu(3, stem_channels, kernel_size=3, stride=1, padding=1)
        self.inception1 = ModifiedInceptionBlock(stem_channels, branch_channels=branch_channels)  # → C3
        self.inception2 = ModifiedInceptionBlock(backbone_channels, branch_channels=branch_channels)  # → C4
        self.inception3 = ModifiedInceptionBlock(backbone_channels, branch_channels=branch_channels)  # → C5

        # FPN lateral connections (top-down pathway)
        self.fpn5 = LateralConnection(backbone_channels, fpn_channels)
        self.fpn4 = LateralConnection(backbone_channels, fpn_channels)
        self.fpn3 = LateralConnection(backbone_channels, fpn_channels)

        # Scale-aware self-attention at each pyramid level
        self.attention5 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)
        self.attention4 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)
        self.attention3 = SelfAttention2D(embed_dim=fpn_channels, num_heads=num_heads)

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        total_features = fpn_channels * 3  # three pyramid scales
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up
        x = self.stem(x)
        c3 = self.inception1(x)
        c4 = self.inception2(c3)
        c5 = self.inception3(c4)

        # Top-down FPN with lateral connections
        p5 = self.fpn5(c5)
        p4 = self.fpn4(c4, p5)
        p3 = self.fpn3(c3, p4)

        # Scale-aware self-attention
        # P5 and P4 are processed at their native resolution.
        # P3 is spatially downsampled to match P4 before attention so that the
        # token sequence length stays tractable for classification tasks.
        p5 = self.attention5(p5)
        p4 = self.attention4(p4)
        p3_pooled = F.adaptive_avg_pool2d(p3, p4.shape[-2:])
        p3 = self.attention3(p3_pooled)

        # Global average pooling per scale, then concatenate and classify
        f5 = self.pool(p5)
        f4 = self.pool(p4)
        f3 = self.pool(p3)

        features = torch.cat([f5, f4, f3], dim=1)
        logits = self.classifier(features)
        return logits.squeeze(1)
