"""
Generic timm-based binary classifier for additional baseline comparisons.
Loads any timm model (pretrained) and attaches a lightweight FC head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class TimmClassifier(nn.Module):
    """
    Wraps any timm backbone with a binary classification head.

    Args:
        model_name:    timm model name (e.g. 'resnet50', 'vit_small_patch16_224')
        pretrained:    load ImageNet weights
        dropout:       dropout rate in the FC head
        freeze_layers: number of top-level backbone children to freeze
        hidden_dim:    width of the single hidden FC layer
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_layers: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        assert TIMM_AVAILABLE, "timm is not installed — run: pip install timm"

        # num_classes=0 + global_pool='avg' → backbone returns (B, num_features)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        if freeze_layers > 0:
            children = list(self.backbone.children())
            for child in children[:freeze_layers]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, feat_dim)
        return self.head(features).squeeze(1)  # (B,)
