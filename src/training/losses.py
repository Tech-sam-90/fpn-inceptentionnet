from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain binary cross-entropy with logits. Used for InceptentionNet baseline."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def weighted_bce_smooth(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Weighted BCE with label smoothing. Used for FPN-Mamba.
    pos_weight/2 keeps loss scale stable while still addressing class imbalance.
    label_smoothing prevents overconfidence on the small MB class.
    """
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + label_smoothing / 2
    weight = torch.tensor([pos_weight / 2], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


def compute_pos_weight(labels: list[int], scale: float = 1.0) -> float:
    n_neg = sum(1 for l in labels if l == 0)
    n_pos = sum(1 for l in labels if l == 1)
    return (n_neg / max(n_pos, 1)) * scale
