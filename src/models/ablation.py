"""
Ablation variant factory.

The five variants form an additive ladder — each row adds exactly one
component over the previous, so the metric delta is unambiguously attributable.

  Variant              use_fpn  use_locality_mixing  use_cross_mamba  use_gem  use_se
  ─────────────────────────────────────────────────────────────────────────────────────
  efficientnet_only    False    False                False            False    False
  fpn_standard         True     False                False            False    False
  fpn_locality         True     True                 False            False    False
  fpn_cross_mamba      True     True                 True             False    False
  fpn_mamba_full       True     True                 True             True     True
"""
from __future__ import annotations

from .fpn_mamba import FPNMambaClassifier

ABLATION_VARIANTS: dict[str, dict] = {
    "efficientnet_only": dict(
        use_fpn=False,
        use_locality_mixing=False,
        use_cross_mamba=False,
        use_gem=False,
        use_se=False,
    ),
    "fpn_standard": dict(
        use_fpn=True,
        use_locality_mixing=False,
        use_cross_mamba=False,
        use_gem=False,
        use_se=False,
    ),
    "fpn_locality": dict(
        use_fpn=True,
        use_locality_mixing=True,
        use_cross_mamba=False,
        use_gem=False,
        use_se=False,
    ),
    "fpn_cross_mamba": dict(
        use_fpn=True,
        use_locality_mixing=True,
        use_cross_mamba=True,
        use_gem=False,
        use_se=False,
    ),
    "fpn_mamba_full": dict(
        use_fpn=True,
        use_locality_mixing=True,
        use_cross_mamba=True,
        use_gem=True,
        use_se=True,
    ),
}


def build_ablation_model(variant: str, **kwargs) -> FPNMambaClassifier:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ABLATION_VARIANTS)}")
    flags = ABLATION_VARIANTS[variant]
    return FPNMambaClassifier(**flags, **kwargs)
