"""Kernel-fused FPNMambaClassifier, with BidirectionalMambaBlock's naive chunked-scan SelectiveSSM
replaced by the official mamba-ssm CUDA kernel (mamba_ssm.Mamba). Self-contained (no import from
src/models) so it can run under the isolated venv (~/venv_mamba_kernel, torch 2.5.1) without
touching the main project venv (torch 2.12.0). Only the 3 Mamba-using ablation variants need this
file -- efficientnet_only and fpn_standard have no Mamba component and are trained via the main venv.

Diverges from src/models/fpn_mamba.py in one respect (a correctness fix, not a stylistic change):
cross-scale Mamba fusion (Eq. 7, Liang et al. 2026) is now the PRIMARY per-level fuse mechanism
inside FPN, with LocalityMixing (Eq. 9) applied afterward as a secondary per-level refinement --
matching the paper's own ablation evidence (Table 5/6: the joint cross-scale fusion drives the
large gains, Locality Mixing contributes <0.5 AP on top). The previous version had this backwards
(LocalityMixing as the primary per-level fuse, cross-mamba bolted on afterward).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from mamba_ssm import Mamba

IMG_SIZE = 256


class KernelBidirectionalMamba(nn.Module):
    """Same interface/semantics as fpn_mamba.BidirectionalMambaBlock (Eq. 8, Liang et al. 2026),
    backed by the fused mamba-ssm CUDA kernel instead of the naive chunked scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.ssm_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.ssm_fwd(x)
        bwd = self.ssm_bwd(x.flip(1)).flip(1)
        return self.norm(fwd + bwd)


class LocalityMixing(nn.Module):
    def __init__(self, channels: int, d_state: int = 16) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.mamba = KernelBidirectionalMamba(channels, d_state)
        self.gate_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f_local = F.silu(self.dw_conv(x))
        seq = f_local.permute(0, 2, 3, 1).reshape(B, H * W, C)
        f_global = self.mamba(seq)
        f_global = f_global.reshape(B, H, W, C).permute(0, 3, 1, 2)
        w = torch.sigmoid(self.gate_proj(seq)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return w * f_local + (1 - w) * f_global


class StandardFPNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN(nn.Module):
    """
    Lateral + top-down pathway (Eq. 5, Liang et al. 2026), then:
      - Eq. 7 (primary fuse): if use_cross_mamba, a single joint bidirectional Mamba pass over the
        Naive-Concat serialization of ALL levels replaces the per-level smoothing conv -- this is
        the mechanism the paper's own ablation (Table 5/6) shows drives the large gains, applied
        BEFORE any per-level refinement (previously this repo had it backwards: LocalityMixing ran
        first, per level, with cross-scale mamba bolted on afterward -- corrected here).
      - Eq. 9 (secondary refinement): if use_locality_mixing, each fused level is further refined by
        DWConv + a per-level Mamba branch, gated -- applied AFTER the primary fuse step, matching the
        paper's Table 5 ablation structure (Locality Mixing toggled on top of an already-fused
        pyramid, whatever fused it).
    """

    def __init__(self, in_channels, out_channels=256, d_state=16, use_cross_mamba=True, use_locality_mixing=True):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.use_cross_mamba = use_cross_mamba
        if use_cross_mamba:
            self.cross_mamba = KernelBidirectionalMamba(out_channels, d_state)
        else:
            self.fuse_conv = nn.ModuleList([StandardFPNBlock(out_channels) for _ in in_channels])
        self.use_locality_mixing = use_locality_mixing
        if use_locality_mixing:
            self.locality = nn.ModuleList([LocalityMixing(out_channels, d_state) for _ in in_channels])

    def forward(self, features):
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up

        if self.use_cross_mamba:
            seqs = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, p.shape[1]) for p in laterals]
            cross_out = self.cross_mamba(torch.cat(seqs, dim=1))
            sizes = [p.shape[2] * p.shape[3] for p in laterals]
            parts = cross_out.split(sizes, dim=1)
            fused = [
                part.reshape(p.shape[0], p.shape[2], p.shape[3], p.shape[1]).permute(0, 3, 1, 2)
                for part, p in zip(parts, laterals)
            ]
        else:
            fused = [conv(lat) for conv, lat in zip(self.fuse_conv, laterals)]

        if self.use_locality_mixing:
            fused = [lm(f) for lm, f in zip(self.locality, fused)]

        return fused


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class FPNMambaClassifier(nn.Module):
    """
    use_topk_head=True replaces GeM+SE+MLP-head with a dense per-position score (1x1 conv per
    pyramid level, no pooling first) + top-k pooling over all positions across all levels. This
    targets a task mismatch: MambaFPN was validated on per-pixel/per-region prediction (detection,
    segmentation), where a tiny object only has to fire locally; global pooling-then-classify (the
    GeM path) forces a tiny lesion's signal to survive being averaged with the rest of the slice
    before any decision is made. Top-k lets the strongest few positions drive the decision directly,
    the rest of the (mostly background) slice contributes nothing. use_gem/use_se are ignored when
    this is enabled.
    """

    def __init__(
        self, num_classes=1, fpn_channels=256, dropout=0.3, d_state=16, freeze_stages=2,
        img_size=IMG_SIZE, use_fpn=True, use_locality_mixing=True, use_cross_mamba=True,
        use_gem=True, use_se=True, use_topk_head=False, topk=10,
    ) -> None:
        super().__init__()
        self.use_fpn = use_fpn
        self.use_gem = use_gem
        self.use_se = use_se
        self.use_topk_head = use_topk_head
        self.topk = topk

        _probe = timm.create_model("efficientnet_b2", pretrained=False, features_only=True)
        with torch.no_grad():
            _feats = _probe(torch.zeros(1, 3, img_size, img_size))
        n_levels = len(_feats)
        out_indices = tuple(range(n_levels - 4, n_levels))

        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=True, features_only=True, out_indices=out_indices
        )
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, img_size, img_size))
            in_ch = [f.shape[1] for f in feats]

        for p in self.backbone.conv_stem.parameters():
            p.requires_grad = False
        for p in self.backbone.bn1.parameters():
            p.requires_grad = False
        for i, stage in enumerate(self.backbone.blocks.children()):
            if i < freeze_stages:
                for p in stage.parameters():
                    p.requires_grad = False

        if use_fpn:
            self.fpn = FPN(in_ch, fpn_channels, d_state, use_cross_mamba, use_locality_mixing)
            n_pyramid_levels = len(in_ch)
            feat_dim = fpn_channels * n_pyramid_levels
        else:
            self.c5_proj = nn.Conv2d(in_ch[-1], fpn_channels, 1)
            n_pyramid_levels = 1
            feat_dim = fpn_channels

        if use_topk_head:
            # one shared 1x1 conv scores every position on every pyramid level directly --
            # no pooling before this point. num_classes must be 1 (raw logit per position).
            assert num_classes == 1, "use_topk_head only supports binary (num_classes=1)"
            self.score_conv = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        else:
            self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d((1, 1))
            self.se = SEBlock(feat_dim) if use_se else nn.Identity()
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """return_aux=True (only meaningful with use_topk_head=True) additionally returns the raw,
        unflattened per-level score maps (list of (B,1,H_i,W_i)) for auxiliary segmentation
        supervision -- lets the per-position classifier be told directly WHERE the lesion is,
        instead of only learning it indirectly through the pooled whole-slice label."""
        feats = self.backbone(x)
        if self.use_fpn:
            pyramid = self.fpn(feats)  # cross-scale fuse (Eq. 7) + locality refinement (Eq. 9) done inside FPN
            if self.use_topk_head:
                score_maps = [self.score_conv(p) for p in pyramid]  # each (B, 1, H_i, W_i)
                scores = [s.flatten(1) for s in score_maps]  # each (B, H_i*W_i)
                all_scores = torch.cat(scores, dim=1)
                k = min(self.topk, all_scores.shape[1])
                topk_vals, _ = torch.topk(all_scores, k, dim=1)
                logit = topk_vals.mean(dim=1, keepdim=True)
                return (logit, score_maps) if return_aux else logit
            pooled = torch.cat([self.pool(p).flatten(1) for p in pyramid], dim=1)
        else:
            c5 = self.c5_proj(feats[-1])
            if self.use_topk_head:
                score_map = self.score_conv(c5)
                scores = score_map.flatten(1)
                k = min(self.topk, scores.shape[1])
                topk_vals, _ = torch.topk(scores, k, dim=1)
                logit = topk_vals.mean(dim=1, keepdim=True)
                return (logit, [score_map]) if return_aux else logit
            pooled = self.pool(c5).flatten(1)
        pooled = self.se(pooled)
        return self.head(pooled)


ABLATION_VARIANTS: dict[str, dict] = {
    "fpn_locality": dict(use_fpn=True, use_locality_mixing=True, use_cross_mamba=False, use_gem=False, use_se=False),
    "fpn_cross_mamba": dict(use_fpn=True, use_locality_mixing=True, use_cross_mamba=True, use_gem=False, use_se=False),
    "fpn_mamba_full": dict(use_fpn=True, use_locality_mixing=True, use_cross_mamba=True, use_gem=True, use_se=True),
    "fpn_mamba_full_topk": dict(use_fpn=True, use_locality_mixing=True, use_cross_mamba=True,
                                 use_gem=True, use_se=True, use_topk_head=True, topk=10),
}


def build_kernel_model(variant: str, **kwargs) -> FPNMambaClassifier:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown kernel variant '{variant}'. Choose from: {list(ABLATION_VARIANTS)}")
    return FPNMambaClassifier(**ABLATION_VARIANTS[variant], **kwargs)
