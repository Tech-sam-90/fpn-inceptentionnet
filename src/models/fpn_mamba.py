"""
FPN-Mamba: EfficientNet-B2 backbone + Feature Pyramid Network with
Bidirectional Mamba LocalityMixing + Cross-Scale Mamba + GeM + SE head.

Architecture follows Liang et al. (Neural Networks 2026) for the FPN-Mamba
design, applied to binary medulloblastoma classification.

Ablation flags allow progressive component knock-out:
  use_fpn             -- if False: use C5 only with GAP (backbone-only baseline)
  use_locality_mixing -- if False: FPN uses standard 3x3 conv (no Mamba in neck)
  use_cross_mamba     -- if False: skip cross-scale Mamba scan
  use_gem             -- if False: replace GeM with standard AvgPool
  use_se              -- if False: skip SE channel attention
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

IMG_SIZE = 224


# ── Selective State Space Model ───────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective SSM. Input/Output: (B, L, D)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.x_proj(x)
        B_ssm = xz[..., : self.d_state]
        C_ssm = xz[..., self.d_state : self.d_state * 2]
        z = xz[..., self.d_state * 2 :]
        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log.float())
        ctx = torch.tanh(
            torch.einsum("bld,dn->bln", delta, A.mean(-1, keepdim=True).expand(-1, 1))
            + B_ssm.mean(-1, keepdim=True)
        )
        y = x * self.D[None, None, :] + ctx * C_ssm.mean(-1, keepdim=True) * x
        return y * torch.sigmoid(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_ = self.conv1d(x_.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        x_ = F.silu(x_)
        y = self._ssm_scan(x_)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


class BidirectionalMambaBlock(nn.Module):
    """
    Eq. 8 (Liang et al. 2026):
      T_Lf = SSM_F(T_{L-1})
      T_Lb = SSM_B(flip(T_{L-1}))
      T_L  = norm(T_Lf + flip(T_Lb))
    Input/Output: (B, L, D)
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.ssm_fwd(x)
        bwd = self.ssm_bwd(x.flip(1)).flip(1)
        return self.norm(fwd + bwd)


# ── LocalityMixing ────────────────────────────────────────────────────────────

class LocalityMixing(nn.Module):
    """
    Eq. 9 (Liang et al. 2026):
      F_local  = DWConv3x3(X)
      F_global = Mamba(F_local)
      W        = sigmoid(Linear(F_local))
      F_out    = W * F_local + (1-W) * F_global
    Input/Output: (B, C, H, W)
    """

    def __init__(self, channels: int, d_state: int = 16) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.mamba = BidirectionalMambaBlock(channels, d_state)
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
    """Standard 3×3 smoothing conv used in the no-Mamba ablation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── FPN Neck ──────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    """
    FPN top-down pathway.
    use_locality_mixing=True  → LocalityMixing at every level (MambaFPN)
    use_locality_mixing=False → standard 3×3 conv (plain FPN ablation)
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        d_state: int = 16,
        use_locality_mixing: bool = True,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        if use_locality_mixing:
            self.fusion = nn.ModuleList([LocalityMixing(out_channels, d_state) for _ in in_channels])
        else:
            self.fusion = nn.ModuleList([StandardFPNBlock(out_channels) for _ in in_channels])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        return [fuse(lat) for fuse, lat in zip(self.fusion, laterals)]


# ── Pooling and Attention ─────────────────────────────────────────────────────

class GeM(nn.Module):
    """Generalized Mean Pooling (Radenovic et al. 2019). Learned p starts at 3."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

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


# ── Full Classifier ───────────────────────────────────────────────────────────

class FPNMambaClassifier(nn.Module):
    """
    EfficientNet-B2 → FPN (with optional LocalityMixing) →
    Cross-scale Mamba → GeM + SE → FC head.

    Ablation flags (all True = full model):
        use_fpn             -- False: backbone C5 only, no pyramid
        use_locality_mixing -- False: plain FPN 3×3 conv
        use_cross_mamba     -- False: skip cross-scale Mamba
        use_gem             -- False: AvgPool instead of GeM
        use_se              -- False: skip SE channel attention
    """

    def __init__(
        self,
        num_classes: int = 1,
        fpn_channels: int = 256,
        dropout: float = 0.3,
        d_state: int = 16,
        freeze_layers: int = 3,
        img_size: int = IMG_SIZE,
        use_fpn: bool = True,
        use_locality_mixing: bool = True,
        use_cross_mamba: bool = True,
        use_gem: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        self.use_fpn = use_fpn
        self.use_cross_mamba = use_cross_mamba
        self.use_gem = use_gem
        self.use_se = use_se

        # ── Backbone ──────────────────────────────────────────────────────────
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

        for i, child in enumerate(self.backbone.children()):
            if i < freeze_layers:
                for p in child.parameters():
                    p.requires_grad = False

        # ── FPN or backbone-only ───────────────────────────────────────────────
        if use_fpn:
            self.fpn = FPN(in_ch, fpn_channels, d_state, use_locality_mixing)
            n_pyramid_levels = len(in_ch)
            feat_dim = fpn_channels * n_pyramid_levels
        else:
            # Backbone-only: lateral project C5 only
            self.c5_proj = nn.Conv2d(in_ch[-1], fpn_channels, 1)
            n_pyramid_levels = 1
            feat_dim = fpn_channels

        # ── Cross-scale Mamba ─────────────────────────────────────────────────
        if use_fpn and use_cross_mamba:
            self.cross_mamba = BidirectionalMambaBlock(fpn_channels, d_state)
        else:
            self.cross_mamba = None

        # ── Pooling ───────────────────────────────────────────────────────────
        self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d((1, 1))

        # ── SE attention ──────────────────────────────────────────────────────
        self.se = SEBlock(feat_dim) if use_se else nn.Identity()

        # ── Classifier head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)

        if self.use_fpn:
            pyramid = self.fpn(feats)

            if self.cross_mamba is not None:
                seqs = [p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, p.shape[1]) for p in pyramid]
                self.cross_mamba(torch.cat(seqs, dim=1))

            pooled = torch.cat([self.pool(p).flatten(1) for p in pyramid], dim=1)
        else:
            c5 = self.c5_proj(feats[-1])
            pooled = self.pool(c5).flatten(1)

        pooled = self.se(pooled)
        return self.head(pooled)
