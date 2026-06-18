from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def compute_flops(model: nn.Module, input_size: tuple[int, int, int] = (3, 224, 224)) -> dict[str, float]:
    """
    Estimates FLOPs using thop if available, falls back to a parameter-count
    proxy otherwise. Returns GFLOPs and MACs.
    """
    try:
        from thop import profile, clever_format
        dummy = torch.zeros(1, *input_size)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        return {
            "macs": float(macs),
            "gflops": float(macs) * 2 / 1e9,
            "macs_str": macs_str,
            "params_str": params_str,
        }
    except ImportError:
        params = count_parameters(model)
        return {
            "macs": float("nan"),
            "gflops": float("nan"),
            "macs_str": "thop not installed",
            "params_str": f"{params['total']:,}",
        }


def model_summary(model: nn.Module, input_size: tuple[int, int, int] = (3, 224, 224)) -> None:
    params = count_parameters(model)
    flops = compute_flops(model, input_size)
    print(f"Parameters : {params['total']:>12,}  (trainable={params['trainable']:,}  frozen={params['frozen']:,})")
    print(f"GFLOPs     : {flops['gflops']:>12.3f}  ({flops['macs_str']})")
