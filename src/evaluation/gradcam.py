"""
Grad-CAM visualization for FPN-Mamba pyramid levels and InceptentionNet.

Usage
-----
    cam = GradCAM(model, target_layer=model.fpn.fusion[0])
    heatmap = cam(image_tensor)          # (H, W) numpy array in [0, 1]
    overlay = cam.overlay(image_pil, heatmap)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, x: torch.Tensor, smooth: bool = True) -> np.ndarray:
        """
        x: (1, C, H, W) input tensor (single image).
        Returns heatmap (H_orig, W_orig) in [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits.squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            raise RuntimeError("No gradients captured — check that target_layer is on the forward path.")

        # Global average pool the gradients: (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        # Resize to input spatial size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

    @staticmethod
    def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """Blend a PIL image with a heatmap (numpy [0,1] array)."""
        import matplotlib.cm as cm
        h, w = heatmap.shape
        img_resized = image.resize((w, h)).convert("RGB")
        colormap = cm.get_cmap("jet")(heatmap)[..., :3]
        colormap_img = Image.fromarray((colormap * 255).astype(np.uint8))
        return Image.blend(img_resized, colormap_img, alpha=alpha)


class MultiLevelGradCAM:
    """
    Runs GradCAM on each FPN pyramid level independently and returns
    one heatmap per level, enabling visualisation of multi-scale attention.
    """

    def __init__(self, model: nn.Module, pyramid_layers: list[nn.Module]) -> None:
        self.cams = [GradCAM(model, layer) for layer in pyramid_layers]

    def __call__(self, x: torch.Tensor) -> list[np.ndarray]:
        heatmaps = []
        for cam in self.cams:
            heatmaps.append(cam(x.clone()))
        return heatmaps

    def remove_hooks(self) -> None:
        for cam in self.cams:
            cam.remove_hooks()


def get_fpn_target_layers(model) -> list[nn.Module]:
    """Returns the LocalityMixing (or StandardFPNBlock) at each FPN level."""
    return list(model.fpn.fusion)
