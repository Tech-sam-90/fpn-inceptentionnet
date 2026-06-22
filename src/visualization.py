from __future__ import annotations

"""Visualization utilities: Grad-CAM and attention heatmaps.

Usage example::

    from visualization import GradCAM, AttentionRollout
    import torch
    from PIL import Image

    # --- Grad-CAM ---
    model = ...  # trained InceptentionNet or FPNInceptentionNet
    cam = GradCAM(model, target_layer=model.inception)
    image_tensor = ...  # shape (1, 3, H, W)
    heatmap = cam(image_tensor)          # numpy array (H, W) in [0, 1]
    overlay = cam.overlay(image_tensor, heatmap)  # PIL Image

    # --- Attention heatmap ---
    rollout = AttentionRollout(model.attention)
    _ = model(image_tensor)              # forward pass populates hook
    attn_map = rollout()                 # numpy array (H, W) in [0, 1]
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single-image tensor (C, H, W) in [-1, 1] normalisation to PIL."""
    img = tensor.detach().cpu().float()
    img = img * 0.5 + 0.5  # undo [-1,1] normalisation
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Registers forward and backward hooks on *target_layer*.  Calling the
    instance with an image tensor returns a spatial heatmap showing which
    regions contributed most to the positive-class logit.

    Args:
        model: Trained classification model with a scalar sigmoid logit output.
        target_layer: The ``nn.Module`` whose output activations are used.
            For ``InceptentionNet`` a good choice is ``model.inception``.
            For ``FPNInceptentionNet`` try ``model.fpn_merge3`` (finest scale).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        def _forward_hook(module, input, output):  # noqa: ARG001
            self._activations = output.detach()

        def _backward_hook(module, grad_input, grad_output):  # noqa: ARG001
            self._gradients = grad_output[0].detach()

        self._fwd_handle = target_layer.register_forward_hook(_forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """Return a Grad-CAM heatmap for the input image.

        Args:
            image: Tensor of shape ``(1, C, H, W)`` on any device.

        Returns:
            NumPy array of shape ``(H_orig, W_orig)`` with values in ``[0, 1]``.
        """
        self.model.eval()
        image = image.requires_grad_(True)

        logit = self.model(image)
        self.model.zero_grad()
        logit.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not fire.  Check that target_layer is part of the forward pass.")

        # Global-average-pool the gradients over the spatial dimensions
        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Resize to input spatial size
        h_in, w_in = image.shape[-2], image.shape[-1]
        cam = torch.nn.functional.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def overlay(
        self,
        image: torch.Tensor,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Image.Image:
        """Overlay a Grad-CAM heatmap on the original image.

        Args:
            image: Input tensor ``(1, C, H, W)``.
            heatmap: Output of :meth:`__call__`, shape ``(H, W)`` in ``[0, 1]``.
            alpha: Blending weight for the heatmap (0 = original, 1 = heatmap).
            colormap: Optional function ``(H, W) → (H, W, 3)`` uint8 array.
                Defaults to a simple red-channel heat colormap.

        Returns:
            PIL ``Image`` with the heatmap blended over the original image.
        """
        base = _to_pil(image.squeeze(0))
        h, w = heatmap.shape

        if colormap is None:
            heat_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            heat_rgb[:, :, 0] = (heatmap * 255).astype(np.uint8)
        else:
            heat_rgb = colormap(heatmap)

        heat_img = Image.fromarray(heat_rgb).resize(base.size, Image.BILINEAR)
        blended = Image.blend(base, heat_img, alpha=alpha)
        return blended


class AttentionRollout:
    """Visualise which spatial tokens the self-attention module attends to.

    Registers a forward hook on a :class:`~models.attention.SelfAttention2D`
    module (or any module that exposes an ``attention`` child which is
    ``nn.MultiheadAttention``).  Returns the mean attention weight over all
    heads, reshaped to the spatial grid of the feature map.

    Args:
        attention_module: A ``SelfAttention2D`` instance.
    """

    def __init__(self, attention_module: nn.Module) -> None:
        self._attn_weights: torch.Tensor | None = None
        self._spatial: tuple[int, int] | None = None

        mha: nn.MultiheadAttention = attention_module.attention

        def _hook(module, args, kwargs, output):  # noqa: ARG001
            # output[1] is the attention weight tensor (B, L, L)
            self._attn_weights = output[1].detach()

        self._handle = mha.register_forward_hook(_hook, with_kwargs=True)

    def remove_hook(self) -> None:
        self._handle.remove()

    def __call__(self) -> np.ndarray:
        """Return the mean spatial attention map from the last forward pass.

        You must run a forward pass through the model **before** calling this
        method so that the hook has recorded the attention weights.

        Returns:
            NumPy array of shape ``(H_attn, W_attn)`` with values in ``[0, 1]``.
        """
        if self._attn_weights is None:
            raise RuntimeError(
                "No attention weights recorded.  Run a forward pass on the model first."
            )
        attn = self._attn_weights  # (B, L, L)
        # Mean over heads is already done by MultiheadAttention when
        # need_weights=True (default) and average_attn_weights=True (default).
        # attn shape: (B, L, L)
        mean_attn = attn[0].mean(dim=0).cpu().numpy()  # (L,) — mean over query dim

        side = int(round(mean_attn.shape[0] ** 0.5))
        if side * side != mean_attn.shape[0]:
            raise ValueError(
                f"Cannot reshape attention of length {mean_attn.shape[0]} to a square grid."
            )

        attn_map = mean_attn.reshape(side, side).astype(np.float32)
        attn_min, attn_max = attn_map.min(), attn_map.max()
        if attn_max > attn_min:
            attn_map = (attn_map - attn_min) / (attn_max - attn_min)
        return attn_map
