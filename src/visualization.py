from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def _min_max_normalize(array: np.ndarray) -> np.ndarray:
    """Normalise values to [0, 1]."""
    lo, hi = array.min(), array.max()
    if hi - lo < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lo) / (hi - lo)).astype(np.float32)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Computes a saliency map for a binary classifier by backpropagating through
    a target convolutional layer and weighting its activation maps by the
    corresponding gradients.

    Usage::

        cam = GradCAM(model, target_layer=model.inception)
        heatmap = cam(input_tensor)   # np.ndarray of shape (H, W) in [0, 1]
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activation(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Return a (H, W) Grad-CAM heatmap in [0, 1] for *x* (batch size 1)."""
        self.model.eval()
        self.model.zero_grad()

        logit = self.model(x)
        # For binary classification there is only one output; backprop through it.
        logit.sum().backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not fire. Verify that target_layer is part of the forward pass.")

        # Global-average-pool the gradients over spatial dimensions → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=False)  # (B, H, W)
        cam = torch.clamp(cam, min=0).squeeze(0).cpu().numpy()  # ReLU, (H, W)
        return _min_max_normalize(cam)

    def remove_hooks(self) -> None:
        """Remove registered forward/backward hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a normalised (H, W) *heatmap* onto *image* and return a PIL image.

    Parameters
    ----------
    image:
        Original RGB PIL image.
    heatmap:
        Normalised saliency map in [0, 1] of any spatial size; it will be
        resized to match *image*.
    alpha:
        Weight of the heatmap overlay (0 = original image, 1 = heatmap only).
    colormap:
        Matplotlib colormap name used to colourise the heatmap.
    """
    try:
        import matplotlib.cm as cm
    except ImportError as exc:
        raise ImportError("matplotlib is required for overlay_heatmap. Install it with: pip install matplotlib") from exc

    heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.BILINEAR)
    heatmap_array = np.array(heatmap_resized) / 255.0

    cmap = cm.get_cmap(colormap)
    coloured = np.uint8(cmap(heatmap_array)[:, :, :3] * 255)
    overlay = Image.fromarray(coloured).convert("RGB")

    blended = Image.blend(image.convert("RGB"), overlay, alpha=alpha)
    return blended


def get_attention_maps(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Extract raw attention weight tensors from all ``SelfAttention2D`` layers.

    Registers temporary hooks on every ``MultiheadAttention`` sub-module inside
    the model, runs a forward pass, and returns the collected attention weights.

    Parameters
    ----------
    model:
        An ``InceptentionNet`` or ``FPNInceptentionNet`` instance.
    x:
        Input tensor of shape ``(1, C, H, W)``.

    Returns
    -------
    list[torch.Tensor]
        Each element is an attention-weight tensor of shape
        ``(1, num_heads, N, N)`` where *N* is the number of spatial tokens.
    """
    attention_weights: list[torch.Tensor] = []
    hooks: list = []

    def _hook(_module, _input, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights) when
        # need_weights=True (the default).
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for hook in hooks:
        hook.remove()

    return attention_weights


def build_inference_transform(image_size: int = 224) -> Callable:
    """Return a deterministic eval transform (Gaussian blur → equalize → tensor)."""
    from transforms import BaselineTransformConfig, build_eval_transform

    cfg = BaselineTransformConfig(image_size=image_size)
    return build_eval_transform(cfg)


def visualize_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image_path: str,
    output_path: str | None = None,
    image_size: int = 224,
    device: torch.device | None = None,
) -> Image.Image:
    """Produce and optionally save a Grad-CAM overlay for a single image.

    Parameters
    ----------
    model:
        Trained model (``InceptentionNet`` or ``FPNInceptentionNet``).
    target_layer:
        Convolutional layer to hook (e.g. ``model.inception`` or
        ``model.inception3``).
    image_path:
        Path to the input image file.
    output_path:
        If provided, the overlay is saved here as a PNG.
    image_size:
        Spatial size used during training (default 224).
    device:
        Torch device; defaults to CPU.

    Returns
    -------
    PIL.Image.Image
        RGB overlay image.
    """
    if device is None:
        device = torch.device("cpu")

    transform = build_inference_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model.to(device), target_layer)
    try:
        heatmap = cam(x)
    finally:
        cam.remove_hooks()

    result = overlay_heatmap(image, heatmap)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

    return result
