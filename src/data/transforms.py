from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BASELINE_MEAN = (0.5, 0.5, 0.5)
BASELINE_STD = (0.5, 0.5, 0.5)


@dataclass
class TransformConfig:
    image_size: int = 224
    gaussian_sigma: float = 0.7
    imagenet_norm: bool = True
    mean: tuple = field(init=False)
    std: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.mean = IMAGENET_MEAN if self.imagenet_norm else BASELINE_MEAN
        self.std = IMAGENET_STD if self.imagenet_norm else BASELINE_STD


class GaussianBlurPIL:
    def __init__(self, radius: float = 0.7) -> None:
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class HistEq:
    def __call__(self, img):
        return ImageOps.equalize(img)


def build_train_transform(cfg: TransformConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        GaussianBlurPIL(cfg.gaussian_sigma),
        HistEq(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])


def build_mb_transform(cfg: TransformConfig) -> transforms.Compose:
    """Heavier augmentation for MB (minority) class to address class imbalance."""
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        GaussianBlurPIL(cfg.gaussian_sigma),
        HistEq(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.3), ratio=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])


def build_eval_transform(cfg: TransformConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        GaussianBlurPIL(cfg.gaussian_sigma),
        HistEq(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
