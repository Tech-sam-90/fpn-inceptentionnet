from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms


@dataclass
class BaselineTransformConfig:
    image_size: int = 224
    gaussian_sigma: float = 2.0
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)


class GaussianAndEqualize:
    def __init__(self, sigma: float = 2.0) -> None:
        self.sigma = sigma

    def __call__(self, image):
        image = image.filter(ImageFilter.GaussianBlur(radius=self.sigma))
        return ImageOps.equalize(image)


class QuadAugmentDatasetTransform:
    def __init__(self, image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.base = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                GaussianAndEqualize(sigma=2.0),
            ]
        )
        self.aug = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(0, 45)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            ]
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image, variant_index: int) -> torch.Tensor:
        image = self.base(image)
        if variant_index > 0:
            image = self.aug(image)
        return self.to_tensor(image)


def build_eval_transform(config: BaselineTransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            GaussianAndEqualize(sigma=config.gaussian_sigma),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
