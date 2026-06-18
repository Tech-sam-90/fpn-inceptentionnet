from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    path: str
    label: int


def _list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _deduplicate(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        digest = _file_md5(p)
        if digest not in seen:
            seen.add(digest)
            unique.append(p)
    return unique


def build_binary_samples(
    data_root: str,
    mb_class_name: str = "Meduloblastoma",
    mb_target_count: int = 131,
    non_mb_target_count: int = 630,
    seed: int = 42,
    deduplicate: bool = True,
) -> list[Sample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    mb_dir = root / mb_class_name
    if not mb_dir.exists():
        available = ", ".join(d.name for d in class_dirs)
        raise ValueError(f"MB class '{mb_class_name}' not found. Available: {available}")

    mb_images = sorted(_list_images(mb_dir))
    non_mb_images: list[Path] = []
    for d in class_dirs:
        if d.name != mb_class_name:
            non_mb_images.extend(_list_images(d))

    if deduplicate:
        mb_images = _deduplicate(mb_images)
        non_mb_images = _deduplicate(non_mb_images)

    rng = random.Random(seed)
    rng.shuffle(mb_images)
    rng.shuffle(non_mb_images)

    if len(mb_images) < mb_target_count:
        raise ValueError(f"Not enough MB images: need {mb_target_count}, have {len(mb_images)}")
    if len(non_mb_images) < non_mb_target_count:
        raise ValueError(f"Not enough non-MB images: need {non_mb_target_count}, have {len(non_mb_images)}")

    samples = [Sample(str(p), 1) for p in mb_images[:mb_target_count]]
    samples += [Sample(str(p), 0) for p in non_mb_images[:non_mb_target_count]]
    rng.shuffle(samples)
    return samples


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


class BrainTumorDataset(Dataset):
    """
    Unified dataset used for both InceptentionNet and FPN-Mamba.
    Optionally applies a separate heavier transform to MB (minority) images.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform,
        mb_transform=None,
        augmentation_factor: int = 1,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mb_transform = mb_transform
        self.augmentation_factor = augmentation_factor

    def __len__(self) -> int:
        return len(self.samples) * self.augmentation_factor

    def __getitem__(self, index: int):
        sample = self.samples[index % len(self.samples)]
        try:
            image = Image.open(sample.path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), 0)

        if sample.label == 1 and self.mb_transform is not None:
            tensor = self.mb_transform(image)
        else:
            tensor = self.transform(image)

        return tensor, torch.tensor(sample.label, dtype=torch.float32)


class FoldEvalDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            image = Image.open(sample.path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), 0)
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)
