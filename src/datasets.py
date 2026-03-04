from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    path: str
    label: int


def _list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as file:
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _deduplicate_by_hash(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        digest = _file_md5(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
    return unique


def build_binary_samples(
    data_root: str,
    mb_class_name: str = "Meduloblastoma",
    mb_target_count: int = 106,
    non_mb_target_count: int = 630,
    seed: int = 42,
    deduplicate: bool = True,
) -> list[Sample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    mb_dir = root / mb_class_name
    if not mb_dir.exists():
        available = ", ".join(d.name for d in class_dirs)
        raise ValueError(f"MB class '{mb_class_name}' not found. Available classes: {available}")

    mb_images = sorted(_list_images(mb_dir))
    non_mb_images: list[Path] = []
    for class_dir in class_dirs:
        if class_dir.name == mb_class_name:
            continue
        non_mb_images.extend(_list_images(class_dir))

    if deduplicate:
        mb_images = _deduplicate_by_hash(mb_images)
        non_mb_images = _deduplicate_by_hash(non_mb_images)

    rng = random.Random(seed)
    rng.shuffle(mb_images)
    rng.shuffle(non_mb_images)

    if len(mb_images) < mb_target_count:
        raise ValueError(f"Not enough MB images after filtering. Required={mb_target_count}, available={len(mb_images)}")
    if len(non_mb_images) < non_mb_target_count:
        raise ValueError(
            f"Not enough non-MB images after filtering. Required={non_mb_target_count}, available={len(non_mb_images)}"
        )

    selected_mb = mb_images[:mb_target_count]
    selected_non_mb = non_mb_images[:non_mb_target_count]

    samples = [Sample(path=str(path), label=1) for path in selected_mb]
    samples.extend(Sample(path=str(path), label=0) for path in selected_non_mb)
    rng.shuffle(samples)
    return samples


class FoldTrainDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, augmentation_factor: int = 4) -> None:
        self.samples = samples
        self.transform = transform
        self.augmentation_factor = augmentation_factor

    def __len__(self) -> int:
        return len(self.samples) * self.augmentation_factor

    def __getitem__(self, index: int):
        base_index = index % len(self.samples)
        variant_index = index // len(self.samples)
        sample = self.samples[base_index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image, variant_index=variant_index)
        return tensor, np.float32(sample.label)


class FoldEvalDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        return tensor, np.float32(sample.label)
