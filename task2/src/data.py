from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from .config import DEFAULT_SEED, NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    train_size: int
    val_size: int
    test_size: int
    label_offset: int


class LabelAdjustedDataset(Dataset):
    def __init__(self, base: Dataset, label_offset: int) -> None:
        self.base = base
        self.label_offset = label_offset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return image, int(target) - self.label_offset


def _extract_raw_targets(dataset: Dataset) -> List[int]:
    for attr in ("_labels", "labels", "targets", "_targets"):
        raw = getattr(dataset, attr, None)
        if raw is None:
            continue
        return [int(x) for x in raw]
    targets: List[int] = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        targets.append(int(target))
    return targets


def _infer_label_offset(targets: Sequence[int]) -> int:
    if not targets:
        return 0
    min_label = int(min(targets))
    max_label = int(max(targets))
    if min_label == 1 and max_label == NUM_CLASSES:
        return 1
    return 0


def _breed_name_from_path(path_like: str) -> str:
    stem = Path(path_like).stem
    stem = re.sub(r"_\d+$", "", stem)
    return stem.replace("_", " ")


def _extract_class_names(dataset: Dataset, label_offset: int) -> List[str]:
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, list) and len(classes) == NUM_CLASSES:
        return [str(name).replace("_", " ") for name in classes]

    image_paths = getattr(dataset, "_images", None)
    raw_targets = _extract_raw_targets(dataset)
    if image_paths is not None and len(image_paths) == len(raw_targets):
        mapping = {}
        for image_path, raw_label in zip(image_paths, raw_targets):
            label = int(raw_label) - label_offset
            mapping.setdefault(label, _breed_name_from_path(str(image_path)))
        if len(mapping) == NUM_CLASSES:
            return [mapping[idx] for idx in range(NUM_CLASSES)]

    return [f"class_{idx}" for idx in range(NUM_CLASSES)]


def _build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    data_root: str,
    *,
    image_size: int,
    batch_size: int,
    seed: int = DEFAULT_SEED,
    num_workers: int = 2,
) -> DatasetBundle:
    train_transform, eval_transform = _build_transforms(image_size)
    base_root = Path(data_root)
    raw_trainval = OxfordIIITPet(root=base_root, split="trainval", target_types="category", download=True)
    raw_targets = _extract_raw_targets(raw_trainval)
    label_offset = _infer_label_offset(raw_targets)
    class_names = _extract_class_names(raw_trainval, label_offset)

    train_dataset = LabelAdjustedDataset(
        OxfordIIITPet(
            root=base_root,
            split="trainval",
            target_types="category",
            transform=train_transform,
            download=False,
        ),
        label_offset=label_offset,
    )
    eval_trainval = LabelAdjustedDataset(
        OxfordIIITPet(
            root=base_root,
            split="trainval",
            target_types="category",
            transform=eval_transform,
            download=False,
        ),
        label_offset=label_offset,
    )
    test_dataset = LabelAdjustedDataset(
        OxfordIIITPet(
            root=base_root,
            split="test",
            target_types="category",
            transform=eval_transform,
            download=True,
        ),
        label_offset=label_offset,
    )

    indices = np.arange(len(raw_trainval))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_indices = indices[:split].tolist()
    val_indices = indices[split:].tolist()

    pin_memory = False
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(eval_trainval, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_dataset),
        label_offset=label_offset,
    )
