from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import DATASET_NAME, DEFAULT_SEED

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
    raw_train = CIFAR10(root=base_root, train=True, download=True)
    class_names = [str(name) for name in raw_train.classes]

    train_dataset = CIFAR10(
        root=base_root,
        train=True,
        transform=train_transform,
        download=False,
    )
    eval_train = CIFAR10(
        root=base_root,
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = CIFAR10(
        root=base_root,
        train=False,
        transform=eval_transform,
        download=True,
    )

    indices = np.arange(len(raw_train))
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
        Subset(eval_train, val_indices),
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
        label_offset=0,
    )
