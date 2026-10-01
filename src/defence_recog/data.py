"""
PyTorch Dataset and augmentation pipeline for ASTRA VISION defence equipment images.
Implements domain-appropriate data augmentations for small datasets without vertical flips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from defence_recog.preprocess import IMAGENET_MEAN, IMAGENET_STD, load_and_sanitize_image


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Data augmentation pipeline tailored for aerial/ground defence equipment.
    Uses RandomResizedCrop, horizontal flipping, subtle affine rotation,
    ColorJitter, and normalization. Vertical flipping is excluded as military
    equipment has canonical upright orientations.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """
    Deterministic evaluation transforms for validation and testing splits:
    Resize shorter side to 256, center crop to 224, convert to tensor, normalize.
    """
    return T.Compose([
        T.Resize(int(image_size * (256.0 / 224.0))),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DefenceDataset(Dataset):
    """
    PyTorch Dataset loading defence images referenced in labels/split CSVs.
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        data_root: Union[str, Path] = "data",
        transform: Optional[Callable] = None,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        if label2id is None:
            unique_classes = sorted(self.df["category"].unique())
            self.label2id = {c: i for i, c in enumerate(unique_classes)}
        else:
            self.label2id = label2id

        self.id2label = {i: c for c, i in self.label2id.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        rel_path = row["file_name"]
        category = row["category"]
        full_path = self.data_root / rel_path

        # Safely load and convert to RGB
        img = load_and_sanitize_image(full_path)

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)

        label_id = self.label2id[category]
        return tensor, label_id, rel_path


def create_dataloaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "data",
    image_size: int = 224,
    batch_size: int = 8,
    label2id: Optional[Dict[str, int]] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and optional test splits.
    """
    train_ds = DefenceDataset(
        csv_file=train_csv,
        data_root=data_root,
        transform=get_train_transforms(image_size),
        label2id=label2id,
    )

    val_ds = DefenceDataset(
        csv_file=val_csv,
        data_root=data_root,
        transform=get_eval_transforms(image_size),
        label2id=label2id,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    if test_csv is not None and Path(test_csv).exists():
        test_ds = DefenceDataset(
            csv_file=test_csv,
            data_root=data_root,
            transform=get_eval_transforms(image_size),
            label2id=label2id,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
