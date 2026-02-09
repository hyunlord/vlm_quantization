from __future__ import annotations

import albumentations as A


def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Light augmentations that preserve semantic content."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def get_consistency_augmentation(image_size: int = 384) -> A.Compose:
    """Stronger augmentation for consistency loss target."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.ToGray(p=0.1),
    ])


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """Deterministic resize for validation."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
    ])
