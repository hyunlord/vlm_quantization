from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor

from src.data.coco import CocoCaptionsDataset
from src.data.transforms import (
    get_consistency_augmentation,
    get_train_transforms,
    get_val_transforms,
)


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack tensors and handle optional weak/aug pixel_values."""
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image_ids": [b["image_id"] for b in batch],
    }
    if "weak_pixel_values" in batch[0]:
        result["weak_pixel_values"] = torch.stack(
            [b["weak_pixel_values"] for b in batch]
        )
    if "aug_pixel_values" in batch[0]:
        result["aug_pixel_values"] = torch.stack(
            [b["aug_pixel_values"] for b in batch]
        )
    return result


class CrossModalHashDataModule(pl.LightningDataModule):
    """Lightning DataModule for COCO Captions cross-modal hashing."""

    def __init__(
        self,
        data_root: str = "./data/coco",
        processor_name: str = "google/siglip2-so400m-patch14-384",
        batch_size: int = 128,
        num_workers: int = 4,
        max_text_length: int = 64,
        image_size: int = 384,
        karpathy_json: str | None = None,
        extra_datasets: list[dict] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root)
        self.processor = AutoProcessor.from_pretrained(processor_name)

    def _build_extra_datasets(self) -> list:
        """Build GenericImageTextDataset instances from extra_datasets config."""
        from src.data.generic import GenericImageTextDataset

        extra = []
        for ds_cfg in self.hparams.extra_datasets or []:
            extra.append(
                GenericImageTextDataset(
                    data_root=ds_cfg["data_root"],
                    jsonl_path=ds_cfg["jsonl_path"],
                    processor=self.processor,
                    transform=get_train_transforms(self.hparams.image_size),
                    consistency_transform=get_consistency_augmentation(
                        self.hparams.image_size
                    ),
                    max_text_length=self.hparams.max_text_length,
                    image_size=self.hparams.image_size,
                )
            )
        return extra

    def _concat_with_extra(self, base_train):
        """Wrap base training dataset with extra datasets via ConcatDataset."""
        extra = self._build_extra_datasets()
        if extra:
            return ConcatDataset([base_train] + extra)
        return base_train

    def setup(self, stage: str | None = None) -> None:
        karpathy = self.hparams.karpathy_json

        if karpathy:
            from src.data.karpathy import KarpathyCocoCaptionsDataset

            if stage in ("fit", "validate") or stage is None:
                base_train = KarpathyCocoCaptionsDataset(
                    data_root=self.data_root,
                    karpathy_json=karpathy,
                    split="train",
                    processor=self.processor,
                    transform=get_train_transforms(self.hparams.image_size),
                    consistency_transform=get_consistency_augmentation(
                        self.hparams.image_size
                    ),
                    max_text_length=self.hparams.max_text_length,
                    image_size=self.hparams.image_size,
                )
                self.train_dataset = self._concat_with_extra(base_train)
                self.val_dataset = KarpathyCocoCaptionsDataset(
                    data_root=self.data_root,
                    karpathy_json=karpathy,
                    split="test",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                )

            if stage == "test":
                self.test_dataset = KarpathyCocoCaptionsDataset(
                    data_root=self.data_root,
                    karpathy_json=karpathy,
                    split="test",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                )
        else:
            # Fallback: standard COCO 2014 splits
            if stage in ("fit", "validate") or stage is None:
                base_train = CocoCaptionsDataset(
                    image_dir=self.data_root / "train2014",
                    ann_file=self.data_root / "annotations" / "captions_train2014.json",
                    processor=self.processor,
                    transform=get_train_transforms(self.hparams.image_size),
                    consistency_transform=get_consistency_augmentation(
                        self.hparams.image_size
                    ),
                    max_text_length=self.hparams.max_text_length,
                    image_size=self.hparams.image_size,
                )
                self.train_dataset = self._concat_with_extra(base_train)
                self.val_dataset = CocoCaptionsDataset(
                    image_dir=self.data_root / "val2014",
                    ann_file=self.data_root / "annotations" / "captions_val2014.json",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                )

            if stage == "test":
                self.test_dataset = CocoCaptionsDataset(
                    image_dir=self.data_root / "val2014",
                    ann_file=self.data_root / "annotations" / "captions_val2014.json",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                )

    def _mp_context(self) -> dict:
        """Use 'spawn' to avoid all CUDA fork deadlocks, and persistent_workers to prevent semaphore leaks."""
        if self.hparams.num_workers > 0:
            return {"multiprocessing_context": "spawn", "persistent_workers": True}
        return {}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            **self._mp_context(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(self.hparams.num_workers, 4),
            collate_fn=collate_fn,
            pin_memory=True,
            **self._mp_context(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(self.hparams.num_workers, 4),
            collate_fn=collate_fn,
            pin_memory=True,
            **self._mp_context(),
        )
