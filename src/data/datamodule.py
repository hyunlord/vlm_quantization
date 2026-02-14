from __future__ import annotations

import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoProcessor, GemmaTokenizer, SiglipImageProcessor, SiglipProcessor

from src.data.coco import CocoCaptionsDataset
from src.data.transforms import (
    get_consistency_augmentation,
    get_train_transforms,
    get_val_transforms,
)


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack tensors and handle optional weak/aug pixel_values and aux text."""
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
    # Multi-caption auxiliary text (P1) — handle mixed batches from ConcatDataset
    # where Karpathy samples have aux_input_ids but extra dataset samples don't.
    # Fallback: duplicate primary caption for samples without a second caption.
    if any("aux_input_ids" in b for b in batch):
        result["aux_input_ids"] = torch.stack([
            b.get("aux_input_ids", b["input_ids"]) for b in batch
        ])
        result["aux_attention_mask"] = torch.stack([
            b.get("aux_attention_mask", b["attention_mask"]) for b in batch
        ])
    # Multi-hot labels for supervised hashing
    if "labels" in batch[0]:
        result["labels"] = torch.stack([b["labels"] for b in batch])
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
        subset_ratio: float = 1.0,
        num_captions: int = 1,
        text_dropout_prob: float = 0.0,
        instances_json: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root)
        try:
            self.processor = AutoProcessor.from_pretrained(processor_name)
        except AttributeError:
            # transformers tokenizer mapping bug: AutoTokenizer returns None
            # for siglip2 model type. Bypass by loading components directly.
            image_processor = SiglipImageProcessor.from_pretrained(processor_name)
            tokenizer = GemmaTokenizer.from_pretrained(processor_name)
            self.processor = SiglipProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )

    def _build_extra_datasets(self) -> list:
        """Build GenericImageTextDataset instances from extra_datasets config."""
        import logging

        from src.data.generic import GenericImageTextDataset

        log = logging.getLogger(__name__)
        extra = []
        for ds_cfg in self.hparams.extra_datasets or []:
            jsonl = Path(ds_cfg["jsonl_path"])
            if not jsonl.exists():
                log.warning("Extra dataset JSONL not found, skipping: %s", jsonl)
                continue
            extra.append(
                GenericImageTextDataset(
                    data_root=ds_cfg["data_root"],
                    jsonl_path=jsonl,
                    processor=self.processor,
                    transform=get_train_transforms(self.hparams.image_size),
                    consistency_transform=get_consistency_augmentation(
                        self.hparams.image_size
                    ),
                    max_text_length=self.hparams.max_text_length,
                    image_size=self.hparams.image_size,
                    text_dropout_prob=self.hparams.text_dropout_prob,
                )
            )
        return extra

    def _concat_with_extra(self, base_train):
        """Wrap base training dataset with extra datasets via ConcatDataset."""
        extra = self._build_extra_datasets()
        if extra:
            return ConcatDataset([base_train] + extra)
        return base_train

    def _maybe_subset(self, dataset):
        """Apply random subsampling if subset_ratio < 1.0."""
        ratio = self.hparams.subset_ratio
        if ratio >= 1.0:
            return dataset
        n = len(dataset)
        k = max(1, int(n * ratio))
        indices = random.sample(range(n), k)
        return Subset(dataset, indices)

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
                    num_captions=self.hparams.num_captions,
                    text_dropout_prob=self.hparams.text_dropout_prob,
                    instances_json=self.hparams.instances_json,
                )
                self.train_dataset = self._maybe_subset(
                    self._concat_with_extra(base_train)
                )
                self.val_dataset = KarpathyCocoCaptionsDataset(
                    data_root=self.data_root,
                    karpathy_json=karpathy,
                    split="test",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                    instances_json=self.hparams.instances_json,
                )

            if stage == "test":
                self.test_dataset = KarpathyCocoCaptionsDataset(
                    data_root=self.data_root,
                    karpathy_json=karpathy,
                    split="test",
                    processor=self.processor,
                    transform=get_val_transforms(self.hparams.image_size),
                    max_text_length=self.hparams.max_text_length,
                    instances_json=self.hparams.instances_json,
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
                self.train_dataset = self._maybe_subset(
                    self._concat_with_extra(base_train)
                )
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
