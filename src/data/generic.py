from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class GenericImageTextDataset(Dataset):
    """Generic image-text dataset reading from JSONL files.

    Each line in the JSONL file must contain:
        {"image_path": "relative/path.jpg", "caption": "text"}
    or for multi-caption:
        {"image_path": "relative/path.jpg", "captions": ["text1", "text2"]}

    Images are loaded from data_root / image_path.
    """

    def __init__(
        self,
        data_root: str | Path,
        jsonl_path: str | Path,
        processor: AutoProcessor,
        transform: A.Compose | None = None,
        consistency_transform: A.Compose | None = None,
        max_text_length: int = 64,
        image_size: int = 384,
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.transform = transform
        self.consistency_transform = consistency_transform
        self.max_text_length = max_text_length
        self.image_size = image_size
        self._orig_resize = A.Resize(image_size, image_size)

        # Load JSONL into memory
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        self._entries: list[dict] = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line %d", i)

        self.image_ids = list(range(len(self._entries)))
        logger.info(
            "Loaded %d entries from %s", len(self._entries), jsonl_path.name
        )

    def __len__(self) -> int:
        return len(self._entries)

    def _get_caption(self, entry: dict) -> str:
        """Extract caption from entry (supports single or multi-caption)."""
        if "captions" in entry:
            return random.choice(entry["captions"])
        return entry["caption"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        entry = self._entries[idx]

        # Load image (with fallback on failure)
        image_path = self.data_root / entry["image_path"]
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            # Skip corrupt images — return next valid sample
            logger.warning("Failed to load image: %s", image_path)
            return self.__getitem__((idx + 1) % len(self))

        caption = self._get_caption(entry)

        # Original view (Resize only — clean anchor for contrastive loss)
        orig_resized = self._orig_resize(image=image)["image"]
        orig_inputs = self.processor(
            images=Image.fromarray(orig_resized), return_tensors="pt",
        )

        # Process text via tokenizer directly
        text_inputs = self.processor.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "pixel_values": orig_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "image_id": idx,
        }

        if "attention_mask" in text_inputs:
            result["attention_mask"] = text_inputs["attention_mask"].squeeze(0)
        else:
            result["attention_mask"] = torch.ones_like(result["input_ids"])

        # Weak augmentation
        if self.transform is not None:
            weak_aug = self.transform(image=image)["image"]
            weak_inputs = self.processor(
                images=Image.fromarray(weak_aug), return_tensors="pt",
            )
            result["weak_pixel_values"] = weak_inputs["pixel_values"].squeeze(0)

        # Strong augmentation (consistency target)
        if self.consistency_transform is not None:
            cons_aug = self.consistency_transform(image=image)["image"]
            cons_inputs = self.processor(
                images=Image.fromarray(cons_aug), return_tensors="pt",
            )
            result["aug_pixel_values"] = cons_inputs["pixel_values"].squeeze(0)

        return result
