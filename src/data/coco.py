from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import AutoProcessor


class CocoCaptionsDataset(Dataset):
    """COCO Captions dataset for cross-modal hashing.

    Each sample returns an image-caption pair.
    One caption is randomly sampled per image per call.
    """

    def __init__(
        self,
        image_dir: str | Path,
        ann_file: str | Path,
        processor: AutoProcessor,
        transform: A.Compose | None = None,
        consistency_transform: A.Compose | None = None,
        max_text_length: int = 64,
        image_size: int = 384,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.transform = transform
        self.consistency_transform = consistency_transform
        self.max_text_length = max_text_length
        self.image_size = image_size
        self._orig_resize = A.Resize(image_size, image_size)

        self.coco = COCO(str(ann_file))
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: int) -> np.ndarray:
        info = self.coco.imgs[image_id]
        path = self.image_dir / info["file_name"]
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)

        # Random caption selection
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = random.choice(anns)["caption"]

        # Original view (Resize only — clean anchor for contrastive loss)
        orig_resized = self._orig_resize(image=image)["image"]
        orig_inputs = self.processor(
            images=Image.fromarray(orig_resized), return_tensors="pt",
        )

        # Process text via tokenizer directly (SigLIP processor may omit attention_mask)
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
            "image_id": image_id,
        }

        if "attention_mask" in text_inputs:
            result["attention_mask"] = text_inputs["attention_mask"].squeeze(0)
        else:
            result["attention_mask"] = torch.ones_like(result["input_ids"])

        # Weak augmentation (RandomCrop, Flip, ColorJitter, GaussianBlur)
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
