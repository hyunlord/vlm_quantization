from __future__ import annotations

import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class _KarpathyCompat:
    """Minimal shim mimicking pycocotools.coco.COCO for MonitorCallback.

    The callback accesses val_dataset.coco.getAnnIds(), .loadAnns(),
    and .imgs[img_id]. This class provides just enough to satisfy that.
    """

    def __init__(self, id_to_entry: dict[int, dict]):
        self._entries = id_to_entry
        self.imgs = _ImgsProxy(id_to_entry)

    def getAnnIds(self, imgIds: int) -> list[int]:  # noqa: N802
        return [imgIds]

    def loadAnns(self, ann_ids: list[int]) -> list[dict]:  # noqa: N802
        img_id = ann_ids[0]
        entry = self._entries.get(img_id, {})
        sentences = entry.get("sentences", [])
        return [{"caption": s["raw"]} for s in sentences]


class _ImgsProxy:
    """Dict-like proxy for coco.imgs[img_id]."""

    def __init__(self, id_to_entry: dict[int, dict]):
        self._entries = id_to_entry

    def __getitem__(self, img_id: int) -> dict:
        entry = self._entries[img_id]
        return {"file_name": f"{entry['filepath']}/{entry['filename']}"}

    def __contains__(self, img_id: int) -> bool:
        return img_id in self._entries


class KarpathyCocoCaptionsDataset(Dataset):
    """COCO Captions dataset using the Karpathy split.

    Reads dataset_coco.json directly (no pycocotools dependency).
    Each sample returns an image-caption pair with one caption
    randomly sampled per image per call.
    """

    def __init__(
        self,
        data_root: str | Path,
        karpathy_json: str | Path,
        split: str,
        processor: AutoProcessor,
        transform: A.Compose | None = None,
        consistency_transform: A.Compose | None = None,
        max_text_length: int = 64,
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.transform = transform
        self.consistency_transform = consistency_transform
        self.max_text_length = max_text_length

        # Parse Karpathy JSON
        karpathy_path = Path(karpathy_json)
        if not karpathy_path.exists():
            raise FileNotFoundError(
                f"Karpathy JSON not found: {karpathy_path}\n"
                "Download from: https://cs.stanford.edu/people/karpathy/"
                "deepimagesent/caption_datasets.zip"
            )

        with open(karpathy_path) as f:
            data = json.load(f)

        # Filter by split (train includes restval)
        target_splits = {"train", "restval"} if split == "train" else {split}
        self._entries = [
            e for e in data["images"] if e["split"] in target_splits
        ]

        # Free memory — don't keep full JSON
        del data

        # Build lookup
        self._id_to_entry = {e["cocoid"]: e for e in self._entries}
        self.image_ids = [e["cocoid"] for e in self._entries]

        # Callback compatibility (MonitorCallback accesses .coco and .image_dir)
        self.image_dir = self.data_root
        self.coco = _KarpathyCompat(self._id_to_entry)

    def __len__(self) -> int:
        return len(self._entries)

    def _load_image(self, entry: dict) -> np.ndarray:
        path = self.data_root / entry["filepath"] / entry["filename"]
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        entry = self._entries[idx]
        image_id = entry["cocoid"]
        image = self._load_image(entry)

        # Random caption selection
        caption = random.choice(entry["sentences"])["raw"]

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=image)
            image_aug = augmented["image"]
        else:
            image_aug = image

        # Process image through SigLIP2 processor
        image_inputs = self.processor(
            images=Image.fromarray(image_aug),
            return_tensors="pt",
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
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "image_id": image_id,
        }

        if "attention_mask" in text_inputs:
            result["attention_mask"] = text_inputs["attention_mask"].squeeze(0)
        else:
            result["attention_mask"] = torch.ones_like(result["input_ids"])

        # Consistency augmentation (stronger transform of same image)
        if self.consistency_transform is not None:
            cons_aug = self.consistency_transform(image=image)
            cons_inputs = self.processor(
                images=Image.fromarray(cons_aug["image"]),
                return_tensors="pt",
            )
            result["aug_pixel_values"] = cons_inputs["pixel_values"].squeeze(0)

        return result
