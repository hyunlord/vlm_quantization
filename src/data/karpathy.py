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

    Supports:
        - Multi-caption (P1): return auxiliary caption for extra positives
        - Text dropout (P5): randomly drop tokens for text augmentation
        - Multi-label (supervised): load COCO 80-category annotations
    """

    NUM_CATEGORIES = 80  # COCO has 80 object categories

    def __init__(
        self,
        data_root: str | Path,
        karpathy_json: str | Path,
        split: str,
        processor: AutoProcessor,
        transform: A.Compose | None = None,
        consistency_transform: A.Compose | None = None,
        max_text_length: int = 64,
        image_size: int = 384,
        num_captions: int = 1,
        text_dropout_prob: float = 0.0,
        instances_json: str | Path | None = None,
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.transform = transform
        self.consistency_transform = consistency_transform
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.num_captions = num_captions
        self.text_dropout_prob = text_dropout_prob
        self._orig_resize = A.Resize(image_size, image_size)

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

        # Multi-label annotations (optional, for supervised hashing)
        self._labels: dict[int, torch.Tensor] | None = None
        if instances_json is not None:
            self._labels = self._load_instance_labels(instances_json)

    def _load_instance_labels(
        self, instances_json: str | Path,
    ) -> dict[int, torch.Tensor]:
        """Load COCO instance annotations and build multi-hot label vectors.

        COCO has 80 categories with non-contiguous IDs (1-90 with gaps).
        We map them to contiguous indices 0-79.

        Karpathy split spans both train2014/ and val2014/ images, so we
        auto-discover and load both instances_train2014.json and
        instances_val2014.json from the same annotations directory.
        """
        instances_path = Path(instances_json)
        if not instances_path.exists():
            raise FileNotFoundError(
                f"COCO instances JSON not found: {instances_path}\n"
                "Download from: https://cocodataset.org/#download"
            )

        # Auto-discover both train and val instance files
        ann_dir = instances_path.parent
        json_files = [instances_path]
        for sibling in ["instances_train2014.json", "instances_val2014.json"]:
            other = ann_dir / sibling
            if other.exists() and other.resolve() != instances_path.resolve():
                json_files.append(other)

        labels: dict[int, torch.Tensor] = {}
        cat_to_idx: dict[int, int] | None = None

        for jf in json_files:
            with open(jf) as f:
                instances = json.load(f)

            # Build category ID -> contiguous index mapping (same across files)
            if cat_to_idx is None:
                cat_ids = sorted(c["id"] for c in instances["categories"])
                cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

            for ann in instances["annotations"]:
                img_id = ann["image_id"]
                cat_idx = cat_to_idx[ann["category_id"]]
                if img_id not in labels:
                    labels[img_id] = torch.zeros(self.NUM_CATEGORIES)
                labels[img_id][cat_idx] = 1.0

        return labels

    def __len__(self) -> int:
        return len(self._entries)

    def _load_image(self, entry: dict) -> np.ndarray:
        path = self.data_root / entry["filepath"] / entry["filename"]
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _tokenize(self, caption: str) -> dict[str, torch.Tensor]:
        """Tokenize a caption and optionally apply text dropout (P5)."""
        text_inputs = self.processor.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].squeeze(0)

        if "attention_mask" in text_inputs:
            attention_mask = text_inputs["attention_mask"].squeeze(0)
        else:
            attention_mask = torch.ones_like(input_ids)

        # Text dropout (P5): randomly replace tokens with pad
        if self.text_dropout_prob > 0:
            input_ids, attention_mask = self._apply_text_dropout(
                input_ids, attention_mask,
            )

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _apply_text_dropout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop tokens for text augmentation (P5)."""
        pad_id = self.processor.tokenizer.pad_token_id or 0
        actual_tokens = attention_mask.bool() & (input_ids != pad_id)

        # Don't drop the first token (BOS/CLS) or last actual token
        if actual_tokens.sum() <= 2:
            return input_ids, attention_mask
        first_idx = actual_tokens.nonzero(as_tuple=True)[0][0]
        last_idx = actual_tokens.nonzero(as_tuple=True)[0][-1]

        drop_mask = torch.rand(input_ids.shape) < self.text_dropout_prob
        drop_mask[first_idx] = False  # keep first token
        drop_mask[last_idx] = False  # keep last actual token
        drop_mask = drop_mask & actual_tokens

        input_ids = input_ids.clone()
        input_ids[drop_mask] = pad_id
        return input_ids, attention_mask

    def _pick_captions(self, entry: dict) -> list[str]:
        """Pick num_captions distinct random captions from the entry."""
        sentences = entry["sentences"]
        n = min(self.num_captions, len(sentences))
        chosen = random.sample(sentences, n)
        captions = [s["raw"] for s in chosen]
        # If fewer captions available than requested, pad with repeats
        while len(captions) < self.num_captions:
            captions.append(captions[0])
        return captions

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        entry = self._entries[idx]
        image_id = entry["cocoid"]
        image = self._load_image(entry)

        # Pick captions (P1: multi-caption support)
        captions = self._pick_captions(entry)

        # Original view (Resize only — clean anchor for contrastive loss)
        orig_resized = self._orig_resize(image=image)["image"]
        orig_inputs = self.processor(
            images=Image.fromarray(orig_resized), return_tensors="pt",
        )

        # Tokenize primary caption
        primary_tokens = self._tokenize(captions[0])

        result = {
            "pixel_values": orig_inputs["pixel_values"].squeeze(0),
            "input_ids": primary_tokens["input_ids"],
            "attention_mask": primary_tokens["attention_mask"],
            "image_id": image_id,
        }

        # Multi-hot labels for supervised hashing (zero vector if image has no annotations)
        if self._labels is not None:
            result["labels"] = self._labels.get(
                image_id, torch.zeros(self.NUM_CATEGORIES)
            )

        # Auxiliary caption (P1: multi-caption, second distinct caption)
        if self.num_captions >= 2:
            aux_tokens = self._tokenize(captions[1])
            result["aux_input_ids"] = aux_tokens["input_ids"]
            result["aux_attention_mask"] = aux_tokens["attention_mask"]

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
