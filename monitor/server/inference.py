"""Inference engine for interactive hash code exploration."""
from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from src.models.cross_modal_hash import CrossModalHashModel
from src.utils.hamming import to_binary_01

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Loads a trained checkpoint and encodes images/text to hash codes."""

    def __init__(self) -> None:
        self.model: CrossModalHashModel | None = None
        self.processor: AutoProcessor | None = None
        self.bit_list: list[int] = []
        self.model_name: str = ""
        self.checkpoint_path: str = ""

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, checkpoint_path: str) -> dict:
        """Load model from Lightning checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Loading checkpoint: %s", checkpoint_path)
        self.model = CrossModalHashModel.load_from_checkpoint(
            str(path), map_location="cpu",
        )
        self.model.eval()
        self.model_name = self.model.hparams.get("model_name", "")
        self.bit_list = list(self.model.hparams.get("bit_list", [64]))
        self.checkpoint_path = checkpoint_path

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info(
            "Model loaded: %s, bit_list=%s", self.model_name, self.bit_list
        )
        return self.status()

    def status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "checkpoint": self.checkpoint_path,
            "model_name": self.model_name,
            "bit_list": self.bit_list,
        }

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> list[dict]:
        """Encode a PIL image into hash codes at all bit levels."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        outputs = self.model.encode_image(pixel_values)
        return self._format_outputs(outputs)

    @torch.no_grad()
    def encode_text(self, text: str) -> list[dict]:
        """Encode text into hash codes at all bit levels."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        text_inputs = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        )

        outputs = self.model.encode_text(input_ids, attention_mask)
        return self._format_outputs(outputs)

    def _format_outputs(self, outputs: list[dict]) -> list[dict]:
        """Convert model outputs to serializable format."""
        result = []
        for k, out in enumerate(outputs):
            binary_01 = to_binary_01(out["binary"]).squeeze(0)
            result.append({
                "bits": self.bit_list[k],
                "binary": binary_01.tolist(),
                "continuous": out["continuous"].squeeze(0).tolist(),
            })
        return result

    @staticmethod
    def compare(codes_a: list[dict], codes_b: list[dict]) -> list[dict]:
        """Compare two sets of hash codes at each bit level."""
        comparisons = []
        for a, b in zip(codes_a, codes_b):
            bits = a["bits"]
            ba = torch.tensor(a["binary"], dtype=torch.uint8)
            bb = torch.tensor(b["binary"], dtype=torch.uint8)
            hamming = int((ba ^ bb).sum().item())
            similarity = 1.0 - hamming / bits
            comparisons.append({
                "bits": bits,
                "hamming": hamming,
                "max_distance": bits,
                "similarity": round(similarity, 4),
            })
        return comparisons

    @staticmethod
    def list_checkpoints(directory: str) -> list[dict]:
        """List all .ckpt files under directory, grouped by run folder."""
        root = Path(directory)
        if not root.is_dir():
            return []
        ckpts = []
        for p in sorted(
            root.rglob("*.ckpt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            stat = p.stat()
            ckpts.append({
                "path": str(p),
                "name": p.name,
                "run_dir": p.parent.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return ckpts

    @staticmethod
    def decode_base64_image(b64: str) -> Image.Image:
        """Decode a base64 string to PIL Image."""
        # Strip data URI prefix if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
