"""Inference engine for interactive hash code exploration."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

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
        # Backbone-only mode (no hash layers)
        self.backbone: AutoModel | None = None
        self.backbone_only: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None or self.backbone is not None

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
        self.backbone = None
        self.backbone_only = False
        logger.info(
            "Model loaded: %s, bit_list=%s", self.model_name, self.bit_list
        )
        return self.status()

    def load_backbone_only(self, model_name: str) -> dict:
        """Load backbone model without hash layers (for baseline comparison)."""
        logger.info("Loading backbone only: %s", model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.model = None
        self.bit_list = []
        self.checkpoint_path = ""
        self.backbone_only = True
        logger.info("Backbone loaded: %s", model_name)
        return self.status()

    def status(self) -> dict:
        hparams: dict = {}
        if self.model is not None:
            hparams = {k: v for k, v in self.model.hparams.items()}
            if "model_name" in hparams:
                hparams["model_name"] = hparams["model_name"].split("/")[-1]
        return {
            "loaded": self.is_loaded,
            "backbone_only": self.backbone_only,
            "checkpoint": self.checkpoint_path,
            "model_name": self.model_name,
            "bit_list": self.bit_list,
            "hparams": hparams,
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

    def _get_backbone(self):
        """Return the backbone model (from hash model or standalone)."""
        if self.model is not None:
            return self.model.backbone
        return self.backbone

    @torch.no_grad()
    def encode_image_backbone(self, image: Image.Image) -> list[float]:
        """Encode a PIL image into raw backbone embedding."""
        backbone = self._get_backbone()
        if backbone is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if self.model is not None:
            emb = self.model.encode_image_backbone(pixel_values)
        else:
            outputs = backbone.vision_model(pixel_values=pixel_values)
            emb = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        return emb.squeeze(0).tolist()

    @torch.no_grad()
    def encode_text_backbone(self, text: str) -> list[float]:
        """Encode text into raw backbone embedding."""
        backbone = self._get_backbone()
        if backbone is None or self.processor is None:
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

        if self.model is not None:
            emb = self.model.encode_text_backbone(input_ids, attention_mask)
        else:
            outputs = backbone.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            emb = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        return emb.squeeze(0).tolist()

    @staticmethod
    def compare_backbone(emb_a: list[float], emb_b: list[float]) -> dict:
        """Compute cosine similarity between two backbone embeddings."""
        a = torch.tensor(emb_a, dtype=torch.float32)
        b = torch.tensor(emb_b, dtype=torch.float32)
        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        return {"cosine_similarity": round(cosine, 4)}

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

            # Group by first subdirectory under root (usually timestamped run dir)
            try:
                parts = p.relative_to(root).parts
                run_dir = parts[0] if len(parts) > 1 else ""
            except ValueError:
                run_dir = p.parent.name

            # Extract epoch from full path (handles subdir like best-epoch=0-val/)
            epoch_match = re.search(r"epoch[=_](\d+)", str(p))
            epoch = int(epoch_match.group(1)) if epoch_match else None

            # Extract step from full path
            step_match = re.search(r"step[=_](\d+)", str(p))
            step = int(step_match.group(1)) if step_match else None

            # Extract val_loss from filename (float with 3+ decimals)
            loss_match = re.search(r"(\d+\.\d{3,})", p.name)
            val_loss = float(loss_match.group(1)) if loss_match else None

            ckpts.append({
                "path": str(p),
                "name": p.name,
                "run_dir": run_dir,
                "size_mb": round(stat.st_size / 1024 / 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
            })
        return ckpts

    @staticmethod
    def peek_hparams(checkpoint_path: str) -> dict:
        """Extract hyperparameters from a checkpoint without full model load.

        Uses a JSON sidecar cache (``<path>.hparams.json``) to avoid
        expensive ``torch.load()`` calls on subsequent accesses.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return {}

        # Fast path: read from cached JSON sidecar
        sidecar = Path(str(path) + ".hparams.json")
        if sidecar.is_file():
            try:
                with open(sidecar) as f:
                    return json.load(f)
            except Exception:
                pass  # fall through to torch.load

        # Slow path: full checkpoint read
        try:
            kwargs: dict = {"map_location": "cpu", "weights_only": False}
            try:
                # mmap avoids reading tensor data into RAM (PyTorch 2.1+)
                ckpt = torch.load(str(path), mmap=True, **kwargs)
            except TypeError:
                ckpt = torch.load(str(path), **kwargs)
            hparams = dict(ckpt.get("hyper_parameters", {}))
            del ckpt

            # Cache for future lookups
            try:
                with open(sidecar, "w") as f:
                    json.dump(hparams, f)
            except Exception as e:
                logger.warning("Failed to write hparams cache: %s", e)

            return hparams
        except Exception as e:
            logger.warning("peek_hparams failed for %s: %s", checkpoint_path, e)
            return {}

    @staticmethod
    def decode_base64_image(b64: str) -> Image.Image:
        """Decode a base64 string to PIL Image."""
        # Strip data URI prefix if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    @staticmethod
    def download_image(url: str) -> Image.Image:
        """Download image from URL and return as PIL Image."""
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
