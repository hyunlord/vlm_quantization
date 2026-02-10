"""Search index for cross-modal retrieval."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils.hamming import hamming_distance

logger = logging.getLogger(__name__)


class SearchIndex:
    """In-memory search index over pre-encoded dataset embeddings."""

    def __init__(self) -> None:
        self._data: dict | None = None
        self._index_path: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._data is not None

    def load(self, index_path: str) -> dict:
        """Load a .pt index file into memory."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        logger.info("Loading search index: %s", index_path)
        self._data = torch.load(str(path), map_location="cpu", weights_only=False)
        self._index_path = index_path
        logger.info(
            "Index loaded: %d items", len(self._data.get("image_ids", []))
        )
        return self.status

    @property
    def status(self) -> dict:
        if self._data is None:
            return {"loaded": False, "index_path": "", "num_items": 0, "bit_list": []}
        bit_list = sorted(self._data.get("hash_image_codes", {}).keys())
        return {
            "loaded": True,
            "index_path": self._index_path,
            "num_items": len(self._data.get("image_ids", [])),
            "bit_list": bit_list,
        }

    def query_backbone(
        self,
        embedding: list[float],
        modality: str = "image",
        top_k: int = 20,
    ) -> list[dict]:
        """Search by cosine similarity against backbone embeddings.

        Args:
            embedding: Query embedding (1152-dim).
            modality: Target modality to search: "image" or "text".
            top_k: Number of results to return.
        """
        if self._data is None:
            raise RuntimeError("Index not loaded")

        query = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # (1, D)
        key = f"backbone_{modality}_emb"
        db = self._data[key]  # (N, D)

        # Cosine similarity
        sims = F.cosine_similarity(query, db, dim=1)  # (N,)
        top_k = min(top_k, len(sims))
        scores, indices = sims.topk(top_k)

        return self._build_results(indices, scores=scores)

    def query_hash(
        self,
        binary_codes: list[int],
        bit: int = 64,
        modality: str = "image",
        top_k: int = 20,
    ) -> list[dict]:
        """Search by Hamming distance against hash codes.

        Args:
            binary_codes: Query hash code in {-1, +1} format.
            bit: Bit level (16, 32, 64, 128).
            modality: Target modality to search: "image" or "text".
            top_k: Number of results to return.
        """
        if self._data is None:
            raise RuntimeError("Index not loaded")

        query = torch.tensor(binary_codes, dtype=torch.float32).unsqueeze(0)  # (1, D)
        key = f"hash_{modality}_codes"
        db = self._data[key][bit].float()  # (N, D)

        dists = hamming_distance(query, db).squeeze(0)  # (N,)
        top_k = min(top_k, len(dists))
        _, indices = dists.topk(top_k, largest=False)  # smallest distance first

        # Compute normalized similarity for display
        scores = 1.0 - dists[indices].float() / bit

        return self._build_results(indices, scores=scores, distances=dists[indices])

    def _build_results(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        distances: torch.Tensor | None = None,
    ) -> list[dict]:
        """Build result dicts from index positions."""
        data = self._data
        results = []
        for rank, idx in enumerate(indices.tolist()):
            result = {
                "rank": rank + 1,
                "image_id": data["image_ids"][idx],
                "caption": data["captions"][idx],
                "thumbnail": data["thumbnails"][idx] if "thumbnails" in data else "",
                "score": round(float(scores[rank]), 4),
            }
            if distances is not None:
                result["distance"] = int(distances[rank].item())
            else:
                result["distance"] = None
            results.append(result)
        return results
