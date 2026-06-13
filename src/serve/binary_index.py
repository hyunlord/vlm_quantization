"""Binary hash search index for fast cross-modal / image retrieval.

The whole point of the hashing model is that a database of N items can be stored
as compact bit-strings and searched with Hamming distance (XOR + popcount), which
is dramatically smaller and faster than float-embedding cosine search:

    fp32 embedding (1152-d)  ->  4608 bytes / item
    64-bit hash code         ->     8 bytes / item   (~576x smaller)

This module provides:
    - pack_codes / to_bits01 : {-1,+1} or {0,1} codes -> packed uint8 bitstrings
    - HammingIndex           : exact Hamming search (numpy popcount or FAISS binary)
    - FloatIndex             : cosine baseline ("normal image search") for comparison
    - MatryoshkaIndex        : coarse-to-fine (short code filter -> long code rerank)
    - ServingIndex           : multi-bit index artifact loader for the serving API

Everything is numpy-based and runs on CPU with zero heavy dependencies. FAISS is
used automatically when installed (`pip install faiss-cpu`) for a faster backend.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# 256-entry population-count lookup table (popcount of each byte value).
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

# FAISS is imported lazily (only when the "faiss" backend is actually requested).
# This keeps `import binary_index` from loading the faiss extension, which avoids a
# known faiss+torch OpenMP/libomp conflict on macOS when both share one process.
_faiss_mod = None


def has_faiss() -> bool:
    """True if faiss is importable — without importing it."""
    return importlib.util.find_spec("faiss") is not None


def _load_faiss():
    global _faiss_mod
    if _faiss_mod is None:
        import faiss

        _faiss_mod = faiss
    return _faiss_mod


def to_bits01(codes: np.ndarray) -> np.ndarray:
    """Map {-1,+1} or {0,1} codes (N, D) to uint8 {0,1}."""
    return (np.asarray(codes) > 0).astype(np.uint8)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Pack {-1,+1}/{0,1} codes (N, D) into uint8 bitstrings (N, ceil(D/8))."""
    bits = to_bits01(codes)
    if bits.ndim == 1:
        bits = bits[None, :]
    return np.ascontiguousarray(np.packbits(bits, axis=1))


def _hamming_block(q_packed: np.ndarray, db_packed: np.ndarray, db_block: int) -> np.ndarray:
    """Hamming distances (Q, N) between packed queries and packed database."""
    n = db_packed.shape[0]
    out = np.empty((q_packed.shape[0], n), dtype=np.uint16)
    for s in range(0, n, db_block):
        e = min(s + db_block, n)
        xor = np.bitwise_xor(q_packed[:, None, :], db_packed[None, s:e, :])
        out[:, s:e] = _POPCOUNT[xor].sum(axis=2)
    return out


def _topk_smallest(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of the k smallest entries per row, sorted."""
    k = min(k, dist.shape[1])
    part = np.argpartition(dist, k - 1, axis=1)[:, :k]
    rows = np.arange(dist.shape[0])[:, None]
    order = np.argsort(dist[rows, part], axis=1)
    idx = part[rows, order]
    return idx, dist[rows, idx]


class HammingIndex:
    """Exact Hamming-distance search over packed binary codes."""

    def __init__(
        self,
        packed: np.ndarray,
        ids: np.ndarray | None = None,
        n_bits: int | None = None,
        backend: str = "numpy",
    ):
        self.packed = np.ascontiguousarray(packed, dtype=np.uint8)
        self.n_bits = int(n_bits) if n_bits is not None else self.packed.shape[1] * 8
        self.ids = np.arange(self.packed.shape[0]) if ids is None else np.asarray(ids)
        self.backend = backend
        self._faiss = None
        if backend == "faiss":
            if not has_faiss():
                raise RuntimeError("faiss is not installed (pip install faiss-cpu)")
            faiss = _load_faiss()
            index = faiss.IndexBinaryFlat(self.n_bits)
            index.add(self.packed)
            self._faiss = index

    @classmethod
    def from_codes(
        cls, codes: np.ndarray, ids: np.ndarray | None = None, backend: str = "numpy"
    ) -> HammingIndex:
        codes = np.asarray(codes)
        return cls(pack_codes(codes), ids, n_bits=codes.shape[1], backend=backend)

    def __len__(self) -> int:
        return self.packed.shape[0]

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes)

    def search(
        self, query_codes: np.ndarray, k: int = 10, q_block: int = 512, db_block: int = 8192
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances), each (Q, k), nearest first."""
        q = pack_codes(query_codes)
        k = min(k, len(self))
        if self._faiss is not None:
            dist, idx = self._faiss.search(q, k)
            return self.ids[idx], dist
        out_ids = np.empty((q.shape[0], k), dtype=self.ids.dtype)
        out_dist = np.empty((q.shape[0], k), dtype=np.uint16)
        for s in range(0, q.shape[0], q_block):
            e = min(s + q_block, q.shape[0])
            dist = _hamming_block(q[s:e], self.packed, db_block)
            idx, d = _topk_smallest(dist, k)
            out_ids[s:e] = self.ids[idx]
            out_dist[s:e] = d
        return out_ids, out_dist


class FloatIndex:
    """Cosine-similarity search over float embeddings — the baseline that a
    'normal image search' system uses. Kept for head-to-head benchmarking."""

    def __init__(self, emb: np.ndarray, ids: np.ndarray | None = None):
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        self.ids = np.arange(emb.shape[0]) if ids is None else np.asarray(ids)
        self.dim = emb.shape[1]

    def __len__(self) -> int:
        return self.emb.shape[0]

    @property
    def nbytes(self) -> int:
        return int(self.emb.nbytes)

    def search(
        self, query_emb: np.ndarray, k: int = 10, q_block: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(query_emb, dtype=np.float32))
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        k = min(k, len(self))
        out_ids = np.empty((q.shape[0], k), dtype=self.ids.dtype)
        out_sim = np.empty((q.shape[0], k), dtype=np.float32)
        for s in range(0, q.shape[0], q_block):
            e = min(s + q_block, q.shape[0])
            sims = q[s:e] @ self.emb.T
            idx, neg = _topk_smallest(-sims, k)
            out_ids[s:e] = self.ids[idx]
            out_sim[s:e] = -neg
        return out_ids, out_sim


class MatryoshkaIndex:
    """Coarse-to-fine retrieval exploiting the prefix-nested codes.

    Filter a large candidate set with a cheap short code, then rerank only those
    candidates with the accurate long code. Gives long-code accuracy at close to
    short-code speed.
    """

    def __init__(
        self,
        short_codes: np.ndarray,
        long_codes: np.ndarray,
        ids: np.ndarray | None = None,
    ):
        self.short = HammingIndex.from_codes(short_codes)  # ids = positions
        self.long_packed = pack_codes(long_codes)
        self.long_bits = np.asarray(long_codes).shape[1]
        self.ids = np.arange(len(self.short)) if ids is None else np.asarray(ids)

    def __len__(self) -> int:
        return len(self.short)

    def search(
        self,
        query_short: np.ndarray,
        query_long: np.ndarray,
        k: int = 10,
        candidates: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = min(candidates, len(self))
        cand_pos, _ = self.short.search(query_short, candidates)  # (Q, C) positions
        ql = pack_codes(query_long)
        out_ids, out_dist = [], []
        for i in range(cand_pos.shape[0]):
            pos = cand_pos[i]
            d = _hamming_block(ql[i : i + 1], self.long_packed[pos], len(pos))[0]
            idx, dd = _topk_smallest(d[None, :], k)
            out_ids.append(self.ids[pos[idx[0]]])
            out_dist.append(dd[0])
        return np.array(out_ids), np.array(out_dist)


class ServingIndex:
    """Multi-bit index artifact (built by scripts/build_index.py) for the API.

    Holds one HammingIndex per available bit length plus item metadata
    (positions as ids, original paths/captions for display).
    """

    def __init__(
        self,
        per_bit: dict[int, HammingIndex],
        paths: np.ndarray | None = None,
        captions: np.ndarray | None = None,
        item_ids: np.ndarray | None = None,
        emb: np.ndarray | None = None,
    ):
        self.per_bit = per_bit
        self.bits = sorted(per_bit)
        self.paths = paths
        self.captions = captions
        self.item_ids = item_ids
        self.emb = emb
        self.size = len(next(iter(per_bit.values()))) if per_bit else 0

    @classmethod
    def load(cls, path: str | Path, backend: str = "numpy") -> ServingIndex:
        data = np.load(str(path), allow_pickle=True)
        ids = data["ids"] if "ids" in data.files else None
        per_bit: dict[int, HammingIndex] = {}
        for key in data.files:
            if key.startswith("packed_"):
                bit = int(key.split("_")[1])
                per_bit[bit] = HammingIndex(
                    data[key], ids=ids, n_bits=bit, backend=backend
                )
        return cls(
            per_bit,
            paths=data["paths"] if "paths" in data.files else None,
            captions=data["captions"] if "captions" in data.files else None,
            item_ids=data["item_ids"] if "item_ids" in data.files else None,
            emb=data["emb"] if "emb" in data.files else None,
        )

    def default_bit(self) -> int:
        return 64 if 64 in self.per_bit else self.bits[-1]

    def search(self, query_codes: np.ndarray, bit: int, k: int = 10) -> list[dict]:
        ids, dist = self.per_bit[bit].search(query_codes, k)
        results = []
        for pos, d in zip(ids[0].tolist(), dist[0].tolist()):
            hit = {
                "position": int(pos),
                "distance": int(d),
                "similarity": round(1.0 - d / bit, 4),
            }
            if self.item_ids is not None:
                hit["item_id"] = (
                    int(self.item_ids[pos])
                    if np.issubdtype(self.item_ids.dtype, np.integer)
                    else str(self.item_ids[pos])
                )
            if self.paths is not None:
                hit["path"] = str(self.paths[pos])
            if self.captions is not None:
                hit["caption"] = str(self.captions[pos])
            results.append(hit)
        return results
