"""Shared contract for the browser-side 1-bit search demo (web/ v1).

This module is the SINGLE source of truth that `build_index.py`, `query_server.py`
and `verify_parity.py` all import — so the index rows and the query codes are packed
*identically*. That identical packing is the one invariant that makes the browser's
local Hamming search agree with faiss `IndexBinaryFlat`.

Two pieces:

  pack_bits(codes)
      ±1 / {0,1} codes (N, bits) -> packed uint8 (N, bits//8), bitorder='big'.
      Thin wrapper over `src.serve.binary_index.pack_codes` (np.packbits) so the web
      demo, the dashboard index, faiss, and the bench scripts all use one packing.
      1024-bit code -> 128 bytes.

  Encoder
      Loads the ft113 hash head (img_h + txt_h) and, lazily, the SigLIP2 text tower,
      and turns (a) cached image embeddings and (b) query text into 1024-bit codes
      using the EXACT contract validated by scripts/eval_korean.py:
          ni = ckpt.get("norm_in", 1)
          inp = F.normalize(emb, dim=1) if ni else emb     # L2-normalize iff norm_in
          binary = head(inp)[bit_index]["binary"]          # ±1, head in .eval()
          packed = pack_bits(binary)                        # big-endian bytes
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make `src.*` importable whether this is imported as `web.common` (server) or run
# from a script that added the repo root to sys.path.
REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")
HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")  # ft113 (KO+EN), DGX
CODE_BITS = int(os.environ.get("CODE_BITS", "1024"))
CODE_BYTES = CODE_BITS // 8  # 128

# 256-entry popcount lookup table. The browser (app.js) builds the identical table;
# verify_parity.py uses this one to mirror the JS search byte-for-byte.
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack_bits(codes: np.ndarray) -> np.ndarray:
    """±1 or {0,1} codes (N, CODE_BITS) -> packed uint8 (N, CODE_BYTES), bitorder='big'.

    THE parity invariant: index rows and query codes must both go through this one
    function. Byte-identical to `src.serve.binary_index.pack_codes` and to the packing
    demo/live_server.py feeds to faiss `IndexBinaryFlat` — inlined here so web/ is
    self-contained (no src.serve dependency) and portable.
    """
    bits01 = (np.asarray(codes) > 0).astype(np.uint8)  # ±1 -> {0,1}; value > 0 -> 1
    if bits01.ndim == 1:
        bits01 = bits01[None, :]
    return np.ascontiguousarray(np.packbits(bits01, axis=1, bitorder="big"), dtype=np.uint8)


def hamming_topk(packed_db: np.ndarray, q_packed: np.ndarray, k: int):
    """Reference Hamming top-k mirroring app.js exactly (LUT popcount of XOR).

    Returns (idx, dist) of the k nearest rows, sorted by (distance asc, index asc) —
    the same deterministic tie-break the browser uses. `q_packed` is a (CODE_BYTES,)
    uint8 query code.
    """
    xor = np.bitwise_xor(packed_db, q_packed[None, :])
    dist = POPCOUNT_LUT[xor].sum(axis=1).astype(np.uint32)  # (N,)
    k = min(k, dist.shape[0])
    # lexsort: primary key last -> sort by dist, ties broken by ascending index.
    order = np.lexsort((np.arange(dist.shape[0]), dist))[:k]
    return order, dist[order]


class Encoder:
    """Loads ft113 head + (lazily) the SigLIP2 text tower; produces 1024-bit codes."""

    def __init__(self, head_path: str = HEAD_PATH, device: str | None = None,
                 bits: int = CODE_BITS):
        import torch
        from src.models.nested_hash_layer import NestedHashLayer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head_path = head_path

        ck = torch.load(head_path, map_location="cpu")
        self.bit_list = [int(b) for b in ck["bits"]]
        if bits not in self.bit_list:
            raise ValueError(f"head {head_path} bits {self.bit_list} has no {bits}-bit code")
        self.bits = bits
        self.bit_index = self.bit_list.index(bits)
        self.norm_in = int(ck.get("norm_in", 1))   # 1 -> L2-normalize input before head
        self.embed = int(ck["embed"])
        self.hidden = int(ck["hidden"])
        self.mode = ck.get("mode")

        self.img_h = NestedHashLayer(self.embed, self.hidden, self.bit_list, 0.0)
        self.img_h.load_state_dict(ck["img_h"])
        self.img_h.to(self.device).eval()
        self.txt_h = NestedHashLayer(self.embed, self.hidden, self.bit_list, 0.0)
        self.txt_h.load_state_dict(ck["txt_h"])
        self.txt_h.to(self.device).eval()

        self._tower = None  # SigLIP2 text model (loaded on first text query)
        self._tok = None

    # ---- embedding input prep (the norm_in contract) ----
    def _prep(self, x):
        if self.norm_in:
            return self.torch.nn.functional.normalize(x, dim=1)
        return x

    def _codes_pm1(self, head, emb_t):
        """emb_t: torch (N, embed) -> numpy ±1 codes (N, bits)."""
        with self.torch.no_grad():
            outs = head(self._prep(emb_t.to(self.device)))
        return outs[self.bit_index]["binary"].cpu().numpy()

    # ---- image side (corpus build): cached embeddings -> packed codes ----
    def image_codes_packed(self, emb: np.ndarray, batch: int = 16384) -> np.ndarray:
        """emb: (N, embed) float32 -> packed uint8 (N, CODE_BYTES)."""
        parts = []
        for s in range(0, len(emb), batch):
            chunk = np.ascontiguousarray(emb[s:s + batch], dtype=np.float32)
            et = self.torch.from_numpy(chunk)
            parts.append(pack_bits(self._codes_pm1(self.img_h, et)))
        return np.ascontiguousarray(np.concatenate(parts, axis=0))

    # ---- text side (query): string -> SigLIP2 emb -> packed code ----
    def _load_tower(self):
        if self._tower is not None:
            return
        from transformers import AutoModel
        torch = self.torch
        dt = torch.bfloat16 if self.device == "cuda" else torch.float32
        bb = AutoModel.from_pretrained(MODEL, dtype=dt).to(self.device).eval()
        try:
            del bb.vision_model  # text-only; free the vision tower
        except Exception:
            pass
        self._tower = bb
        try:
            from transformers import AutoProcessor
            self._tok = AutoProcessor.from_pretrained(MODEL).tokenizer
        except Exception:
            from transformers import GemmaTokenizer
            self._tok = GemmaTokenizer.from_pretrained(MODEL)

    @staticmethod
    def _pool(o):
        return o.pooler_output if getattr(o, "pooler_output", None) is not None \
            else o.last_hidden_state.mean(1)

    def encode_text_emb(self, text: str) -> np.ndarray:
        """text -> raw SigLIP2 text embedding (1, embed) float32 (un-normalized)."""
        self._load_tower()
        torch = self.torch
        t = self._tok([text], padding="max_length", max_length=64, truncation=True,
                      return_tensors="pt")
        am = t.get("attention_mask")
        with torch.no_grad():
            e = self._pool(self._tower.text_model(
                input_ids=t["input_ids"].to(self.device),
                attention_mask=am.to(self.device) if am is not None else None,
            )).float()
        return e.cpu().numpy()

    def text_code_packed(self, text: str) -> np.ndarray:
        """text -> packed query code (CODE_BYTES,) uint8 (same packing as index rows)."""
        e = self.encode_text_emb(text)
        codes = self._codes_pm1(self.txt_h, self.torch.from_numpy(e))  # (1, bits) ±1
        return pack_bits(codes)[0]

    def text_code_bytes(self, text: str) -> bytes:
        return self.text_code_packed(text).tobytes()
