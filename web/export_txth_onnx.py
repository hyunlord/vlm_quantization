"""Stage A (cycle 6) — export the tiny custom head txt_h' to ONNX (fp32) and dump parity
assets for the Node test. The e5 backbone is NOT exported here: the browser/Node uses the
stock `intfloat/multilingual-e5-small` ONNX via transformers.js (q8). Only txt_h' is custom.

Quantization policy: e5 = int8 (transformers.js q8); txt_h' = fp32 (0-sensitive hash head).

Exports:
  web/static/onnx/txt_h.onnx   ONNX: emb(B,384 L2) -> continuous@1024 (B,1024); JS packs >0
Dumps (DGX /tmp/parity, NOT committed):
  gallery.bin   5000x128 uint8  (frozen ft113 img_h codes for the test gallery)
  c1_en.bin / c1_ko.bin  5000x128 uint8  (PyTorch C1 query codes — for byte closeness)
  queries.json  {ids:[...], en:[strings], ko:[strings]}

The ONNX head reconstructs ONLY the 1024-bit continuous path of NestedHashLayer (hash_head
-> slice -> BN -> L2 -> tanh), avoiding the SignSTE custom op. tanh(x)>0 == sign(x)>0 ==
pack bit, so JS packBits(continuous) is byte-identical to common.py pack_bits(binary).

Run on DGX:
  HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/export_txth_onnx.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402
from web.common import CODE_BITS, Encoder, pack_bits  # noqa: E402
from web.headadapt_eval import E5  # noqa: E402  (native e5 mean-pool helper)

TXT_HEAD = os.environ.get("TXT_HEAD", "/tmp/txt_h_e5.pt")
ONNX_OUT = Path(REPO) / "web/static/onnx/txt_h.onnx"
PARITY = Path("/tmp/parity")
_ID = re.compile(r"_0*(\d+)\.jpg")


class TxtHeadONNX(torch.nn.Module):
    """1024-bit continuous path of txt_h' only (no SignSTE) — ONNX-friendly."""

    def __init__(self, txth: NestedHashLayer, bit: int):
        super().__init__()
        self.hash_head = txth.hash_head
        self.bn = txth.batch_norms[txth.bit_list.index(bit)]
        self.bit = bit

    def forward(self, emb):  # emb: (B, e5_dim) L2-normalized
        raw = self.hash_head(emb)[:, : self.bit]
        return torch.tanh(torch.nn.functional.normalize(self.bn(raw), p=2, dim=1))


def main() -> None:
    ck = torch.load(TXT_HEAD, map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    txth = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0)
    txth.load_state_dict(ck["txt_h"]); txth.eval()
    wrap = TxtHeadONNX(txth, CODE_BITS).eval()

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, ck["embed"])
    torch.onnx.export(wrap, dummy, str(ONNX_OUT), input_names=["emb"], output_names=["code"],
                      dynamic_axes={"emb": {0: "B"}, "code": {0: "B"}}, opset_version=17,
                      dynamo=False)  # legacy exporter -> single self-contained file (no .onnx.data)
    sz = ONNX_OUT.stat().st_size / 1024
    print(f"[export] txt_h.onnx -> {ONNX_OUT} ({sz:.1f} KB, fp32)", flush=True)

    # sanity: wrapper continuous>0 packing must equal NestedHashLayer binary packing
    bi = bits.index(CODE_BITS)
    with torch.no_grad():
        e = torch.nn.functional.normalize(torch.randn(8, ck["embed"]), dim=1)
        cont = wrap(e).numpy()
        ref = txth(e)[bi]["binary"].numpy()
    same = bool((pack_bits(cont) == pack_bits(ref)).all())
    print(f"[export] wrapper continuous>0 packing == NestedHashLayer binary packing: {same}", flush=True)

    # onnxruntime numeric verify: ONNX output vs torch wrapper (+ packing identity)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ev = torch.nn.functional.normalize(torch.randn(16, ck["embed"]), dim=1)
        tcont = wrap(ev).numpy()
    ocont = sess.run(["code"], {"emb": ev.numpy()})[0]
    print(f"[export] onnxruntime vs torch max|d|: {np.abs(ocont - tcont).max():.2e} | "
          f"pack match: {bool((pack_bits(ocont) == pack_bits(tcont)).all())}", flush=True)

    # ---- parity assets ----
    PARITY.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    en_caps = [str(c) for c in EC["test"]["captions"]]
    te_img = EC["test"]["img"].float().numpy()
    ko_full = {}
    for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
        e = json.loads(line); ko_full[int(_ID.search(e["image_path"]).group(1)) if _ID.search(e["image_path"]) else -1] = e.get("captions", [])
    ko_caps = [(ko_full.get(c, [""]) or [""])[0] for c in te_ids]

    gallery = pack_bits(enc._codes_pm1(enc.img_h, torch.from_numpy(te_img)))
    gallery.tofile(PARITY / "gallery.bin")
    json.dump({"ids": te_ids, "en": en_caps, "ko": ko_caps},
              open(PARITY / "queries.json", "w"), ensure_ascii=False)

    e5 = E5(ck["student"], enc.device)
    txth.to(enc.device)  # move head to device for the C1 reference codes
    for lang, caps in [("en", en_caps), ("ko", ko_caps)]:
        emb = e5.embed(caps)
        with torch.no_grad():
            o = txth(torch.from_numpy(emb).to(enc.device))
        pack_bits(o[bi]["binary"].cpu().numpy()).tofile(PARITY / f"c1_{lang}.bin")
    print(f"[export] assets -> {PARITY} (gallery {gallery.shape}, queries {len(te_ids)}, c1 codes)",
          flush=True)
    print("[export] DONE", flush=True)


if __name__ == "__main__":
    main()
