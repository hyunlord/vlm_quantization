"""Export a trained image-side head img_h'_E to ONNX (fp32) + Python<->ONNX parity — mirror of
export_txth_onnx.py (txt_h.onnx). The vision TOWER itself is NOT exported here: on-device it runs via
transformers.js/ONNX (q8) like the e5 text tower (TJS-turnkey for siglip/dinov2/mobileclip2). Only the tiny
custom img_h'_E is exported (0-sensitive hash head -> fp32). Same continuous-path trick (tanh(L2(BN(slice)))
> 0 == sign > 0 == pack bit) so JS packBits(continuous) == common.py pack_bits(binary).

Run on DGX (after the head exists at /tmp/imgh_<enc>.pt):
  .venv/bin/python web/export_imgh_onnx.py --enc siglip2-base
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

CODE_BITS = 1024


def pack_bits(codes):
    b = (np.asarray(codes) > 0).astype(np.uint8)
    if b.ndim == 1:
        b = b[None, :]
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="big"), dtype=np.uint8)


class ImgHeadONNX(torch.nn.Module):
    """1024-bit continuous path of img_h'_E only (no SignSTE) — ONNX-friendly (mirror TxtHeadONNX)."""

    def __init__(self, head: NestedHashLayer, bit: int):
        super().__init__()
        self.hash_head = head.hash_head
        self.bn = head.batch_norms[head.bit_list.index(bit)]
        self.bit = bit

    def forward(self, emb):  # emb: (B, E_dim) L2-normalized
        raw = self.hash_head(emb)[:, : self.bit]
        return torch.tanh(torch.nn.functional.normalize(self.bn(raw), p=2, dim=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enc", required=True); args = ap.parse_args()
    ck = torch.load(f"/tmp/imgh_{args.enc}.pt", map_location="cpu")
    bits = [int(b) for b in ck["bits"]]
    head = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0)
    head.load_state_dict(ck["img_h"]); head.eval()
    wrap = ImgHeadONNX(head, CODE_BITS).eval()
    out = Path(REPO) / f"web/static/onnx/img_h_{args.enc}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, ck["embed"])
    torch.onnx.export(wrap, dummy, str(out), input_names=["emb"], output_names=["code"],
                      dynamic_axes={"emb": {0: "B"}, "code": {0: "B"}}, opset_version=17, dynamo=False)
    sz = out.stat().st_size / 1024
    print(f"[export] img_h_{args.enc}.onnx -> {out} ({sz:.1f} KB, fp32) | encoder={ck.get('encoder')} embed={ck['embed']}", flush=True)

    bi = bits.index(CODE_BITS)
    with torch.no_grad():
        e = torch.nn.functional.normalize(torch.randn(8, ck["embed"]), dim=1)
        cont = wrap(e).numpy(); ref = head(e)[bi]["binary"].numpy()
    same = bool((pack_bits(cont) == pack_bits(ref)).all())
    print(f"[export] wrapper continuous>0 packing == NestedHashLayer binary packing: {same}", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ev = torch.nn.functional.normalize(torch.randn(16, ck["embed"]), dim=1)
        tcont = wrap(ev).numpy()
    ocont = sess.run(["code"], {"emb": ev.numpy()})[0]
    maxd = float(np.abs(ocont - tcont).max())
    packmatch = bool((pack_bits(ocont) == pack_bits(tcont)).all())
    print(f"[export] onnxruntime vs torch max|d|: {maxd:.2e} | pack match (Hamming=0): {packmatch}", flush=True)
    print(f"[export] PARITY {'OK' if (same and packmatch and maxd < 1e-4) else 'FAIL'} -> {out}", flush=True)


if __name__ == "__main__":
    main()
