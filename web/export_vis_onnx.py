"""Export MobileCLIP2-S2 VISION tower -> ONNX (fp32) for in-browser on-device image indexing.
Output emb is L2-normalized (matches img_h' training input). Parity: torch vs onnxruntime, and
full chain vis->img_h'->packbits == python. Also dumps preprocessing params for JS Canvas replication.
"""
import json, sys, os
import numpy as np
import torch
from torch import nn
import open_clip

from pathlib import Path as _P
REPO = os.environ.get("REPO", str(_P(__file__).resolve().parent.parent))
sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer

OUT = f"{REPO}/web/static/onnx/vis_mobileclip2-s2.onnx"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

m, _, pp = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained="dfndr2b", force_quick_gelu=False)
m = m.eval()
try:
    from open_clip import reparameterize_model
    m = reparameterize_model(m); print("[exp] reparameterized")
except Exception as e:
    print("[exp] no reparameterize_model:", repr(e)[:80])

# extract preprocessing (size, mean, std, resize mode) from the torchvision Compose
import torchvision.transforms as T
size, mean, std, crop = None, None, None, None
for tr in pp.transforms:
    n = type(tr).__name__
    if n == "Resize":
        sz = tr.size; size = sz if isinstance(sz, int) else sz[0]
    if n == "CenterCrop":
        cs = tr.size; crop = cs if isinstance(cs, int) else cs[0]
    if n == "Normalize":
        mean = [float(x) for x in tr.mean]; std = [float(x) for x in tr.std]
print(f"[exp] preprocess: resize={size} crop={crop} mean={mean} std={std}")
IN = crop or size or 256

class Vis(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, px):
        e = self.model.encode_image(px)
        return torch.nn.functional.normalize(e, p=2, dim=1)

vis = Vis(m).eval()
dummy = torch.randn(1, 3, IN, IN)
with torch.no_grad():
    torch.onnx.export(vis, dummy, OUT, input_names=["px"], output_names=["emb"],
                      dynamic_axes={"px": {0: "B"}, "emb": {0: "B"}}, opset_version=17, dynamo=False)
mb = os.path.getsize(OUT) / 1e6
print(f"[exp] wrote {OUT} ({mb:.1f} MB fp32), input {IN}x{IN}")

# parity 1: torch vs onnxruntime on identical pixels
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
px = torch.randn(4, 3, IN, IN)
with torch.no_grad():
    t_emb = vis(px).numpy()
o_emb = sess.run(["emb"], {"px": px.numpy()})[0]
d_emb = float(np.abs(o_emb - t_emb).max())
print(f"[exp] PARITY vision torch vs ort: max|d|={d_emb:.2e} dim={o_emb.shape[1]}")

# parity 2: full chain vis->img_h'->packbits  (python torch) vs (onnx vis -> onnx imgh)
ck = torch.load("/tmp/imgh_mobileclip2-s2.pt", map_location="cpu")
bits = [int(b) for b in ck["bits"]]; bi = bits.index(1024)
head = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0); head.load_state_dict(ck["img_h"]); head.eval()
def pack(c):
    b = (np.asarray(c) > 0).astype(np.uint8)
    if b.ndim == 1: b = b[None, :]
    return np.packbits(b, axis=1, bitorder="big").astype(np.uint8)
with torch.no_grad():
    py_code = head(torch.from_numpy(t_emb))[bi]["binary"].numpy()
imgh_sess = ort.InferenceSession(f"{REPO}/web/static/onnx/img_h_mobileclip2-s2.onnx", providers=["CPUExecutionProvider"])
onnx_cont = imgh_sess.run(["code"], {"emb": o_emb})[0]
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
ham = int(LUT[np.bitwise_xor(pack(py_code), pack(onnx_cont))].sum())
print(f"[exp] PARITY full-chain (torch vis+head) vs (onnx vis+head) Hamming total over 4 imgs = {ham} (bits=4096)")
print(f"[exp] PARITY {'OK' if d_emb < 1e-3 and ham == 0 else 'CHECK'}")

json.dump({"model": "MobileCLIP2-S2", "input": IN, "resize": size, "crop": crop,
           "mean": mean, "std": std, "dim": int(o_emb.shape[1]), "onnx_mb": round(mb, 1)},
          open(f"{REPO}/web/static/onnx/vis_mobileclip2-s2.meta.json", "w"), indent=2)
print("[exp] DONE")
