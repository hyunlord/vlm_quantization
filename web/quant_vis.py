"""Quantize vis_mobileclip2-s2.onnx -> int8 (dynamic) + fp16 for browser; parity vs fp32 (full chain)."""
import os, sys, numpy as np, torch
REPO = os.path.expanduser("~/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
import onnxruntime as ort
F32 = f"{REPO}/web/static/onnx/vis_mobileclip2-s2.onnx"
I8  = f"{REPO}/web/static/onnx/vis_mobileclip2-s2.int8.onnx"
F16 = f"{REPO}/web/static/onnx/vis_mobileclip2-s2.fp16.onnx"
IN = 256
def pack(c):
    b=(np.asarray(c)>0).astype(np.uint8); b=b[None,:] if b.ndim==1 else b
    return np.packbits(b,axis=1,bitorder="big").astype(np.uint8)
LUT=np.array([bin(i).count("1") for i in range(256)],dtype=np.uint16)

# int8 dynamic
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(F32, I8, weight_type=QuantType.QInt8)
print(f"[q] int8 {os.path.getsize(I8)/1e6:.1f} MB")
# fp16
try:
    from onnxconverter_common import float16
    import onnx
    onnx.save(float16.convert_float_to_float16(onnx.load(F32), keep_io_types=True), F16)
    print(f"[q] fp16 {os.path.getsize(F16)/1e6:.1f} MB")
except Exception as e:
    print("[q] fp16 skip:", repr(e)[:80])

# parity: fp32 vs int8/fp16 full chain (vis -> img_h') on random pixels
ck=torch.load("/tmp/imgh_mobileclip2-s2.pt",map_location="cpu"); bits=[int(b) for b in ck["bits"]]
imgh=ort.InferenceSession(f"{REPO}/web/static/onnx/img_h_mobileclip2-s2.onnx",providers=["CPUExecutionProvider"])
px=np.random.randn(8,3,IN,IN).astype(np.float32)
def chain(vp):
    s=ort.InferenceSession(vp,providers=["CPUExecutionProvider"])
    emb=s.run(["emb"],{"px":px})[0]
    return pack(imgh.run(["code"],{"emb":emb})[0])
c32=chain(F32)
for tag,vp in [("int8",I8),("fp16",F16)]:
    if not os.path.exists(vp): continue
    ck2=chain(vp); ham=int(LUT[np.bitwise_xor(c32,ck2)].sum())
    print(f"[q] PARITY {tag} vs fp32 full-chain Hamming total = {ham} / {8*1024} bits ({100*ham/(8*1024):.2f}%)")
print("[q] DONE")
