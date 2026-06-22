"""Export MobileCLIP2-S0 vision (fp32, ~45MB) + img_h'_s0 ONNX for a lighter ort-web-runnable model.
s0 dim=512 (same as s2) so the panel pipeline is identical. All-fp32 ops → runs in ort-web (unlike int8)."""
import os, sys, json, numpy as np, torch
from torch import nn
import open_clip
from pathlib import Path as _P; REPO=os.environ.get("REPO", str(_P(__file__).resolve().parent.parent)); sys.path.insert(0,REPO)
from src.models.nested_hash_layer import NestedHashLayer
O=f"{REPO}/web/static/onnx"; os.makedirs(O,exist_ok=True)
CODE_BITS=1024

# --- vision tower ---
m,_,pp=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",force_quick_gelu=False)
m=m.eval()
import torchvision.transforms as T
size=crop=None
for tr in pp.transforms:
    n=type(tr).__name__
    if n=="Resize": size=tr.size if isinstance(tr.size,int) else tr.size[0]
    if n=="CenterCrop": crop=tr.size if isinstance(tr.size,int) else tr.size[0]
IN=crop or size or 256
class Vis(nn.Module):
    def __init__(s,model): super().__init__(); s.m=model
    def forward(s,px): return torch.nn.functional.normalize(s.m.encode_image(px),p=2,dim=1)
vis=Vis(m).eval()
VOUT=f"{O}/vis_mobileclip2-s0.onnx"
with torch.no_grad():
    torch.onnx.export(vis, torch.randn(1,3,IN,IN), VOUT, input_names=["px"], output_names=["emb"],
        dynamic_axes={"px":{0:"B"},"emb":{0:"B"}}, opset_version=17, dynamo=False)
mb=os.path.getsize(VOUT)/1e6
import onnxruntime as ort
sess=ort.InferenceSession(VOUT,providers=["CPUExecutionProvider"])
px=torch.randn(4,3,IN,IN)
with torch.no_grad(): te=vis(px).numpy()
oe=sess.run(["emb"],{"px":px.numpy()})[0]
print(f"[s0] vis {mb:.1f}MB in={IN} dim={oe.shape[1]} torch-vs-ort max|d|={np.abs(oe-te).max():.2e}")
json.dump({"model":"MobileCLIP2-S0","input":IN,"resize":size,"crop":crop,"mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0],"dim":int(oe.shape[1]),"onnx_mb":round(mb,1)}, open(f"{O}/vis_mobileclip2-s0.meta.json","w"), indent=2)

# --- head img_h'_s0 ---
ck=torch.load("/tmp/imgh_mobileclip2-s0.pt",map_location="cpu")
bits=[int(b) for b in ck["bits"]]; bi=bits.index(CODE_BITS)
head=NestedHashLayer(ck["embed"],ck["hidden"],bits,0.0); head.load_state_dict(ck["img_h"]); head.eval()
class Head(nn.Module):
    def __init__(s,h,bit): super().__init__(); s.hash_head=h.hash_head; s.bn=h.batch_norms[h.bit_list.index(bit)]; s.bit=bit
    def forward(s,emb): return torch.tanh(torch.nn.functional.normalize(s.bn(s.hash_head(emb)[:,:s.bit]),p=2,dim=1))
hw=Head(head,CODE_BITS).eval()
HOUT=f"{O}/img_h_mobileclip2-s0.onnx"
with torch.no_grad():
    torch.onnx.export(hw, torch.randn(1,ck["embed"]), HOUT, input_names=["emb"], output_names=["code"],
        dynamic_axes={"emb":{0:"B"},"code":{0:"B"}}, opset_version=17, dynamo=False)
print(f"[s0] head {os.path.getsize(HOUT)/1e6:.1f}MB embed={ck['embed']}")
print("[s0] DONE")
