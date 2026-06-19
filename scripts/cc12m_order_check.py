"""Determine which tar dir is the FIRST embedding batch (rows [0:403280]) vs the
SECOND (rows [403280:]). Embed the first kept image of each dir and cosine-compare
to big.pt row 0 and row 403280. cos~1 reveals the mapping."""
import sys, os, io, glob, tarfile
import torch, torch.nn.functional as F
from PIL import Image
REPO = "/home/hyunlord/github/vlm_quantization"; sys.path.insert(0, REPO)
MODEL = "google/siglip2-so400m-patch14-384"
big = torch.load("/tmp/cc12m_pairs_big.pt", map_location="cpu")["img_emb"]   # normalized (806560,1152)
row0, rowB2 = big[0], big[403280]
from transformers import AutoModel, SiglipImageProcessor
ip = SiglipImageProcessor.from_pretrained(MODEL)
bb = AutoModel.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
try: del bb.text_model
except Exception: pass
def pool(o): return o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
def first_img(tdir):
    t = sorted(glob.glob(tdir + "/*.tar"))[0]
    with tarfile.open(t) as tar:
        for m in tar:
            if m.name.lower().endswith(".jpg"):
                b = tar.extractfile(m).read()
                return os.path.basename(t), os.path.splitext(os.path.basename(m.name))[0], Image.open(io.BytesIO(b)).convert("RGB")
def emb(im):
    px = ip(images=im, return_tensors="pt")["pixel_values"].cuda()
    with torch.no_grad():
        v = pool(bb.vision_model(pixel_values=px)).float()
    return F.normalize(v, dim=1)[0].cpu()
for d in ["/tmp/cc12m", "/tmp/cc12m2"]:
    tar, k, im = first_img(d)
    e = emb(im)
    print(f"{d} ({tar} key={k}): cos(row0)={float(e@row0):.3f}  cos(row403280)={float(e@rowB2):.3f}", flush=True)
print("ORDER_CHECK_DONE", flush=True)
