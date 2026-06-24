"""Part 2 — q19-type outlier debug. COCO 5K T2I, 1024-bit, gold=diagonal.
Failure taxonomy among queries whose gold is NOT in top-10:
  confusable (cand A = label limit): the rank-1 (wrong) image is CLOSE (small Hamming) -> likely a
                                     semantic near-duplicate, instance label only differs.
  isolated   (cand B = q19 type)   : the rank-1 (wrong) image is also FAR (large Hamming) AND gold far
                                     -> nothing matches well; query code sits in a sparse spot.
Outputs paper/outlier_debug.csv (rates) + viz/data/outlier_cases.json (cases w/ caption+ids+filepaths+hamming)
+ thumbnails viz/data/thumbs_outlier/{id}.jpg for the sampled cases (gold + top-3 wrong).
Run: .venv/bin/python scripts/outlier_debug.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO=os.environ.get("REPO","/home/hyunlord/github/vlm_quantization"); sys.path.insert(0,REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev="cuda" if torch.cuda.is_available() else "cpu"
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; TXTH_E5="/tmp/txt_h_e5.pt"; E5_EN="/tmp/e5_test_en.pt"
COCO=os.path.join(REPO,"data","coco"); DJSON=os.path.join(COCO,"dataset_coco.json")
OUTp=os.path.join(REPO,"paper"); OUTv=os.path.join(REPO,"viz","data"); THUMB=os.path.join(OUTv,"thumbs_outlier")
os.makedirs(OUTp,exist_ok=True); os.makedirs(THUMB,exist_ok=True); rng=np.random.RandomState(0)
def mk(s,e,h,b): m=NestedHashLayer(e,h,b,0.0).to(dev).eval(); m.load_state_dict(s); return m
def run_distill(caps):
    from transformers import AutoModel, AutoTokenizer
    ck=torch.load(DISTILL,map_location="cpu"); ml=int(ck["maxlen"])
    m=AutoModel.from_pretrained(ck["student"]); m.load_state_dict(ck["backbone"]); m=m.to(dev).eval()
    tok=AutoTokenizer.from_pretrained(ck["student"])
    proj=nn.Linear(ck["proj"]["weight"].shape[1],ck["proj"]["weight"].shape[0]); proj.load_state_dict(ck["proj"]); proj=proj.to(dev).eval()
    out=[]
    with torch.no_grad():
        for s in range(0,len(caps),256):
            t=tok(caps[s:s+256],padding="max_length",max_length=ml,truncation=True,return_tensors="pt")
            o=m(t["input_ids"].to(dev),t["attention_mask"].to(dev)).last_hidden_state
            msk=t["attention_mask"].to(dev).unsqueeze(-1).float()
            out.append(F.normalize(proj((o*msk).sum(1)/msk.sum(1).clamp(min=1e-9)),dim=1).cpu())
    return torch.cat(out,0).to(dev)
# id -> relative path map from Karpathy json
idpath={}
dj=json.load(open(DJSON))
for im in dj["images"]:
    idpath[int(im["cocoid"])]=f"{im['filepath']}/{im['filename']}"
ck=torch.load(HEAD_PATH,map_location="cpu"); bits,embed,hidden=[int(b) for b in ck["bits"]],ck["embed"],ck["hidden"]
img_h=mk(ck["img_h"],embed,hidden,bits); txt_h=mk(ck["txt_h"],embed,hidden,bits)
EC=torch.load(EC_PATH,map_location="cpu")["test"]
img=F.normalize(EC["img"].float().to(dev),dim=1); true_txt=F.normalize(EC["txt"].float().to(dev),dim=1)
caps=[str(c) for c in EC["captions"]]; ids=[int(x) for x in EC["ids"]]; N=img.shape[0]
with torch.no_grad():
    pred=run_distill(caps); e5=F.normalize(torch.load(E5_EN,map_location="cpu").float().to(dev),dim=1)
    st=torch.load(TXTH_E5,map_location="cpu"); st=st["txt_h"] if isinstance(st,dict) and "txt_h" in st else st
    ha=mk(st,e5.shape[1],hidden,bits)
PATHS={"ceiling":(txt_h,true_txt),"distillation":(txt_h,pred),"head-adapt":(ha,e5)}
def codes(head,x):
    with torch.no_grad():
        raw=head.hash_head(x)[:,:1024]; z=F.normalize(head.batch_norms[-1](raw),p=2,dim=1); return torch.sign(z)
bg=codes(img_h,img)
CLOSE=226  # global median paired Hamming (from anatomy); rank-1 wrong below this = "close/confusable"
summary=[]; cases=[]; thumb_ids=set()
for path,(head,x) in PATHS.items():
    bq=codes(head,x)
    d=((1024-bq@bg.t())/2)  # (N,N) Hamming
    order=d.argsort(1)
    gold_rank=(order==torch.arange(N,device=dev)[:,None]).float().argmax(1)+1
    gold_ham=d[torch.arange(N),torch.arange(N)]
    top1=order[:,0]; top1_ham=d[torch.arange(N),top1]
    fail=(gold_rank>10).cpu().numpy()
    gh=gold_ham.cpu().numpy(); t1=top1_ham.cpu().numpy()
    gh_med=np.median(gh)
    # taxonomy among fails
    far_gold = gh> gh_med
    confusable = fail & (t1< CLOSE)              # a close wrong exists (label-limit candidate)
    isolated   = fail & (t1>=CLOSE)              # even rank-1 is far (q19/isolated candidate)
    summary.append({"path":path,"n":N,"fail_pct":round(100*fail.mean(),2),
                    "confusable_pct":round(100*confusable.mean(),2),"isolated_pct":round(100*isolated.mean(),2),
                    "fail_gold_ham_med":round(float(np.median(gh[fail])),1),"fail_top1_ham_med":round(float(np.median(t1[fail])),1),
                    "close_thresh":CLOSE})
    print(f"  {path:>12}: fail {100*fail.mean():.1f}% | confusable(close wrong) {100*confusable.mean():.1f}% | isolated(far wrong) {100*isolated.mean():.1f}% | fail gold_ham med {np.median(gh[fail]):.0f} top1_ham med {np.median(t1[fail]):.0f}",flush=True)
    if path=="ceiling":  # build case gallery from the deployed-anchor path
        ci=list(rng.choice(np.where(confusable)[0],min(10,confusable.sum()),replace=False))
        io=list(rng.choice(np.where(isolated)[0],min(10,isolated.sum()),replace=False))
        for typ,qs in [("confusable",ci),("isolated",io)]:
            for q in qs:
                q=int(q); top3=[int(t) for t in order[q,:3].cpu().numpy()]
                cases.append({"type":typ,"qid_row":q,"caption":caps[q],"gold_id":ids[q],"gold_path":idpath.get(ids[q],""),
                              "gold_ham":int(gh[q]),"gold_rank":int(gold_rank[q].item()),
                              "top3":[{"id":ids[t],"path":idpath.get(ids[t],""),"ham":int(d[q,t].item())} for t in top3]})
                thumb_ids.add(ids[q]); [thumb_ids.add(ids[t]) for t in top3]
with open(os.path.join(OUTp,"outlier_debug.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
json.dump({"cases":cases,"note":"ceiling path, 1024-bit. confusable=rank1-wrong close (<226 Hamming, label-limit candidate A); isolated=rank1-wrong far (>=226, q19/new-failure candidate B). HUMAN must eyeball thumbs to confirm A vs B."}, open(os.path.join(OUTv,"outlier_cases.json"),"w"))
# thumbnails
from PIL import Image
nok=0
for cid in thumb_ids:
    rp=idpath.get(cid);
    if not rp: continue
    src=os.path.join(COCO,rp); dst=os.path.join(THUMB,f"{cid}.jpg")
    try:
        im=Image.open(src).convert("RGB"); im.thumbnail((180,180)); im.save(dst,quality=80); nok+=1
    except Exception as e:
        print(f"  thumb fail {cid}: {e}",flush=True)
print(f"thumbs: {nok}/{len(thumb_ids)} saved -> {THUMB}",flush=True)
print("OUTLIER_DONE -> paper/outlier_debug.csv, viz/data/outlier_cases.json, thumbs_outlier/",flush=True)
