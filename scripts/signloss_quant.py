"""Part 1 — sign information-loss quantification (A-bet GO/NO-GO).
Per bit {64,128,256,512,1024} × path {ceiling,distill,head-adapt}, COCO 5K T2I, gold=diagonal.
Three retrieval modes (gallery=image, query=text):
  continuous : score = z_q · z_g        (both pre-sign, unit) — ceiling before sign
  asymmetric : score = z_q · b_g        (query continuous, gallery 1-bit) — deployable (gallery stays binary)
  hamming    : score = b_q · b_g        (current deployed)
Metrics: R@10 for each + gaps; Kendall/Spearman(top-100) cont-vs-ham; top-10 overlap; gold-rank shift.
Outputs paper/signloss_quant.csv + viz/data/signloss.json. Run: .venv/bin/python scripts/signloss_quant.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
REPO=os.environ.get("REPO","/home/hyunlord/github/vlm_quantization"); sys.path.insert(0,REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev="cuda" if torch.cuda.is_available() else "cpu"
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; TXTH_E5="/tmp/txt_h_e5.pt"; E5_EN="/tmp/e5_test_en.pt"
OUTp=os.path.join(REPO,"paper"); OUTv=os.path.join(REPO,"viz","data"); os.makedirs(OUTp,exist_ok=True); os.makedirs(OUTv,exist_ok=True)
BITS=[64,128,256,512,1024]; rng=np.random.RandomState(0)
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
ck=torch.load(HEAD_PATH,map_location="cpu"); bits,embed,hidden=[int(b) for b in ck["bits"]],ck["embed"],ck["hidden"]
img_h=mk(ck["img_h"],embed,hidden,bits); txt_h=mk(ck["txt_h"],embed,hidden,bits)
EC=torch.load(EC_PATH,map_location="cpu")["test"]
img=F.normalize(EC["img"].float().to(dev),dim=1); true_txt=F.normalize(EC["txt"].float().to(dev),dim=1)
caps=[str(c) for c in EC["captions"]]; N=img.shape[0]; gold=torch.arange(N,device=dev)
with torch.no_grad():
    pred=run_distill(caps); e5=F.normalize(torch.load(E5_EN,map_location="cpu").float().to(dev),dim=1)
    st=torch.load(TXTH_E5,map_location="cpu"); st=st["txt_h"] if isinstance(st,dict) and "txt_h" in st else st
    ha=mk(st,e5.shape[1],hidden,bits)
PATHS={"ceiling":(txt_h,true_txt),"distillation":(txt_h,pred),"head-adapt":(ha,e5)}
def perbit(head,x,b):
    raw=head.hash_head(x)[:,:b]; z=F.normalize(head.batch_norms[bits.index(b)](raw),p=2,dim=1); return z, torch.sign(z)
def r10(score):  # higher=nearer; gold=diagonal
    order=score.argsort(1,descending=True); rel=(order==gold[:,None]); return round((rel[:,:10].sum(1)>0).float().mean().item()*100,2)
print(f"N={N}",flush=True); rows=[]; samp=rng.choice(N,200,replace=False)
for path,(head,x) in PATHS.items():
    for b in BITS:
        with torch.no_grad():
            zg,bg=perbit(img_h,img,b); zq,bq=perbit(head,x,b)
            Sc=zq@zg.t(); Sh=bq@bg.t(); Sa=zq@bg.t()
        rc,ra,rh=r10(Sc),r10(Sa),r10(Sh)
        # ranking-agreement on a query sample (cont vs ham), top-100 by continuous
        ktau=[]; sp=[]; ov=[]; shift=[]
        Scn=Sc.cpu().numpy(); Shn=Sh.cpu().numpy()
        for q in samp:
            sc=Scn[q]; sh=Shn[q]
            t100=np.argsort(-sc)[:100]
            sp.append(spearmanr(sc[t100],sh[t100]).correlation)
            ktau.append(kendalltau(sc[t100],sh[t100]).correlation)
            ov.append(len(set(np.argsort(-sc)[:10]) & set(np.argsort(-sh)[:10]))/10.0)
            gr_c=int(np.where(np.argsort(-sc)==q)[0][0])+1; gr_h=int(np.where(np.argsort(-sh)==q)[0][0])+1
            shift.append(gr_h-gr_c)
        row={"path":path,"bit":b,"r10_continuous":rc,"r10_asym":ra,"r10_hamming":rh,
             "gap_cont_ham":round(rc-rh,2),"gap_asym_ham":round(ra-rh,2),
             "kendall_top100":round(float(np.nanmean(ktau)),3),"spearman_top100":round(float(np.nanmean(sp)),3),
             "top10_overlap":round(float(np.mean(ov)),3),"gold_rank_shift_med":int(np.median(shift))}
        rows.append(row)
        print(f"  {path:>12} {b:>4}b: R@10 cont {rc} / asym {ra} / ham {rh} | gap c-h {rc-rh:+.2f} a-h {ra-rh:+.2f} | top10ov {np.mean(ov):.2f} kτ {np.nanmean(ktau):.2f}",flush=True)
with open(os.path.join(OUTp,"signloss_quant.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({"rows":rows,"note":"R@10 continuous(z·z) / asymmetric(z_q·b_g, deployable) / hamming(b·b); gaps = sign loss / search-only-recoverable"}, open(os.path.join(OUTv,"signloss.json"),"w"))
print("SIGNLOSS_DONE -> paper/signloss_quant.csv, viz/data/signloss.json",flush=True)
