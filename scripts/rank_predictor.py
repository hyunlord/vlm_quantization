"""Part 1 — is effective-dimension / entropy the governing predictor of binary R@10,
where the 5 prior proxies (cosine/parity/margin/neighbor/diralign) failed?

Cross-path (ceiling/distill/head-adapt, COCO 5K) + cross-language (XM3600 36-lang, ceiling path).
Variables per row:
  upstream_effdim : text-embedding PCA participation ratio — computed WITHOUT codes/retrieval (actionable)
  code_entropy    : sum per-bit entropy of 1024-bit codes (downstream; tautology-risk)
  code_rank       : matrix rank of the binary code matrix (downstream/structural)
  cosine          : mean paired cos(text_emb, image_emb)            [proxy, embedding]
  margin          : mean |z| pre-sign                                [proxy]
  diralign        : mean cos(text_code, paired image_code)           [proxy, pair-Hamming-derived → TAUTOLOGY-flag]
  parity          : mean bit-agreement of code vs reference code      [proxy; cross-path vs ceiling, cross-lang vs en]
  r10             : binary T2I R@10
Spearman ρ vs r10 computed cross-language (n=36). Run: .venv/bin/python scripts/rank_predictor.py
"""
from __future__ import annotations
import csv, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(REPO, "paper")
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; TXTH_E5="/tmp/txt_h_e5.pt"; E5_EN="/tmp/e5_test_en.pt"; XM="/tmp/xm_so400m_lc.pt"
NONLATIN={"ar","bn","el","fa","he","hi","ja","ko","ru","th","uk","zh","te","mi"}


def wcsv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote paper/{name} ({len(rows)})", flush=True)


def mk(state,e,h,b):
    m=NestedHashLayer(e,h,b,0.0).to(dev).eval(); m.load_state_dict(state); return m
def zf(head,x): return F.normalize(head.batch_norms[-1](head.hash_head(x)[:,:1024]),p=2,dim=1)
def effdim(X):
    X=X.detach().float(); Xc=X-X.mean(0,keepdim=True); s=torch.linalg.svdvals(Xc).cpu().numpy()**2
    return float((s.sum()**2)/(s**2).sum())
def code_entropy(code):
    p=((code>0).float().mean(0)).cpu().numpy(); p=np.clip(p,1e-6,1-1e-6)
    return float((-(p*np.log2(p)+(1-p)*np.log2(1-p))).sum())
def code_rank(code): return int(torch.linalg.matrix_rank(code.float()).item())
def margin(z): return float(z.abs().mean().item())
def ham(q,g): return (1024 - q@g.t())/2
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
caps=[str(c) for c in EC["captions"]]; ids=EC["ids"]; labels=(ids if torch.is_tensor(ids) else torch.tensor([int(x) for x in ids])).to(dev); N=img.shape[0]


def metrics(text_emb, text_z, text_code, gal_code, gold, ref_code=None, paired_img_emb=None):
    d=ham(text_code, gal_code); order=d.argsort(1); rel=(order==gold[:,None])
    r10=(rel[:,:10].sum(1)>0).float().mean().item()*100
    pair=d[torch.arange(text_code.shape[0]),gold]
    diralign=float((1 - 2*pair/1024).mean().item())
    cos=float(F.cosine_similarity(text_emb, paired_img_emb, dim=1).mean().item()) if paired_img_emb is not None else float("nan")
    parity=float((text_code==ref_code).float().mean().item()) if ref_code is not None else float("nan")
    return dict(r10=round(r10,2), upstream_effdim=round(effdim(text_emb),2), code_entropy=round(code_entropy(text_code),1),
                code_rank=code_rank(text_code), code_effrank=round(effdim(text_code.float()),2),
                cosine=round(cos,4), margin=round(margin(text_z),5),
                diralign=round(diralign,4), parity=round(parity,4) if not np.isnan(parity) else "")

rows=[]
# ---- cross-path (COCO) ----
with torch.no_grad():
    gz=zf(img_h,img); gcode=torch.sign(gz)
    pred=run_distill(caps); e5=F.normalize(torch.load(E5_EN,map_location="cpu").float().to(dev),dim=1)
    st=torch.load(TXTH_E5,map_location="cpu"); st=st["txt_h"] if isinstance(st,dict) and "txt_h" in st else st
    ha=mk(st,e5.shape[1],hidden,bits)
    cz=zf(txt_h,true_txt); ccode=torch.sign(cz)
gold_coco=torch.arange(N,device=dev)
PATHS={"ceiling":(true_txt,cz,ccode,img),"distillation":(pred,zf(txt_h,pred),torch.sign(zf(txt_h,pred)),img),
       "head-adapt":(e5,zf(ha,e5),torch.sign(zf(ha,e5)),None)}
for name,(te,z,code,pimg) in PATHS.items():
    m=metrics(te,z,code,gcode,gold_coco,ref_code=(None if name=="ceiling" else ccode),paired_img_emb=pimg)
    m={"axis":"path","key":name,**m}; rows.append(m)
    print(f"   [path] {name:>12}: effdim {m['upstream_effdim']:>6} codeH {m['code_entropy']:>6} rank {m['code_rank']} R@10 {m['r10']}", flush=True)

# ---- cross-lang (XM3600) ----
xm=torch.load(XM,map_location="cpu"); xi=F.normalize(xm["img_emb"].float().to(dev),dim=1)
with torch.no_grad(): xgcode=torch.sign(zf(img_h,xi))
en_code=None
for lg in ["en"]+[l for l in xm["per_lang"] if l!="en"]:
    pl=xm["per_lang"][lg]; te=F.normalize(pl["text_emb"].float().to(dev),dim=1)
    gold=torch.tensor([int(g) for g in pl["gold"]],device=dev)
    with torch.no_grad(): z=zf(txt_h,te); code=torch.sign(z)
    if lg=="en": en_code_full=code; en_gold=gold
    # parity vs en requires same indexing; en has its own gold ordering -> compare per-image mean code instead
    m=metrics(te,z,code,xgcode,gold,ref_code=None,paired_img_emb=xi[gold])
    m={"axis":"lang","key":lg,"script":"non-latin" if lg in NONLATIN else "latin",**m}
    rows.append(m)
    print(f"   [lang] {lg:>3}: effdim {m['upstream_effdim']:>6} codeH {m['code_entropy']:>6} rank {m['code_rank']} cos {m['cosine']} margin {m['margin']} R@10 {m['r10']}", flush=True)
# fill 'script' for path rows
for r in rows:
    r.setdefault("script","")
wcsv("rank_predictor.csv", rows)

# ---- Spearman cross-lang ----
lang=[r for r in rows if r["axis"]=="lang"]
y=np.array([r["r10"] for r in lang])
print("\n=== Spearman ρ vs R@10 (cross-language, n=36) ===", flush=True)
sp={}
for v,tag in [("upstream_effdim","UPSTREAM/actionable"),("code_effrank","downstream-effrank"),("code_entropy","downstream/tautology-risk"),
              ("cosine","proxy(embed)"),("margin","proxy"),("diralign","proxy/pair-Ham-derived")]:
    try:
        x=np.array([float(r[v]) for r in lang]); rho=spearmanr(x,y).correlation; sp[v]=rho
        print(f"   {v:>16} [{tag:>22}]: ρ = {rho:+.3f}", flush=True)
    except Exception as e:
        print(f"   {v}: skip ({e})", flush=True)
print("ANATOMY_RANK_DONE", flush=True)
