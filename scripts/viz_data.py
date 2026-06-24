"""Dump JSON for the interactive viz/ HTML (no heavy compute in browser).
Reuses ft113 anchor + 3 text paths (ceiling/distill/head-adapt) + image anchor.
Outputs viz/data/*.json. Run: .venv/bin/python scripts/viz_data.py

SCHEMA (documented for the HTML authors):
 forward_stages.json : {paths:{name:{samples:[{idx,sliced:[1024],bn:[1024],z:[1024]}], ...}}, note}
 distributions.json  : {paths:{name:{bn:{absz_hist:{x:[],y:[]}, bit_pos_frac:[1024]},
                                      nobn:{...}, corr:[[96x96]]}}, bins, note}
 sign_info.json      : {bits:[64,256,1024], queries:[{qid, lang, paths:{name:{ bit:{ cont_top:[{g,score,ham,gold}],
                                      ham_top:[{g,ham,score,gold}], gold_cont_rank, gold_ham_rank }}}}], note}
 embedding_2d.json   : {siglip:{umap:[[x,y]],pca:[[x,y]],label:[]}, e5:{...}, code:{...},
                        langs:{umap:[[x,y]],lang:[]}, links:[[i_img,i_txt]], note}
 retrieval_anatomy.json:{paths:{name:{queries:[{qid,success,ham_gold,ham_nwrong,
                                      ham_hist:{x,y}, xor_gold_hex}]}}, note}
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = os.environ.get("REPO", "/home/hyunlord/github/vlm_quantization"); sys.path.insert(0, REPO)
from src.models.nested_hash_layer import NestedHashLayer
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(REPO, "viz", "data"); os.makedirs(OUT, exist_ok=True)
HEAD_PATH="/tmp/ft_ko_113.pt"; EC_PATH="/tmp/emb_cache.pt"; DISTILL="/tmp/distill_e5.pt"; TXTH_E5="/tmp/txt_h_e5.pt"; E5_EN="/tmp/e5_test_en.pt"; XM="/tmp/xm_so400m_lc.pt"
torch.manual_seed(0); rng = np.random.RandomState(0)
def r3(a): return [round(float(v), 3) for v in np.asarray(a).ravel()]
def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w")); print(f"  wrote viz/data/{name} ({os.path.getsize(os.path.join(OUT,name))//1024} KB)", flush=True)
def mk(state,e,h,b): m=NestedHashLayer(e,h,b,0.0).to(dev).eval(); m.load_state_dict(state); return m
def stages(head,x):
    raw=head.hash_head(x)[:,:1024]; bn=head.batch_norms[-1](raw); z=F.normalize(bn,p=2,dim=1)
    znob=F.normalize(raw,p=2,dim=1)  # BN-off: L2 of raw slice
    return raw, bn, z, znob, torch.sign(z)
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
caps=[str(c) for c in EC["captions"]]; N=img.shape[0]; lab=torch.arange(N,device=dev)
with torch.no_grad():
    pred=run_distill(caps); e5=F.normalize(torch.load(E5_EN,map_location="cpu").float().to(dev),dim=1)
    st=torch.load(TXTH_E5,map_location="cpu"); st=st["txt_h"] if isinstance(st,dict) and "txt_h" in st else st
    ha=mk(st,e5.shape[1],hidden,bits)
# stages per source
SRC={"image":(img,img_h),"ceiling":(true_txt,txt_h),"distillation":(pred,txt_h),"head-adapt":(e5,ha)}
ST={}
with torch.no_grad():
    for nm,(x,h) in SRC.items():
        raw,bn,z,znob,b=stages(h,x); ST[nm]=dict(raw=raw,bn=bn,z=z,znob=znob,b=b)
gz=ST["image"]["z"]; gcode=ST["image"]["b"]
print(f"loaded N={N}", flush=True)

# ---------------- 1) forward_stages ----------------
S=12; samp=rng.choice(N,S,replace=False).tolist()
fs={"paths":{}, "bits":[64,256,1024], "note":"per-sample 1024-d vectors at stages sliced(raw)->bn->z(pre-sign). sign(z)=±1."}
for nm in SRC:
    fs["paths"][nm]={"samples":[{"idx":int(i),
        "sliced":r3(ST[nm]["raw"][i].cpu()),"bn":r3(ST[nm]["bn"][i].cpu()),"z":r3(ST[nm]["z"][i].cpu())} for i in samp]}
dump("forward_stages.json", fs)

# ---------------- 2) distributions (BN on/off) ----------------
def hist(a,lo,hi,nb=80):
    h,e=np.histogram(np.asarray(a),bins=nb,range=(lo,hi)); return {"x":r3(e[:-1]),"y":[int(v) for v in h]}
def corr96(code):
    c=code.cpu().numpy(); cm=np.corrcoef(c[:, :1024].T)  # 1024x1024
    # block-average to 96x96
    B=1024//96 if 1024%96==0 else None
    g=1024//16  # use 64x64 by 16-block -> 64
    k=16; m=1024//k  # 64
    cm2=cm[:m*k,:m*k].reshape(m,k,m,k).mean(axis=(1,3))
    return [[round(float(v),3) for v in row] for row in cm2]
dist={"paths":{}, "note":"|z| hist + per-bit +1 fraction + 64x64 bit-corr, for BN (current) vs BN-off (L2 of raw). bits=1024."}
for nm in SRC:
    z=ST[nm]["z"]; znob=ST[nm]["znob"]; b=ST[nm]["b"]; bnob=torch.sign(znob)
    dist["paths"][nm]={
      "bn":{"absz_hist":hist(z.abs().cpu().numpy().ravel(),0,float(np.percentile(z.abs().cpu().numpy(),99.5))),
            "bit_pos_frac":r3(((b>0).float().mean(0)).cpu()), "corr":corr96(b)},
      "nobn":{"absz_hist":hist(znob.abs().cpu().numpy().ravel(),0,float(np.percentile(znob.abs().cpu().numpy(),99.5))),
            "bit_pos_frac":r3(((bnob>0).float().mean(0)).cpu()), "corr":corr96(bnob)},
    }
dump("distributions.json", dist)

# ---------------- 3) sign_info (continuous vs Hamming ranking) ----------------
def perbit(head, x, bit):
    """faithful nested code at `bit` using the head's own per-bit BatchNorm."""
    raw=head.hash_head(x)[:, :bit]; bn=head.batch_norms[bits.index(bit)](raw)
    z=F.normalize(bn,p=2,dim=1); return z, torch.sign(z)
with torch.no_grad():  # per-bit gallery (image) z + code, faithful
    GB={bit:perbit(img_h,img,bit) for bit in (64,256,1024)}
def topk_lists(qz_b, qb_b, bit, k=15):
    gzb,gcb=GB[bit]
    cs=(qz_b @ gzb.t()).cpu().numpy()                 # continuous score z·z (unit per bit), higher=near
    hm=((bit - qb_b @ gcb.t())/2).cpu().numpy()       # Hamming, lower=near
    return cs, hm, np.argsort(-cs), np.argsort(hm)
# choose queries: 12 success + 12 fail (by ceiling 1024 Hamming)
with torch.no_grad():
    d=( (1024 - ST["ceiling"]["b"] @ gcode.t())/2 )
ok=(d.argmin(1)==lab).cpu().numpy()
succ_ids=np.where(ok)[0]; fail_ids=np.where(~ok)[0]
qsel=list(rng.choice(succ_ids,8,replace=False))+list(rng.choice(fail_ids,8,replace=False)) if len(fail_ids)>=8 else list(rng.choice(N,16,replace=False))
si={"bits":[64,256,1024],"queries":[],"note":"per query: gallery ranked by continuous z·z (score, higher=near) vs Hamming (lower=near). gold=paired image. shows what sign discards."}
QHEAD={"ceiling":(txt_h,true_txt),"distillation":(txt_h,pred),"head-adapt":(ha,e5)}
with torch.no_grad():
    QB={nm:{bit:perbit(h,x,bit) for bit in (64,256,1024)} for nm,(h,x) in QHEAD.items()}
for qid in qsel:
    qid=int(qid); entry={"qid":qid,"caption":caps[qid][:80],"paths":{}}
    for nm in ("ceiling","distillation","head-adapt"):
        entry["paths"][nm]={}
        for bit in (64,256,1024):
            qz_b=QB[nm][bit][0][qid]; qb_b=QB[nm][bit][1][qid]
            cs,hm,co,ho=topk_lists(qz_b,qb_b,bit)
            gold_cr=int(np.where(co==qid)[0][0])+1; gold_hr=int(np.where(ho==qid)[0][0])+1
            entry["paths"][nm][str(bit)]={
              "cont_top":[{"g":int(g),"score":round(float(cs[g]),3),"ham":int(hm[g]),"gold":int(g==qid)} for g in co[:15]],
              "ham_top":[{"g":int(g),"ham":int(hm[g]),"score":round(float(cs[g]),3),"gold":int(g==qid)} for g in ho[:15]],
              "gold_cont_rank":gold_cr,"gold_ham_rank":gold_hr}
        entry["success"]=int(ok[qid])
    si["queries"].append(entry)
dump("sign_info.json", si)

# ---------------- 4) embedding_2d (precomputed UMAP+PCA) ----------------
from sklearn.decomposition import PCA
import umap
def proj(X, n=700):
    idx=rng.choice(X.shape[0],min(n,X.shape[0]),replace=False)
    Xs=X[idx]
    p=PCA(n_components=2).fit_transform(Xs)
    u=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=0).fit_transform(Xs)
    return idx, p, u
emb2={"note":"precomputed 2D. siglip space = image+ceiling+distill (1152). e5/code separate spaces. langs=XM3600 per-lang text."}
# siglip joint
Xj=np.concatenate([ST[n]["raw"].cpu().numpy() if False else SRC[n][0].cpu().numpy() for n in ("image","ceiling","distillation")],0)
labj=np.concatenate([[n]*SRC[n][0].shape[0] for n in ("image","ceiling","distillation")])
ij=rng.choice(Xj.shape[0],2100,replace=False)
p=PCA(2).fit_transform(Xj[ij]); u=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=0).fit_transform(Xj[ij])
emb2["siglip"]={"pca":[[round(float(a),3),round(float(b),3)] for a,b in p],
                "umap":[[round(float(a),3),round(float(b),3)] for a,b in u],
                "label":[labj[i] for i in ij]}
# code space (binary) joint image+ceiling+distill
Cj=np.concatenate([ST[n]["b"].cpu().numpy() for n in ("image","ceiling","distillation")],0)[ij].astype(np.float32)
uc=umap.UMAP(n_neighbors=30,min_dist=0.1,random_state=0,metric="hamming").fit_transform(Cj)
emb2["code"]={"umap":[[round(float(a),3),round(float(b),3)] for a,b in uc],"label":[labj[i] for i in ij]}
# languages (XM3600 text emb)
xm=torch.load(XM,map_location="cpu")
langs=["en","de","es","fr","ru","ar","ko","zh","th","hi","te","sw"]; langs=[l for l in langs if l in xm["per_lang"]]
XL=[]; LL=[]
for l in langs:
    te=xm["per_lang"][l]["text_emb"].float().numpy(); k=rng.choice(te.shape[0],150,replace=False)
    XL.append(te[k]); LL+=[l]*150
XL=np.concatenate(XL,0); ul=umap.UMAP(n_neighbors=30,min_dist=0.2,random_state=0).fit_transform(XL)
emb2["langs"]={"umap":[[round(float(a),3),round(float(b),3)] for a,b in ul],"lang":LL}
dump("embedding_2d.json", emb2)

# ---------------- 5) retrieval_anatomy ----------------
def to_hex(bits_bool):
    return np.packbits(np.asarray(bits_bool,dtype=np.uint8)).tobytes().hex()
ra={"paths":{}, "note":"per query: Hamming to gold image vs nearest-wrong + full Hamming hist; xor_gold_hex=1024-bit code XOR with gold image code (which bits differ)."}
qra=list(rng.choice(succ_ids,10,replace=False))+list(rng.choice(fail_ids,10,replace=False)) if len(fail_ids)>=10 else list(rng.choice(N,20,replace=False))
for nm in ("ceiling","distillation","head-adapt"):
    qb_all=ST[nm]["b"]; ra["paths"][nm]={"queries":[]}
    d=((1024 - qb_all @ gcode.t())/2)
    succ_nm=(d.argmin(1)==lab).cpu().numpy()
    for qid in qra:
        qid=int(qid); dh=d[qid].cpu().numpy()
        gold=dh[qid]; nw=dh.copy(); nw[qid]=1e9; nwrong=float(nw.min())
        xor=((qb_all[qid]>0)!=(gcode[qid]>0)).cpu().numpy()
        ra["paths"][nm]["queries"].append({"qid":qid,"success":int(succ_nm[qid]),
            "ham_gold":int(gold),"ham_nwrong":int(nwrong),
            "ham_hist":hist(dh,0,1024,64),"xor_gold_hex":to_hex(xor)})
dump("retrieval_anatomy.json", ra)
print("VIZ_DATA_DONE", flush=True)
