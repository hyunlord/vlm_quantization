"""Image-side head-adaptation: train img_h'_E mapping a frozen vision encoder E's image embeddings into
the FROZEN SigLIP2-So400m 1024-bit code space (the anchor that built index.bin). Mirror of headadapt_train.py
(text txt_h'), but the "student" is an IMAGE tower and pairs are the SAME image (E_emb_i, SigLIP2_emb_i).

Quantifies on-device image indexing: can a small/mobile/SSL encoder on a phone hash a new photo into the
SAME code space as the shipped index? Per-encoder: encode COCO-train subset + 5K test with E (frozen) ->
train img_h'_E (CombinedHashLoss, io=frozen ft113 img_h(SigLIP2_emb) anchor, to=img_h'_E(E_emb)) -> eval.

Eval on COCO 5K test (EN/KO): (a) E-indexed gallery searched by server SigLIP2-text and offline e5 queries
R@{1,5,10}; (b) mixed gallery (half SigLIP2 codes + half E codes) R@10 — phone-added photos coexisting with
the shipped index; (c) code fidelity = per-image Hamming(E code, SigLIP2 code) + top-10 overlap. Cost: params,
int8 MB (= params bytes), GB10 encode img/s, on-device availability. Appends paper/image_encoders_headadapt.csv.

Anchors UNCHANGED: ft113 img_h, index code space, text side. Run per encoder:
  HEAD_PATH=/tmp/ft_ko_113.pt TRAIN_N=30000 .venv/bin/python web/paper_image_headadapt.py --enc siglip2-base
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.losses.combined import CombinedHashLoss  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

PAPER = Path(REPO) / "paper"
HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
TRAIN_N = int(os.environ.get("TRAIN_N", "30000"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BS = 512
KS = (1, 5, 10)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# encoder registry: type + loader id + on-device availability tag
ENCODERS = {
    "siglip2-base":   {"type": "tf-siglip", "id": "google/siglip2-base-patch16-256", "ondevice": "TJS(siglip)"},
    "dinov2-small":   {"type": "tf-dinov2", "id": "facebook/dinov2-small", "ondevice": "TJS"},
    "dinov2-base":    {"type": "tf-dinov2", "id": "facebook/dinov2-base", "ondevice": "TJS"},
    # MobileCLIP2 dfndr2b weights are quick_gelu=FALSE (open_clip warns if forced True) — do NOT force.
    "mobileclip2-s0": {"type": "oc", "id": "MobileCLIP2-S0", "pretrained": "dfndr2b", "qgelu": False, "ondevice": "TJS/WebGPU"},
    "mobileclip2-s2": {"type": "oc", "id": "MobileCLIP2-S2", "pretrained": "dfndr2b", "qgelu": False, "ondevice": "TJS/WebGPU"},
    "mobileclip2-s4": {"type": "oc", "id": "MobileCLIP2-S4", "pretrained": "dfndr2b", "qgelu": False, "ondevice": "export"},
    "eva02-b16":      {"type": "oc", "id": "EVA02-B-16", "pretrained": "merged2b_s8b_b131k", "ondevice": "export"},
    "openvision-s16": {"type": "oc-hf", "id": "hf-hub:UCSC-VLAA/openvision-vit-small-patch16-224", "ondevice": "export"},
    "tinyclip-8m":    {"type": "oc", "id": "TinyCLIP-ViT-8M-16-Text-3M", "pretrained": "YFCC15M", "ondevice": "export"},
}


def autocast():
    import contextlib
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()


def load_encoder(spec):
    """Return (encode_fn(paths)->Tensor(N,D) float, params:int, dim:int, label:str)."""
    t = spec["type"]
    if t in ("oc", "oc-hf"):
        import open_clip
        if t == "oc-hf":
            model, _, preprocess = open_clip.create_model_and_transforms(spec["id"])
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                spec["id"], pretrained=spec["pretrained"], force_quick_gelu=spec.get("qgelu", False))
        model = model.to(dev).eval()
        visual = model.visual
        params = sum(p.numel() for p in visual.parameters())

        @torch.no_grad()
        def enc(paths, batch=128):
            out = []
            for s in range(0, len(paths), batch):
                px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(dev)
                with autocast():
                    e = model.encode_image(px)
                out.append(e.float().cpu())
            return torch.cat(out)
        # infer dim from a tiny run is costly; read from a dummy forward at call time -> set after first batch
        return enc, params, None, spec["id"]
    elif t == "tf-siglip":
        from transformers import AutoModel
        m = AutoModel.from_pretrained(spec["id"], dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        try:  # AutoProcessor is tf5.1-broken for siglip2 (None model_type) -> SiglipImageProcessor directly
            from transformers import AutoProcessor
            _proc = AutoProcessor.from_pretrained(spec["id"])
            proc_px = lambda imgs: _proc(images=imgs, return_tensors="pt")["pixel_values"]
        except Exception:
            from transformers import SiglipImageProcessor
            _ip = SiglipImageProcessor.from_pretrained(spec["id"])
            proc_px = lambda imgs: _ip(images=imgs, return_tensors="pt")["pixel_values"]
        params = sum(p.numel() for p in m.vision_model.parameters())

        @torch.no_grad()
        def enc(paths, batch=128):
            out = []
            for s in range(0, len(paths), batch):
                imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
                px = proc_px(imgs).to(dev, m.dtype)
                o = m.vision_model(pixel_values=px)
                e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
                out.append(e.float().cpu())
            return torch.cat(out)
        return enc, params, None, spec["id"]
    elif t == "tf-dinov2":
        from transformers import AutoModel, AutoImageProcessor
        m = AutoModel.from_pretrained(spec["id"]).to(dev).eval()
        proc = AutoImageProcessor.from_pretrained(spec["id"])
        params = sum(p.numel() for p in m.parameters())

        @torch.no_grad()
        def enc(paths, batch=128):
            out = []
            for s in range(0, len(paths), batch):
                imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
                with autocast():
                    o = m(pixel_values=px)
                e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
                out.append(e.float().cpu())
            return torch.cat(out)
        return enc, params, None, spec["id"]
    raise ValueError(t)


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def pack_bits(codes):
    b = (np.asarray(codes) > 0).astype(np.uint8)
    if b.ndim == 1:
        b = b[None, :]
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="big"), dtype=np.uint8)


def recall(ix, q, gold):
    _, I = ix.search(np.ascontiguousarray(q), max(KS))
    return {k: round(100 * float(np.mean([gold[i] in I[i, :k] for i in range(len(gold))])), 2) for k in KS}


def coco_paths(ids):
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    return [f'{REPO}/data/coco/{dco[c]["filepath"]}/{dco[c]["filename"]}' for c in ids], dco


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enc", required=True); args = ap.parse_args()
    spec = ENCODERS[args.enc]; PAPER.mkdir(parents=True, exist_ok=True)
    ck = torch.load(HEAD_PATH, map_location="cpu")
    BITS = [int(b) for b in ck["bits"]]; bi = BITS.index(1024); hidden, embed = ck["hidden"], ck["embed"]
    P = json.load(open("/tmp/hp_results.json"))["best_params"]

    # frozen anchor head (the exact head that built index.bin)
    img_h = NestedHashLayer(embed, hidden, BITS, 0.0).to(dev).eval(); img_h.load_state_dict(ck["img_h"])
    for p in img_h.parameters():
        p.requires_grad_(False)

    EA = torch.load("/tmp/emb_aug.pt", map_location="cpu")["train"]
    tr_ids = [int(x) for x in EA["ids"].tolist()][:TRAIN_N]
    tr_anchor = F.normalize(EA["clean"].float()[:TRAIN_N], dim=1)            # SigLIP2 train img emb
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")["test"]
    te_ids = [int(x) for x in EC["ids"].tolist()]
    te_anchor = F.normalize(EC["img"].float(), dim=1)                        # SigLIP2 test img emb
    n_te = len(te_ids); gold = list(range(n_te))

    tr_paths, _ = coco_paths(tr_ids); te_paths, _ = coco_paths(te_ids)

    # ---- encode images with E ----
    enc, params, _, label = load_encoder(spec)
    t0 = time.perf_counter()
    E_tr = enc(tr_paths)
    imgps = round(len(tr_paths) / (time.perf_counter() - t0), 1)
    E_te = enc(te_paths)
    Edim = E_tr.shape[1]
    print(f"[img:{args.enc}] {label} params={params/1e6:.1f}M dim={Edim} | encoded {len(tr_paths)}tr "
          f"@ {imgps} img/s + {len(te_paths)}te", flush=True)
    E_tr = F.normalize(E_tr, dim=1); E_te = F.normalize(E_te, dim=1)

    # ---- train img_h'_E (mirror headadapt_train: io = frozen anchor, to = trainable on E emb) ----
    torch.manual_seed(42)
    ih = NestedHashLayer(Edim, hidden, BITS, P["dropout"]).to(dev).train()
    lf = CombinedHashLoss(BITS, contrastive_weight=1.0, ortho_weight=P["ortho"], quantization_weight=P["quant"],
                          balance_weight=P["balance"], consistency_weight=P["cons"], lcs_weight=P["lcs"],
                          temperature=P["temperature"]).to(dev)
    opt = torch.optim.AdamW(ih.parameters(), lr=P["lr"], weight_decay=P["wd"])
    N = E_tr.shape[0]; steps = EPOCHS * (N // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=P["lr"], total_steps=max(steps, 1), pct_start=0.3)
    g = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        for s in range(0, N - BS + 1, BS):
            idx = perm[s:s + BS]
            with torch.no_grad():
                io = img_h(tr_anchor[idx].to(dev))                  # frozen SigLIP2 anchor codes
            to = ih(E_tr[idx].to(dev))                              # trainable img_h'_E on E emb
            loss = lf(io, to, progress=g / max(steps, 1))["total"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); g += 1
    ih.eval()
    torch.save({"img_h": ih.state_dict(), "bits": BITS, "hidden": hidden, "embed": Edim,
                "encoder": label, "mode": f"imgheadadapt_{args.enc}"}, f"/tmp/imgh_{args.enc}.pt")
    print(f"[img:{args.enc}] trained img_h' in {(time.perf_counter()-t0)/60:.1f}min", flush=True)

    # ---- codes ----
    @torch.no_grad()
    def codes(head, emb):
        return head(emb.to(dev))[bi]["binary"].detach().cpu().numpy()
    sig_te_codes = codes(img_h, te_anchor)          # shipped SigLIP2 test gallery codes
    E_te_codes = codes(ih, E_te)                    # E-indexed test gallery codes
    gal_E = faiss_bin(pack_bits(E_te_codes), 1024)
    gal_sig = faiss_bin(pack_bits(sig_te_codes), 1024)

    # ---- text query codes (server SigLIP2-text lowercased + offline e5), via ft113 txt_h / txt_h_e5 ----
    from web.common import Encoder
    enc_ft = Encoder()  # ft113 txt_h + img_h
    qcodes = {}
    en_lc = torch.load("/tmp/coco_en_lc.pt", map_location="cpu").float()      # lowercased SigLIP2 EN test txt
    KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")["txt_emb"].float()
    qcodes[("server", "EN")] = pack_bits(enc_ft._codes_pm1(enc_ft.txt_h, en_lc))
    qcodes[("server", "KO")] = pack_bits(enc_ft._codes_pm1(enc_ft.txt_h, KO))
    # offline e5
    try:
        e5ck = torch.load("/tmp/txt_h_e5.pt", map_location="cpu")
        bits5 = [int(b) for b in e5ck["bits"]]; obi = bits5.index(1024)
        th5 = NestedHashLayer(e5ck["embed"], e5ck["hidden"], bits5, 0.0); th5.load_state_dict(e5ck["txt_h"]); th5.to(dev).eval()
        from transformers import AutoModel as AM, AutoTokenizer
        e5m = AM.from_pretrained(e5ck["student"]).to(dev).eval(); e5t = AutoTokenizer.from_pretrained(e5ck["student"])
        en_caps = [str(c) for c in EC["captions"]]
        import re as _re
        _ID = _re.compile(r"_0*(\d+)\.jpg"); kf = {}
        for line in open(f"{REPO}/data/coco_ko/coco_ko.jsonl"):
            e = json.loads(line); m = _ID.search(e.get("image_path", ""))
            if m:
                kf[int(m.group(1))] = e.get("captions", [])
        ko_caps = [(kf.get(c, [""]) or [""])[0] for c in te_ids]

        @torch.no_grad()
        def e5codes(caps):
            out = []
            pref = e5ck.get("prefix") or ""
            for s in range(0, len(caps), 256):
                tt = e5t([pref + x for x in caps[s:s + 256]], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                o = e5m(tt["input_ids"].to(dev), tt["attention_mask"].to(dev)).last_hidden_state
                msk = tt["attention_mask"].to(dev).unsqueeze(-1).float()
                em = F.normalize((o * msk).sum(1) / msk.sum(1).clamp(min=1e-9), dim=1)
                out.append(th5(em)[obi]["binary"].detach().cpu().numpy())
            return pack_bits(np.concatenate(out))
        qcodes[("offline", "EN")] = e5codes(en_caps)
        qcodes[("offline", "KO")] = e5codes(ko_caps)
        offline_ok = True
    except Exception as ex:
        offline_ok = False
        print(f"[img:{args.enc}] offline-q skipped: {repr(ex)[:80]}", flush=True)

    # ---- (a) recall on E gallery ----
    rE = {p: recall(gal_E, qcodes[p], gold) for p in qcodes}
    # baseline: same queries on the SHIPPED SigLIP2 gallery (reference ceiling)
    rSig = {p: recall(gal_sig, qcodes[p], gold) for p in qcodes}
    # ---- (b) mixed gallery: even idx = SigLIP2 code, odd idx = E code ----
    mix = sig_te_codes.copy(); mix[1::2] = E_te_codes[1::2]
    gal_mix = faiss_bin(pack_bits(mix), 1024)
    mixed_r10 = recall(gal_mix, qcodes[("server", "EN")], gold)[10]
    # ---- (c) code fidelity: E code vs SigLIP2 code (same image) ----
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    ham = float(_LUT[np.bitwise_xor(pack_bits(E_te_codes), pack_bits(sig_te_codes))].sum(1).mean())
    _, IE = gal_E.search(np.ascontiguousarray(qcodes[("server", "EN")]), 10)
    _, IS = gal_sig.search(np.ascontiguousarray(qcodes[("server", "EN")]), 10)
    top10_overlap = round(float(np.mean([len(set(IE[i]) & set(IS[i])) / 10 for i in range(n_te)])), 3)

    int8_mb = round(params / 1e6, 1)  # int8 ~ 1 byte/param
    row = {"encoder": args.enc, "hf_id": label, "params_M": round(params / 1e6, 1), "dim": Edim,
           "int8_MB": int8_mb, "ondevice": spec["ondevice"], "train_img_s": imgps,
           "Rs_EN_R1": rE[("server", "EN")][1], "Rs_EN_R10": rE[("server", "EN")][10],
           "Rs_KO_R10": rE[("server", "KO")][10],
           "Roff_EN_R10": rE[("offline", "EN")][10] if offline_ok else "",
           "Roff_KO_R10": rE[("offline", "KO")][10] if offline_ok else "",
           "mixed_server_EN_R10": mixed_r10, "code_hamming_vs_sig": round(ham, 1),
           "top10_overlap_vs_sig": top10_overlap,
           "ref_sig_server_EN_R10": rSig[("server", "EN")][10], "ref_sig_server_KO_R10": rSig[("server", "KO")][10]}

    cols = list(row.keys())
    out = PAPER / "image_encoders_headadapt.csv"
    rows = [r for r in csv.DictReader(open(out))] if out.exists() else []
    rows = [r for r in rows if r.get("encoder") != args.enc]
    rows.append(row)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"[img:{args.enc}] RESULT {json.dumps(row, ensure_ascii=False)}", flush=True)
    print(f"[img:{args.enc}] vs shipped SigLIP2 ceiling: E server EN R@10 {row['Rs_EN_R10']} "
          f"(sig {row['ref_sig_server_EN_R10']}) | mixed {mixed_r10} | code-overlap {top10_overlap} | "
          f"ham {ham:.1f}/1024", flush=True)
    print(f"[img:{args.enc}] DONE -> paper/image_encoders_headadapt.csv", flush=True)


if __name__ == "__main__":
    main()
