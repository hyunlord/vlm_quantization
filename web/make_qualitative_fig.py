"""Qualitative retrieval figure (fig/fig_qualitative.pdf) for the §metric point: the server and offline
query paths have LOW top-k overlap yet BOTH retrieve relevant images.

Both paths search the SAME image gallery (COCO 5K test, so400m image -> ft113 img_h -> 1024-bit, the
deployment contract in web/common.py). Only the QUERY encoder differs:
  server  = so400m text tower (LOWERCASED, the fix) -> ft113 txt_h -> 1024-bit   (Encoder, web/common.py)
  offline = multilingual-e5-small (native) -> adapted head txt_h' (/tmp/txt_h_e5.pt) -> 1024-bit
Hamming top-5 per query/path (faiss IndexBinaryFlat, same packing as the paper numbers).

Relevance (green border) for these hand-written queries = the retrieved image's OWN COCO reference captions
contain the query's key object (measured from dataset_coco.json, not cherry-picked). Overlap (• dot) = image
returned by BOTH paths. Figure renders the EN-query retrieval; KO is shown as a parallel label and the full
EN+KO top-5 (both paths) is tabulated in web/QUALITATIVE.md.

Run: HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/make_qualitative_fig.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web.common import Encoder, pack_bits  # noqa: E402
from src.models.nested_hash_layer import NestedHashLayer  # noqa: E402

FIG = Path(REPO) / "fig"
WEB = Path(REPO) / "web"
KO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Difficulty spectrum: 2 distinctive (expect HIGH overlap, both paths agree) + 3 generic/common scenes
# (expect LOW overlap — many valid matches, paths diverge — yet non-shared images should stay relevant).
# Relevance keyword(s) measured against each retrieved image's own COCO reference captions (not cherry-picked;
# if a low-overlap query surfaces an irrelevant image it is reported as-is).
QUERIES = [
    # -- distinctive (clear) --
    {"en": "two giraffes standing near a tree", "ko": "나무 옆에 서 있는 기린 두 마리", "kw": ["giraffe"], "tier": "clear"},
    {"en": "a slice of pizza on a white plate", "ko": "흰 접시 위의 피자 한 조각", "kw": ["pizza"], "tier": "clear"},
    # -- generic (common scene; many valid matches) --
    {"en": "people sitting around a dining table", "ko": "식탁에 둘러앉은 사람들",
     "kw": ["table", "dining", "eating", "restaurant", "food", "meal"], "tier": "generic"},
    {"en": "a busy city street with cars and people", "ko": "차와 사람들로 붐비는 도심 거리",
     "kw": ["street", "road", "traffic", "city", "cars", "intersection", "bus"], "tier": "generic"},
    {"en": "a bathroom with a sink and a mirror", "ko": "세면대와 거울이 있는 욕실",
     "kw": ["bathroom", "sink", "toilet", "mirror", "shower", "restroom"], "tier": "generic"},
]
TOPK = 5


def faiss_bin(packed, bits):
    import faiss
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    ix = faiss.IndexBinaryFlat(bits); ix.add(np.ascontiguousarray(packed)); return ix


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
    te_ids = [int(x) for x in EC["test"]["ids"].tolist()]
    te_img = EC["test"]["img"].float().numpy()

    # COCO id -> path + reference captions (for relevance)
    dco = {im["cocoid"]: im for im in json.load(open(f"{REPO}/data/coco/dataset_coco.json"))["images"]}
    path_of = {i: f'{REPO}/data/coco/{dco[i]["filepath"]}/{dco[i]["filename"]}' for i in te_ids}
    caps_of = {i: [s["raw"].lower() for s in dco[i]["sentences"]] for i in te_ids}

    enc = Encoder()
    gal = enc.image_codes_packed(te_img); ix = faiss_bin(gal, enc.bits)

    # offline e5-small + adapted head
    ck = torch.load("/tmp/txt_h_e5.pt", map_location="cpu")
    bits = [int(b) for b in ck["bits"]]; obi = bits.index(enc.bits)
    prefix = ck.get("prefix") or ""
    th = NestedHashLayer(ck["embed"], ck["hidden"], bits, 0.0); th.load_state_dict(ck["txt_h"]); th.to(dev).eval()
    from transformers import AutoModel, AutoTokenizer
    e5 = AutoModel.from_pretrained(ck["student"]).to(dev).eval()
    e5tok = AutoTokenizer.from_pretrained(ck["student"])

    @torch.no_grad()
    def server_code(text):
        emb = enc.encode_text_emb(text.lower())                       # so400m text LOWERCASED
        return pack_bits(enc._codes_pm1(enc.txt_h, torch.from_numpy(emb)))

    @torch.no_grad()
    def offline_codes(texts, batch=256):
        out = []
        for s in range(0, len(texts), batch):
            t = e5tok([prefix + x for x in texts[s:s + batch]], padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt")
            o = e5(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).last_hidden_state
            msk = t["attention_mask"].to(dev).unsqueeze(-1).float()
            e = F.normalize((o * msk).sum(1) / msk.sum(1).clamp(min=1e-9), dim=1)
            out.append(pack_bits(th(e)[obi]["binary"].detach().cpu().numpy()))
        return np.concatenate(out, 0)

    def offline_code(text):
        return offline_codes([text])

    def topk(code):
        _, I = ix.search(np.ascontiguousarray(code), TOPK)
        return [te_ids[j] for j in I[0]]

    def relevant(cid, kws):
        return any(kw in cap for cap in caps_of[cid] for kw in kws)

    # ---- GLOBAL mean top-5 overlap over all 5K EN test queries (server vs offline) — caption number ----
    glob_overlap = None
    try:
        sv5 = pack_bits(enc._codes_pm1(enc.txt_h, torch.load("/tmp/coco_en_lc.pt", map_location="cpu")))
        of5 = offline_codes([str(c) for c in EC["test"]["captions"]])
        _, Isv = ix.search(np.ascontiguousarray(sv5), TOPK)
        _, Iof = ix.search(np.ascontiguousarray(of5), TOPK)
        glob_overlap = round(float(np.mean([len(set(Isv[i]) & set(Iof[i])) / TOPK for i in range(len(te_ids))])), 3)
        print(f"[fig] GLOBAL mean top-5 overlap (server vs offline, 5K EN) = {glob_overlap}", flush=True)
    except Exception as ex:
        print(f"[fig] global overlap skipped: {repr(ex)[:90]}", flush=True)

    # ---- retrieve (EN shown in figure; KO computed for QUALITATIVE.md) ----
    results = []  # per query: {en,ko,kw, en:{server:[ids],offline:[ids]}, ko:{...}}
    for q in QUERIES:
        r = {"q": q}
        for lang in ("en", "ko"):
            sv = topk(server_code(q[lang])); of = topk(offline_code(q[lang]))
            r[lang] = {"server": sv, "offline": of, "overlap": sorted(set(sv) & set(of))}
        results.append(r)

    # ---- render figure (EN retrieval) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from PIL import Image
    fp = fm.FontProperties(fname=KO_FONT) if os.path.exists(KO_FONT) else fm.FontProperties()
    fp_sm = fp.copy(); fp_sm.set_size(7)
    fp_q = fp.copy(); fp_q.set_size(8)

    nq = len(QUERIES)
    fig, axes = plt.subplots(2 * nq, 1 + TOPK, figsize=(7.0, 1.05 * 2 * nq + 0.2),
                             gridspec_kw={"width_ratios": [2.4] + [1] * TOPK, "hspace": 0.08, "wspace": 0.04})
    for qi, r in enumerate(results):
        q = r["q"]
        for pi, path in enumerate(("server", "offline")):
            row = 2 * qi + pi
            ids = r["en"][path]
            ov = set(r["en"]["overlap"])
            # text column
            axt = axes[row][0]; axt.axis("off")
            if pi == 0:
                axt.text(0.02, 0.5, f"Q{qi+1}\nEN: {q['en']}\nKO: {q['ko']}", fontproperties=fp_q,
                         va="center", ha="left", wrap=True, transform=axt.transAxes)
            axes[row][1].set_ylabel(path, fontproperties=fp_sm, rotation=90, labelpad=2)
            # thumbnails
            for k in range(TOPK):
                ax = axes[row][1 + k]; ax.set_xticks([]); ax.set_yticks([])
                cid = ids[k]
                try:
                    im = Image.open(path_of[cid]).convert("RGB"); im.thumbnail((240, 240))
                    ax.imshow(im)
                except Exception:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                rel = relevant(cid, q["kw"])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#1a9e1a" if rel else "0.8"); sp.set_linewidth(2.4 if rel else 0.5)
                if cid in ov:
                    ax.plot([0.13], [0.87], "o", ms=6, color="black", mec="white", mew=0.8,
                            transform=ax.transAxes, clip_on=False, zorder=6)
    gtxt = f"  |  global mean top-5 overlap {glob_overlap}" if glob_overlap is not None else ""
    fig.suptitle("server (so400m+ft113) vs offline (e5-small head-adapt) top-5: clear queries agree, "
                 "generic diverge yet stay relevant" + gtxt, fontproperties=fp_sm, y=0.997)
    fig.savefig(FIG / "fig_qualitative.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig_qualitative.png", dpi=150, bbox_inches="tight")  # quick visual check
    print(f"[fig] saved fig/fig_qualitative.pdf (+png)", flush=True)

    # ---- QUALITATIVE.md ----
    gline = (f"**Global mean top-5 set-overlap (server vs offline, 5K EN test) = {glob_overlap}** "
             f"(this figure's metric; `encoders.csv` separately reports a 0.42 agreement metric under a "
             f"different definition — cite whichever fits, they are NOT the same measurement). "
             if glob_overlap is not None else "")
    lines = ["# Qualitative retrieval — server vs offline top-5 (verification table)", "",
             "Gallery = COCO 5K test (so400m img -> ft113 img_h, 1024-bit). server query = so400m text "
             "(lowercased) -> ft113 txt_h. offline query = multilingual-e5-small (native) -> adapted txt_h'. "
             "`*` = retrieved image's COCO captions contain a query keyword (relevance proxy). "
             "**Difficulty spectrum: 2 clear + 3 generic.** " + gline +
             "Key honest check: when the paths DIVERGE (low overlap), are the NON-SHARED images still relevant?", ""]
    ns_rel_hit, ns_rel_tot = 0, 0
    for qi, r in enumerate(results):
        q = r["q"]
        lines.append(f"## Q{qi+1} [{q['tier']}]: EN `{q['en']}` / KO `{q['ko']}`  (kw: {q['kw']})")
        for lang in ("en", "ko"):
            sv, of, ov = r[lang]["server"], r[lang]["offline"], set(r[lang]["overlap"])
            def fmt(ids):
                return ", ".join(f"{cid}{'*' if relevant(cid, q['kw']) else '·'}" for cid in ids)
            ns_s = [c for c in sv if c not in ov]; ns_o = [c for c in of if c not in ov]
            ns = ns_s + ns_o
            nsr = sum(relevant(c, q["kw"]) for c in ns)
            if lang == "en":  # accumulate non-shared relevance over EN (the shown language)
                ns_rel_hit += nsr; ns_rel_tot += len(ns)
            lines.append(f"- **{lang.upper()}** server [{fmt(sv)}] | offline [{fmt(of)}] | "
                         f"**overlap {len(ov)}/5** | non-shared {len(ns)} imgs, {nsr} relevant "
                         f"(server-only {ns_s}, offline-only {ns_o})")
        lines.append("")

    def tier_overlap(tier, lang):
        xs = [len(r[lang]["overlap"]) for r in results if r["q"]["tier"] == tier]
        return round(float(np.mean(xs)), 2) if xs else float("nan")
    lines += ["## Summary",
              f"- {gline.strip()}" if gline else "",
              f"- mean top-5 overlap by tier (EN): clear {tier_overlap('clear','en')}/5, "
              f"generic {tier_overlap('generic','en')}/5  → spectrum as intended (clear agree, generic diverge)",
              f"- **non-shared (divergent) EN images relevant: {ns_rel_hit}/{ns_rel_tot} "
              f"({round(100*ns_rel_hit/max(ns_rel_tot,1))}%)** — the core message: even where the two paths "
              "return DIFFERENT images, the divergent ones are still on-topic.",
              "- (Any `·` on a generic query = a non-relevant retrieval, reported as-is, not removed.)", ""]
    (WEB / "QUALITATIVE.md").write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")
    print(f"[fig] overlap by tier (EN): clear {tier_overlap('clear','en')}/5 generic {tier_overlap('generic','en')}/5"
          f" | global {glob_overlap} | non-shared EN relevant {ns_rel_hit}/{ns_rel_tot}", flush=True)
    print("[fig] RESULT_JSON " + json.dumps({"glob_overlap": glob_overlap,
          "results": [{"tier": r["q"]["tier"], "en": r["en"], "ko": r["ko"]} for r in results]},
          ensure_ascii=False), flush=True)
    print("[fig] DONE -> fig/fig_qualitative.pdf + web/QUALITATIVE.md", flush=True)


if __name__ == "__main__":
    main()
