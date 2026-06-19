"""Float (un-quantized cosine) T2I ceiling — EN vs KO on the same 5K COCO test images.
Head-independent: raw SigLIP2 1152-dim cosine retrieval = upper bound any hash head
can approach. Run on CPU (CUDA_VISIBLE_DEVICES=) so it never contends with GPU jobs.
"""
import torch, torch.nn.functional as F

EC = torch.load("/tmp/emb_cache.pt", map_location="cpu")
KO = torch.load("/tmp/coco_ko_test.pt", map_location="cpu")
img = F.normalize(EC["test"]["img"].float(), dim=1)
en = F.normalize(EC["test"]["txt"].float(), dim=1)
ko = F.normalize(KO["txt_emb"].float(), dim=1)
ids = torch.tensor([int(x) for x in EC["test"]["ids"].tolist()])


def t2i(t):
    s = t @ img.t()
    order = s.argsort(1, descending=True)
    gold = ids[order] == ids[:, None]
    rank = gold.float().argmax(1) + 1
    r = {f"R@{k}": round((gold[:, :k].any(1)).float().mean().item() * 100, 2) for k in (1, 5, 10)}
    r["MRR"] = round((1.0 / rank.float()).mean().item(), 4)
    r["MedR"] = int(rank.median().item())
    return r


print(f"test images {img.shape[0]:,} | EN caps {en.shape[0]:,} | KO caps {ko.shape[0]:,}")
print("FLOAT ceiling (1152-dim cosine, T2I):")
print("  EN", t2i(en))
print("  KO", t2i(ko))
