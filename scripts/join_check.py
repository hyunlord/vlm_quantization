import gzip, json, glob, tarfile, os
cf = glob.glob("/tmp/cc12m_caps/**/*.jsonl.gz", recursive=True)[0]
with gzip.open(cf, "rt", encoding="utf-8") as fh:
    d0 = json.loads(next(fh))
print("jsonl keys:", list(d0.keys()), flush=True)
print("sample key:", repr(d0.get("key")), flush=True)
print("sample caption_llava:", str(d0.get("caption_llava"))[:140], flush=True)
keys = set()
with gzip.open(cf, "rt", encoding="utf-8") as fh:
    for line in fh:
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = d.get("key")
        if k is not None:
            keys.add(str(k))
print("total caption keys:", len(keys), flush=True)
t = sorted(glob.glob("/tmp/cc12m/*.tar"))[0]
m = tot = 0
with tarfile.open(t) as tar:
    for mm in tar:
        if mm.name.endswith(".jpg"):
            tot += 1
            if os.path.splitext(os.path.basename(mm.name))[0] in keys:
                m += 1
print(f"shard0: {m}/{tot} matched ({m/max(tot,1)*100:.0f}%)", flush=True)
print("JOIN_CHECK_DONE", flush=True)
