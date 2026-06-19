#!/usr/bin/env bash
# Wait for cc12m image+caption downloads, verify the key-join works (abort if not,
# so we never waste an 8h embed on a broken join), then run the full embed.
set -u
cd "$HOME/github/vlm_quantization" || exit 1

echo "[cc12m-chain] waiting for downloads..."
while pgrep -f "pixparse/cc12m" >/dev/null || pgrep -f "conceptual-captions-cc12m" >/dev/null; do sleep 30; done
echo "[cc12m-chain] downloads finished $(date +%H:%M)"
echo "[cc12m-chain] tars: $(ls /tmp/cc12m/*.tar 2>/dev/null | wc -l) | caps: $(du -sh /tmp/cc12m_caps 2>/dev/null | cut -f1)"

CHK=$(.venv/bin/python - <<'PY'
import gzip, json, glob, tarfile, os
keys = set()
for cf in glob.glob("/tmp/cc12m_caps/**/*.jsonl.gz", recursive=True):
    with gzip.open(cf, "rt", encoding="utf-8") as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            k = d.get("key", d.get("__key__"))
            if k is not None: keys.add(str(k))
    break
m = 0
for t in sorted(glob.glob("/tmp/cc12m/*.tar"))[:2]:
    with tarfile.open(t) as tar:
        for mm in tar:
            if mm.name.lower().endswith(".jpg") and os.path.splitext(os.path.basename(mm.name))[0] in keys:
                m += 1
print(f"{len(keys)}|{m}")
PY
)
NKEYS="${CHK%%|*}"; MATCH="${CHK##*|}"
echo "[cc12m-chain] caption keys=$NKEYS | matches in first 2 shards=$MATCH"
if [ "${MATCH:-0}" -lt 100 ]; then
  echo "CC12M_JOIN_FAIL keys=$NKEYS match=$MATCH — aborting (check key format)"; exit 1
fi

echo "[cc12m-chain] join OK; starting full embed $(date +%H:%M)"
LIMIT="${LIMIT:-400000}" BATCH=192 WORKERS=16 TARS=/tmp/cc12m CAPS=/tmp/cc12m_caps \
  OUT=/tmp/cc12m_pairs.pt CAPCOL=caption_llava PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python /tmp/embed_cc12m.py
echo "CC12M_EMBED_DONE $(date +%H:%M)"
