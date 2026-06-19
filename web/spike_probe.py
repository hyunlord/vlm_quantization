"""v2 feasibility spike (cycle 3) — measure the make-or-break axis (SIZE) and dump the
tokenizer reference for the JS parity check. Read-only: NO ONNX export, NO UI, no model
change. Answers "can the SigLIP2 text tower ship to a browser?" cheaply.

Prints the decisive number — SigLIP2-so400m text-tower param count → fp32/fp16/int8
download-size estimates — plus the tiny `txt_h` head size, and writes /tmp/tok_ref.json
(Gemma token ids for KO+EN samples, max_length=64 like the demo) for spike_tok_check.mjs.

Size estimates are derived from exact parameter counts (weights dominate; ONNX graph
overhead is a few %). A real multi-GB ONNX export was deliberately NOT performed because
the size axis already returns a hard no-go for shipping the full tower (see web/V2_SPIKE.md).

Run on DGX:
  .venv/bin/python web/spike_probe.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parent.parent))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MODEL = "google/siglip2-so400m-patch14-384"
HEAD_PATH = os.environ.get("HEAD_PATH", "/tmp/ft_ko_113.pt")
SAMPLES = ["바닷가 강아지", "눈 덮인 산", "a dog on the beach", "two giraffes"]


def _mb(n_params: int, bytes_per: int) -> float:
    return n_params * bytes_per / 1e6


def main() -> None:
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
    npar = lambda mod: sum(p.numel() for p in mod.parameters())
    total, txt = npar(m), npar(m.text_model)
    vis = npar(m.vision_model) if hasattr(m, "vision_model") else 0
    print(f"[spike] {MODEL}")
    print(f"[spike] params: total={total/1e6:.1f}M  text_model={txt/1e6:.1f}M  "
          f"vision={vis/1e6:.1f}M", flush=True)
    print("[spike] text-tower download-size estimate (weights only, ~ONNX size):")
    for prec, b in [("fp32", 4), ("fp16", 2), ("int8", 1)]:
        print(f"    {prec:>4}: ~{_mb(txt, b):,.0f} MB", flush=True)

    if os.path.exists(HEAD_PATH):
        h = torch.load(HEAD_PATH, map_location="cpu")
        hp = sum(v.numel() for v in h["txt_h"].values())
        print(f"[spike] txt_h head: {hp/1e6:.3f}M params (~{_mb(hp,4):.1f} MB fp32) — "
              f"trivial to fold into the graph", flush=True)

    # tokenizer reference (demo uses the Gemma tokenizer via common.py; load it directly)
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    except Exception:
        from transformers import GemmaTokenizer
        tok = GemmaTokenizer.from_pretrained(MODEL)
    print(f"[spike] tokenizer: {type(tok).__name__} is_fast={tok.is_fast} "
          f"vocab={tok.vocab_size} pad={tok.pad_token_id}", flush=True)
    ref = {}
    for s in SAMPLES:
        ids = tok([s], padding="max_length", max_length=64, truncation=True)["input_ids"][0]
        ref[s] = [int(i) for i in ids]
    out = "/tmp/tok_ref.json"
    json.dump(ref, open(out, "w"), ensure_ascii=False)
    print(f"[spike] wrote {out} (for web/spike_tok_check.mjs)", flush=True)
    print("[spike] VERDICT: text tower is 700M+ params -> >=708 MB int8; browser-prohibitive. "
          "See web/V2_SPIKE.md.", flush=True)


if __name__ == "__main__":
    main()
