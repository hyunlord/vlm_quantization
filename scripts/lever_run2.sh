#!/bin/bash
# Deep-lever PASS 2 — composition-response arm for Lever B's second gate:
# does aux-loss weight move R@10 once per-bit BN is removed? (vs prior inert <=0.2pt)
# Run only if pass-1 makes it useful. bn-aux0 + none-infonce(=none-aux1) already in pass 1.
set -u
cd /home/hyunlord/github/vlm_quantization
PY=.venv/bin/python
export EPOCHS=15 BITS=64,128,256,512,1024 CSV=/tmp/lever_all2.csv
run(){ echo "### $*"; env "$@" $PY /tmp/lever_sweep.py 2>&1 \
  | grep -v "UserWarning\|queued_call\|capability\|Minimum and Maximum\|^    "; }
for SEED in 0 1; do
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=bn-aux4-s$SEED
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=0 SEED=$SEED TAG=none-aux0-s$SEED
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=none-aux4-s$SEED
done
echo "ALL_LEVER2_DONE"
