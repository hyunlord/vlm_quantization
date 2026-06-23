#!/bin/bash
# Deep-lever gate sweep driver (run on DGX). Head-only, cached embeddings.
# Writes all rows to /tmp/lever_all.csv (split into lever_A/B locally by head_norm/loss).
# Levers A (negsep training loss) + B (head arch) + composition-response arm, seeds {0,1}.
set -u
cd /home/hyunlord/github/vlm_quantization
PY=.venv/bin/python
export EPOCHS=25 BITS=64,128,256,512,1024 CSV=/tmp/lever_all.csv
run(){ echo "### $*"; env "$@" $PY /tmp/lever_sweep.py 2>&1 \
  | grep -v "UserWarning\|queued_call\|capability\|Minimum and Maximum\|^    "; }

for SEED in 0 1; do
  # shared baseline (BN + full recipe == current head)
  run HEAD_NORM=bn   LOSS=infonce          SEED=$SEED TAG=bn-infonce-s$SEED
  # ---- Lever A: negative-separation training losses ----
  run HEAD_NORM=bn   LOSS=hardneg          SEED=$SEED TAG=bn-hardneg-s$SEED
  run HEAD_NORM=bn   LOSS=hmargin          SEED=$SEED TAG=bn-hmargin-s$SEED
  run HEAD_NORM=bn   LOSS=infonce_hardneg  SEED=$SEED TAG=bn-inceHardneg-s$SEED
  run HEAD_NORM=bn   LOSS=infonce_hmargin  SEED=$SEED TAG=bn-inceHmargin-s$SEED
  # ---- Lever B: head architecture (swap per-bit BatchNorm) ----
  run HEAD_NORM=ln         LOSS=infonce    SEED=$SEED TAG=ln-infonce-s$SEED
  run HEAD_NORM=none       LOSS=infonce    SEED=$SEED TAG=none-infonce-s$SEED
  run HEAD_NORM=none_scale LOSS=infonce    SEED=$SEED TAG=nonescale-infonce-s$SEED
  run HEAD_NORM=rotation   LOSS=infonce    SEED=$SEED TAG=rotation-infonce-s$SEED
  # ---- Lever B-d: BN-free + negsep (interaction) ----
  run HEAD_NORM=none LOSS=infonce_hmargin  SEED=$SEED TAG=none-inceHmargin-s$SEED
  # ---- composition-response arm: does aux weight move R@10 under BN-free vs BN? ----
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=0 SEED=$SEED TAG=none-aux0-s$SEED
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=none-aux4-s$SEED
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=0 SEED=$SEED TAG=bn-aux0-s$SEED
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=bn-aux4-s$SEED
done
echo "ALL_LEVER_DONE"
