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
  # pure negsep (controls + characterization: smoke showed pure hmargin is strong)
  run HEAD_NORM=bn   LOSS=hardneg          SEED=$SEED TAG=bn-hardneg-s$SEED
  run HEAD_NORM=bn LOSS=hmargin HMARGIN_M=0.05 SEED=$SEED TAG=bn-hmargin-m05-s$SEED
  run HEAD_NORM=bn LOSS=hmargin HMARGIN_M=0.1  SEED=$SEED TAG=bn-hmargin-m10-s$SEED
  run HEAD_NORM=bn LOSS=hmargin HMARGIN_M=0.2  SEED=$SEED TAG=bn-hmargin-m20-s$SEED
  # InfoNCE + negsep, weights chosen so the negsep term is on a COMPARABLE
  # scale to InfoNCE (hmargin~0.05 -> w 2..10; hardneg~CE -> w 0.3..1) — fair gate
  run HEAD_NORM=bn LOSS=infonce_hmargin NEGSEP_W=2  HMARGIN_M=0.1  SEED=$SEED TAG=bn-inceHmargin-w2-s$SEED
  run HEAD_NORM=bn LOSS=infonce_hmargin NEGSEP_W=10 HMARGIN_M=0.1  SEED=$SEED TAG=bn-inceHmargin-w10-s$SEED
  run HEAD_NORM=bn LOSS=infonce_hmargin NEGSEP_W=10 HMARGIN_M=0.05 SEED=$SEED TAG=bn-inceHmargin-w10m05-s$SEED
  run HEAD_NORM=bn LOSS=infonce_hardneg NEGSEP_W=0.3 SEED=$SEED TAG=bn-inceHardneg-w03-s$SEED
  run HEAD_NORM=bn LOSS=infonce_hardneg NEGSEP_W=1.0 SEED=$SEED TAG=bn-inceHardneg-w1-s$SEED
  # ---- Lever B: head architecture (swap per-bit BatchNorm) ----
  run HEAD_NORM=ln         LOSS=infonce    SEED=$SEED TAG=ln-infonce-s$SEED
  run HEAD_NORM=none       LOSS=infonce    SEED=$SEED TAG=none-infonce-s$SEED
  run HEAD_NORM=none_scale LOSS=infonce    SEED=$SEED TAG=nonescale-infonce-s$SEED
  run HEAD_NORM=rotation   LOSS=infonce    SEED=$SEED TAG=rotation-infonce-s$SEED
  # ---- Lever B-d: BN-free + negsep at meaningful strength (interaction) ----
  run HEAD_NORM=none LOSS=infonce_hmargin NEGSEP_W=10 HMARGIN_M=0.1 SEED=$SEED TAG=none-inceHmargin-w10-s$SEED
  # ---- composition-response arm: does aux weight move R@10 under BN-free vs BN? ----
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=0 SEED=$SEED TAG=none-aux0-s$SEED
  run HEAD_NORM=none LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=none-aux4-s$SEED
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=0 SEED=$SEED TAG=bn-aux0-s$SEED
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=4 SEED=$SEED TAG=bn-aux4-s$SEED
done
echo "ALL_LEVER_DONE"
