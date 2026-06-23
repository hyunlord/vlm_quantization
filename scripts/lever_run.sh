#!/bin/bash
# Deep-lever gate sweep — PASS 1 (primary gates). Head-only, cached embeddings.
# EPOCHS=15 (3-epoch poke already ~98% of ceiling => fast convergence; baseline and
# variants share the count so the controlled Delta is fair). One bn-infonce@25
# validation run guards against under-training the baseline. Writes /tmp/lever_all.csv.
# Aux-scale composition-response arm is PASS 2 (lever_run2.sh), run only if useful.
set -u
cd /home/hyunlord/github/vlm_quantization
PY=.venv/bin/python
export BITS=64,128,256,512,1024 CSV=/tmp/lever_all.csv
run(){ echo "### $*"; env EPOCHS=15 "$@" $PY /tmp/lever_sweep.py 2>&1 \
  | grep -v "UserWarning\|queued_call\|capability\|Minimum and Maximum\|^    "; }

# fairness validation: baseline at 25 epochs (seed 0) — compare to bn-infonce@15
echo "### VALIDATION bn-infonce@25 s0"
env EPOCHS=25 HEAD_NORM=bn LOSS=infonce SEED=0 TAG=bn-infonce-e25-s0 CSV=/tmp/lever_all.csv \
  $PY /tmp/lever_sweep.py 2>&1 | grep -v "UserWarning\|queued_call\|capability\|Minimum and Maximum\|^    "

for SEED in 0 1; do
  # ---- decisive + light first (early checkpoint => Lever A read) ----
  run HEAD_NORM=bn   LOSS=infonce              SEED=$SEED TAG=bn-infonce-s$SEED        # baseline (full recipe)
  run HEAD_NORM=bn   LOSS=infonce AUX_SCALE=0  SEED=$SEED TAG=bn-aux0-s$SEED           # pure InfoNCE (contrastive-only)
  run HEAD_NORM=bn   LOSS=hmargin HMARGIN_M=0.1  SEED=$SEED TAG=bn-hmargin-m10-s$SEED  # pure hmargin (star)
  run HEAD_NORM=bn   LOSS=hmargin HMARGIN_M=0.05 SEED=$SEED TAG=bn-hmargin-m05-s$SEED
  run HEAD_NORM=bn   LOSS=hmargin HMARGIN_M=0.2  SEED=$SEED TAG=bn-hmargin-m20-s$SEED
  run HEAD_NORM=bn   LOSS=hardneg              SEED=$SEED TAG=bn-hardneg-s$SEED        # control
  run HEAD_NORM=bn   LOSS=infonce_hmargin NEGSEP_W=10 HMARGIN_M=0.1 SEED=$SEED TAG=bn-inceHmargin-w10-s$SEED
  run HEAD_NORM=bn   LOSS=infonce_hardneg NEGSEP_W=1.0 SEED=$SEED TAG=bn-inceHardneg-w1-s$SEED
  # ---- Lever B: head architecture ----
  run HEAD_NORM=ln         LOSS=infonce       SEED=$SEED TAG=ln-infonce-s$SEED
  run HEAD_NORM=none       LOSS=infonce       SEED=$SEED TAG=none-infonce-s$SEED
  run HEAD_NORM=none_scale LOSS=infonce       SEED=$SEED TAG=nonescale-infonce-s$SEED
  run HEAD_NORM=rotation   LOSS=infonce       SEED=$SEED TAG=rotation-infonce-s$SEED
  # ---- Lever B-d: BN-free + negsep (interaction) ----
  run HEAD_NORM=none LOSS=infonce_hmargin NEGSEP_W=10 HMARGIN_M=0.1 SEED=$SEED TAG=none-inceHmargin-w10-s$SEED
done
echo "ALL_LEVER_DONE"
