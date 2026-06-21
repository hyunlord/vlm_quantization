#!/bin/bash
# Tier-1 core image head-adapt sweep (serialized on the single GB10). Each --enc run is idempotent
# (replaces its CSV row). Not committed — staging/run helper.
cd ~/github/vlm_quantization || exit 1
for e in siglip2-base dinov2-small dinov2-base mobileclip2-s0 mobileclip2-s2; do
  echo "##### START $e $(date +%H:%M:%S) #####"
  HEAD_PATH=/tmp/ft_ko_113.pt TRAIN_N=30000 EPOCHS=25 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    .venv/bin/python web/paper_image_headadapt.py --enc "$e" 2>&1
  echo "##### END $e rc=$? $(date +%H:%M:%S) #####"
done
echo "ALLDONE $(date +%H:%M:%S)"
