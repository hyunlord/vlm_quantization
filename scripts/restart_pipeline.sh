#!/usr/bin/env bash
# Restart the 167K embed + train download with reduced concurrency so they
# coexist without the 160-thread contention that killed the first embed.
# Spawns both detached and exits cleanly (run via: ssh host 'bash this.sh').
set -u
cd "$HOME/github/vlm_quantization" || exit 1

pkill -f "[o]i_download_s3" 2>/dev/null || true
pkill -f "[e]mbed_corpus"   2>/dev/null || true
sleep 2

# train download, throttled to 16 workers (network-bound, resume-safe)
IDS=data/image_only/train_ids.txt OUT=data/image_only/oi/train WORKERS=16 \
  setsid nohup .venv/bin/python /tmp/oi_download_s3.py \
  >/tmp/oi_train_dl.log 2>&1 </dev/null &
disown 2>/dev/null || true

# 167K embed, 16 dataloader workers, fresh log (success-only flag)
WORKERS=16 BATCH=256 \
  setsid nohup bash /tmp/embed_167k_run.sh \
  >/tmp/embed_167k.log 2>&1 </dev/null &
disown 2>/dev/null || true

sleep 3
echo "restarted: dl=$(pgrep -f '[o]i_download_s3' >/dev/null && echo ALIVE || echo DEAD) embed=$(pgrep -f '[e]mbed_corpus' >/dev/null && echo ALIVE || echo DEAD)"
exit 0
