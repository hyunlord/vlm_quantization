#!/usr/bin/env bash
# 원클릭 viz 서버. 사용: bash serve.sh   (또는: python3 serve.py)
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "❌ python3 가 필요합니다."; exit 1; fi
exec "$PY" serve.py "$@"
