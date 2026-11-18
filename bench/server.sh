#!/usr/bin/env bash
# usage: COMFYUI_DIR=/path/to/ComfyUI server.sh <logfile> [extra comfy args...]
C=${COMFYUI_DIR:?set COMFYUI_DIR to your ComfyUI folder}
PY=${COMFYUI_PYTHON:-$C/.venv/bin/python}
S=$(cd "$(dirname "$0")" && pwd)/.run
LOG=$1; shift
mkdir -p "$S/user" "$S/out"
echo $$ > "$S/server.pid"
cd "$C" && exec "$PY" main.py --port 8190 --listen 127.0.0.1 --disable-auto-launch \
  --user-directory "$S/user" --output-directory "$S/out" "$@" > "$LOG" 2>&1
