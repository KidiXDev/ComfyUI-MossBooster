#!/usr/bin/env bash
# usage: COMFYUI_DIR=/path/to/ComfyUI start.sh <logname> [extra comfy args...]
D=$(cd "$(dirname "$0")" && pwd); S=$D/.run
"$D/stop.sh"; mkdir -p "$S"; L=$S/$1.log; shift
nohup "$D/server.sh" "$L" "$@" >/dev/null 2>&1 &
for _ in $(seq 90); do curl -s 127.0.0.1:8190/system_stats >/dev/null && exit 0; sleep 1; done
tail -20 "$L"; exit 1
