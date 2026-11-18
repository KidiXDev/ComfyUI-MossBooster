#!/usr/bin/env bash
P=$(cd "$(dirname "$0")" && pwd)/.run/server.pid
[ -f "$P" ] || exit 0
kill "$(cat "$P")" 2>/dev/null; while kill -0 "$(cat "$P")" 2>/dev/null; do sleep 0.5; done; rm -f "$P"
