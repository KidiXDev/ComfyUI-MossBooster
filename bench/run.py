"""Submit an API-format workflow: one warm-up run (seed+1), then N timed runs at the workflow seed."""

import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("workflow")
ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--prefix", default="bench")
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--port", type=int, default=8190)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"


def post(wf):
    req = urllib.request.Request(
        URL + "/prompt", json.dumps({"prompt": wf}).encode(), {"Content-Type": "application/json"}
    )
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t = time.time()
    while True:
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                sys.exit("FAILED: " + json.dumps(st)[:3000])
            return time.time() - t, h[pid]
        time.sleep(0.2)


wf = json.load(open(a.workflow))
seed = wf["70"]["inputs"]["seed"]
if not a.no_warmup:
    wf["70"]["inputs"]["seed"] = seed + 1
    wf["9"]["inputs"]["filename_prefix"] = a.prefix + "_warm"
    dt, _ = post(wf)
    print(f"warmup {dt:.2f}s", flush=True)
for i in range(a.runs):
    wf["70"]["inputs"]["seed"] = seed if i % 2 == 0 else seed + 7
    wf["9"]["inputs"]["filename_prefix"] = f"{a.prefix}_{i}"
    dt, h = post(wf)
    imgs = [o["filename"] for n in h["outputs"].values() for o in n.get("images", [])]
    print(f"run{i} {dt:.2f}s {imgs}", flush=True)
