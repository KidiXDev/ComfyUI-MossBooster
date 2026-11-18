"""gen.py <base|lora> <bake 0/1> <int8 0/1> <start_percent>: workflow with MOSS Booster inserted, prints its path."""

import json
import sys
from pathlib import Path

here = Path(__file__).parent
src, bake, int8, start = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", float(sys.argv[4])
wf = json.loads((here / f"wf_{src}.json").read_text())
wf["80"] = {
    "class_type": "MossBooster",
    "inputs": {
        "model": wf["69"]["inputs"]["model"],
        "bake_lora": bake,
        "int8_attention": int8,
        "int8_start_percent": start,
    },
}
wf["69"]["inputs"]["model"] = ["80", 0]
out = here / ".run" / f"wf_{src}_b{int(bake)}_i{int(int8)}_s{start}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(wf, indent=1))
print(out)
