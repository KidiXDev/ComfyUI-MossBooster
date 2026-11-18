"""side.py out.jpg a.png b.png [...] -> horizontal strip, downscaled"""

import sys

from PIL import Image

ims = [Image.open(p).convert("RGB") for p in sys.argv[2:]]
h = 640
ims = [i.resize((i.width * h // i.height, h), Image.LANCZOS) for i in ims]
out = Image.new("RGB", (sum(i.width for i in ims), h))
x = 0
for i in ims:
    out.paste(i, (x, 0))
    x += i.width
out.save(sys.argv[1], quality=90)
