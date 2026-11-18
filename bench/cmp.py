"""PSNR / SSIM of images against a reference: cmp.py ref.png a.png [b.png ...]"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]


def ssim(a, b):
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    w = g[:, None] * g[None]
    w = (w / w.sum()).expand(3, 1, 11, 11)

    def f(x):
        return F.conv2d(x, w, groups=3)

    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma**2, f(b * b) - mb**2, f(a * b) - ma * mb
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))).mean().item()


r = load(sys.argv[1])
for p in sys.argv[2:]:
    x = load(p)
    mse = ((x - r) ** 2).mean().item()
    print(f"{p.split('/')[-1]:40s} PSNR {10 * np.log10(1 / max(mse, 1e-12)):6.2f} dB  SSIM {ssim(x, r):.4f}")
