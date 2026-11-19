"""Smoke test: python tests/smoke_test.py  (after train.py)"""
import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_digits
from common import ImageRejected, load_checkpoint, predict, prepare_image, patchify
import torch

def png(img, fmt="PNG"):
    b = io.BytesIO(); img.save(b, fmt); return b.getvalue()

cnn, _ = load_checkpoint("models/cnn.pt"); vit, _ = load_checkpoint("models/vit.pt")
x = torch.zeros(2, 1, 8, 8)
assert patchify(x).shape == (2, 16, 4) and cnn(x).shape == (2, 10) and vit(x).shape == (2, 10)

# blank / broken / missing must be rejected
for name, src in {"blank white": png(Image.new("RGB", (200, 200), "white")),
                  "blank black": png(Image.new("RGB", (200, 200), "black")),
                  "garbage bytes": b"not an image", "empty": b"", "none": None}.items():
    try:
        prepare_image(src); raise SystemExit(f"FAIL: {name} accepted")
    except ImageRejected as e:
        print(f"ok reject {name}: {e}")

# render real digits big (dark on white, JPEG) and light on dark; see predictions
d = load_digits()
hits = {"cnn": 0, "vit": 0}; n = 0
for i in range(0, 200, 4):
    a = (d.images[i] / 16 * 255).astype("uint8")
    big = Image.fromarray(255 - a).resize((200, 200), Image.BICUBIC).convert("RGB")
    arr = prepare_image(png(big, "JPEG"))
    for k, m in (("cnn", cnn), ("vit", vit)):
        hits[k] += predict(m, arr)[0] == d.target[i]
    n += 1
print("upscaled-digit JPEG round trip acc:", {k: v / n for k, v in hits.items()})

# font-rendered digits like a user drawing
font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
    if os.path.exists(p): font = ImageFont.truetype(p, 160)
if font:
    ok = {"cnn": 0, "vit": 0}
    for dig in range(10):
        im = Image.new("RGB", (280, 280), "black"); ImageDraw.Draw(im).text((70, 40), str(dig), fill="white", font=font)
        arr = prepare_image(im)
        for k, m in (("cnn", cnn), ("vit", vit)):
            ok[k] += predict(m, arr)[0] == dig
    print("font-rendered digits (out-of-distribution) correct /10:", ok)
print("SMOKE TEST DONE")
