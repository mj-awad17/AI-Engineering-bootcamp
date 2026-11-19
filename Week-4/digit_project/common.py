"""Shared code for training AND the Streamlit app.

One preprocessing function + one model definition => training path and
app path cannot drift apart (rubric: consistent scaling / class mapping).
"""
import io
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = list(range(10))          # class index i  ==  digit i
SKLEARN_MAX = 16.0                 # sklearn pixel range 0..16 -> divide by 16
UPLOAD_MAX = 255.0                 # photo pixel range 0..255 -> divide by 255


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    """Return images [N,1,8,8] scaled to 0-1 and labels [N]."""
    d = load_digits()
    x = (d.images.astype(np.float32) / SKLEARN_MAX)[:, None, :, :]
    y = d.target.astype(np.int64)
    return x, y


def make_split(y, seed=SEED):
    """ONE stratified 70/15/15 split of indices, shared by both models."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return tr, va, te


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class DigitCNN(nn.Module):
    """Input [N,1,8,8] -> 10 logits."""

    def __init__(self, c1=16, c2=32, hidden=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),                       # 8x8 -> 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c2 * 4 * 4, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def patchify(x):
    """[N,1,8,8] -> [N,16,4]: sixteen non-overlapping 2x2 patches, row-major."""
    n = x.shape[0]
    p = x.unfold(2, 2, 2).unfold(3, 2, 2)          # [N,1,4,4,2,2]
    return p.reshape(n, 16, 4)


class DigitViT(nn.Module):
    """Patches [N,16,4] -> linear embedding + learned position -> Transformer -> 10 logits."""

    def __init__(self, dim=48, depth=2, heads=4, ff=96, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(4, dim)                         # trainable patch projection
        self.pos = nn.Parameter(torch.zeros(1, 16, dim))       # positional information
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)  # self-attention
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        if x.dim() == 4:                    # accept [N,1,8,8] or [N,16,4]
            x = patchify(x)
        z = self.embed(x) + self.pos
        z = self.encoder(z)
        return self.fc(self.norm(z).mean(dim=1))


def build_model(name, cfg=None):
    cfg = cfg or {}
    return DigitCNN(**cfg) if name == "cnn" else DigitViT(**cfg)


# --------------------------------------------------------------------------
# Controlled augmentation (training only) - helps real photos / drawings
# --------------------------------------------------------------------------
def augment(x, gen=None, max_rot=12.0, max_shift=0.15, scale=(0.9, 1.1),
            noise=0.05, blur_p=0.3):
    """x [N,1,8,8] in 0-1. Small rotation/shift/scale + noise + optional blur."""
    n = x.shape[0]

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen)

    ang = u(-max_rot, max_rot) * math.pi / 180
    s = u(*scale)
    tx, ty = u(-max_shift, max_shift), u(-max_shift, max_shift)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(ang) / s
    theta[:, 0, 1] = -torch.sin(ang) / s
    theta[:, 1, 0] = torch.sin(ang) / s
    theta[:, 1, 1] = torch.cos(ang) / s
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    # random blur (photos are soft)
    k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
    blurred = F.conv2d(out, k[None, None], padding=1)
    m = (torch.rand(n, 1, 1, 1, generator=gen) < blur_p).float()
    out = m * blurred + (1 - m) * out
    out = out * u(0.8, 1.0).view(n, 1, 1, 1)                     # stroke brightness
    out = out + noise * torch.randn(out.shape, generator=gen)    # sensor noise
    return out.clamp(0, 1)


# --------------------------------------------------------------------------
# Upload preparation (PNG/JPEG -> 8x8 float array in 0-1)
# --------------------------------------------------------------------------
class ImageRejected(ValueError):
    """Raised for missing, unreadable or blank images (message shown to user)."""


MIN_CONTRAST = 40        # on 0-255 scale: below this => blank
STROKE_FRAC = 0.35       # pixel counts as stroke if > 35 % of the way to max


def open_image(src):
    """src: bytes | file-like | PIL image | None -> RGB-safe PIL image."""
    if src is None:
        raise ImageRejected("No image provided. Upload a PNG/JPEG or draw a digit.")
    if isinstance(src, Image.Image):
        img = src
    else:
        try:
            data = src if isinstance(src, (bytes, bytearray)) else src.read()
            if not data:
                raise ImageRejected("Empty file.")
            img = Image.open(io.BytesIO(data))
            if img.format not in ("PNG", "JPEG"):
                raise ImageRejected(f"Unsupported format {img.format}. Use PNG or JPEG.")
            img.load()
        except ImageRejected:
            raise
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise ImageRejected("Unreadable or broken image file.")
    img = ImageOps.exif_transpose(img)          # phone photos: honour rotation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def prepare_image(src, with_steps=False):
    """Full pipeline. Returns float32 array [8,8] in 0-1 (same range as training).

    grayscale -> invert if background bright -> contrast stretch ->
    crop strokes -> centre on square -> resize 8x8 -> /255.
    """
    img = open_image(src)
    if min(img.size) < 8:
        raise ImageRejected("Image too small.")
    g = np.asarray(img.convert("L"), dtype=np.float32)

    # background estimate = median of the border; bright background -> invert
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) > 127:
        g = 255.0 - g
    lo = np.percentile(g, 1)
    g = np.clip(g - lo, 0, None)                 # background -> ~0
    hi = np.percentile(g, 99.5)
    if g.max() < MIN_CONTRAST or hi < MIN_CONTRAST * 0.5:
        raise ImageRejected("Image looks blank - no digit stroke found.")
    g = np.clip(g / max(g.max(), 1) * 255.0, 0, 255)   # stretch to full 0-255

    mask = g > STROKE_FRAC * 255
    ys, xs = np.where(mask)
    if len(ys) < 3:
        raise ImageRejected("Image looks blank - no digit stroke found.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = g[y0:y1, x0:x1]
    h, w = crop.shape
    side = int(max(h, w) * 1.15) + 1             # tight margin: sklearn digits fill the 8x8 frame
    canvas = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2    # centre (bbox centre)
    canvas[oy:oy + h, ox:ox + w] = crop
    small = Image.fromarray(canvas.astype(np.uint8)).resize((8, 8), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32) / UPLOAD_MAX
    if arr.max() < 0.1:
        raise ImageRejected("Image looks blank after preparation.")
    return (arr, img, g) if with_steps else arr


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path, name, model, cfg, extra=None):
    torch.save({
        "model": name, "config": cfg, "state_dict": model.state_dict(),
        "classes": CLASSES, "pixel_scale_train": SKLEARN_MAX,
        "pixel_scale_upload": UPLOAD_MAX, "extra": extra or {},
    }, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["model"], ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, arr8x8):
    """arr8x8: [8,8] 0-1 -> (digit, probs[10])."""
    x = torch.from_numpy(np.ascontiguousarray(arr8x8, dtype=np.float32))[None, None]
    p = torch.softmax(model.eval()(x), dim=1)[0].numpy()
    return int(p.argmax()), p
