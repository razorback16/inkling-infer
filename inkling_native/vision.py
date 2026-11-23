# Original Inkling model implementation copyright © 2026 Apple Inc.
"""Inkling image encoder, adapted from mlx-lm 0c4c991 (native/MLX-LICENSE.txt)."""
import base64
import hashlib
import io
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .metal import bf16
from .model import softmax

IMAGE_TOKEN = 200054


def read_image(url):
    if url.startswith("data:"):
        data = base64.b64decode(url.split(",", 1)[1], validate=True)
    elif url.startswith(("http://", "https://")):
        with urlopen(url, timeout=30) as response:
            data = response.read()
    else:
        raise ValueError("Expected an image data URL or HTTP(S) URL")
    with Image.open(io.BytesIO(data)) as im:
        if im.has_transparency_data:
            rgba = np.asarray(im.convert("RGBA")).astype(np.int32)
            row, col = np.ogrid[:im.height, :im.width]
            background = np.where((row // 8 + col // 8) % 2 == 0, 204, 153)[:, :, None]
            alpha = rgba[:, :, 3:]
            rgb = (rgba[:, :, :3] * alpha + background * (255-alpha) + 127) // 255
        else:
            rgb = np.asarray(im.convert("RGB"))
        height, width = rgb.shape[:2]
        rows, cols = (height + 39) // 40, (width + 39) // 40
        pixels = np.full((rows*40, cols*40, 3), -1, np.float32)
        pixels[:height, :width] = rgb
        pixels = (pixels * np.float32(1 / 255) - .5) * 2
    return hashlib.sha256(data).digest(), pixels, rows * cols


def encode_image(model, pixels):
    prefix = "visual.backbone."
    def weight(key):
        return model.w.array(prefix + key)
    def linear(x, key):
        w = prefix + key
        output = model.linear(x, w + ".weight", False)
        if w + ".bias" in model.w.tensors:
            output += model.w.array(w + ".bias")
        return bf16(output)
    def norm(x, key):
        centered = x - x.mean(axis=-1, keepdims=True)
        standardized = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
        return bf16(bf16(standardized) * weight(key + ".weight") + weight(key + ".bias"))
    def gelu(x):
        # erf approximation (Abramowitz and Stegun 7.1.26), below BF16 precision.
        z = x / np.sqrt(np.float32(2))
        t = 1 / (1 + .3275911 * np.abs(z))
        polynomial = (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - .284496736)*t + .254829592)*t
        return bf16(.5 * x * (1 + np.sign(z) * (1 - polynomial*np.exp(-z*z))))
    rows, cols = pixels.shape[0] // 40, pixels.shape[1] // 40
    patches = pixels.reshape(rows, 2, 20, cols, 2, 20, 3).transpose(0, 3, 1, 4, 2, 5, 6)
    x = linear(bf16(patches.reshape(-1, 1200)), "embeddings.patch_embedding")
    index = np.arange(len(x))
    row = index // (4 * cols) * 2 + index // 2 % 2
    col = index // 4 % cols * 2 + index % 2
    def taps(pos, length):
        src = (pos.astype(np.float32) + .5) * 32 / length - .5
        indices = np.floor(src)[:, None] + np.array([0, 1], np.float32)
        return indices.astype(int).clip(0, 31), 1 - np.abs(src[:, None] - indices)
    rr, rw = taps(row, rows * 2)
    cc, cw = taps(col, cols * 2)
    pos = weight("embeddings.position_embedding")[rr[:, :, None]*32 + cc[:, None, :]]
    x = bf16(x + bf16((pos * (rw[:, :, None] * cw[:, None, :])[..., None]).sum(axis=(1, 2))))
    cls = bf16(weight("embeddings.class_embedding") + weight("embeddings.cls_position_embedding"))
    x = norm(np.vstack((cls, x)), "ln_pre")
    freq = 10000 ** (-np.arange(0, 32, 2, dtype=np.float32) / 32)
    angles = np.column_stack((np.r_[0, col + 1][:, None]*freq, np.r_[0, row + 1][:, None]*freq))
    angles = np.repeat(angles.astype(np.float32), 2, axis=-1)[:, None]
    cos, sin = np.cos(angles), np.sin(angles)
    def rotate(x):
        pairs = x.reshape(len(x), 16, 32, 2)
        rotated = np.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)
        return bf16(x * cos + rotated * sin)
    for i in range(23):
        p = f"layers.{i}."
        h = norm(x, p + "ln_1")
        q, k, v = [linear(h, p+"self_attn."+s+"_proj").reshape(-1, 16, 64) for s in "qkv"]
        q, k, v = rotate(q).transpose(1, 0, 2), rotate(k).transpose(1, 2, 0), v.transpose(1, 0, 2)
        attended = np.empty_like(q)
        for start in range(0, q.shape[1], 32):
            sl = slice(start, start+32)
            attended[:, sl] = softmax((q[:, sl] @ k) / 8) @ v
        h = linear(bf16(attended.transpose(1, 0, 2).reshape(len(x), -1)), p+"self_attn.out_proj")
        x = bf16(x + bf16(h * weight(p+"ls_1.gamma")))
        h = linear(gelu(linear(norm(x, p+"ln_2"), p+"mlp.c_fc")), p+"mlp.c_proj")
        x = bf16(x + bf16(h * weight(p+"ls_2.gamma")))
    h = gelu(model.linear(x[1:].reshape(rows * cols, 4096), "visual.proj_in.weight"))
    return model.linear(h, "visual.proj_out.weight")
