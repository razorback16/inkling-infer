"""Regenerate the architecture's BF16-indexed FP32 SwiGLU factors using CPU PyTorch."""
from pathlib import Path
import zlib

import torch


def main():
    x = torch.arange(65536).to(torch.int16).view(torch.bfloat16).float()
    gate = (torch.tanh(x / 4) * 4) * torch.sigmoid(x)
    up = torch.tanh(x / 25) * 25
    factors = torch.stack((gate, up), dim=-1).numpy().astype("<f4")
    path = Path(__file__).resolve().parents[1] / "inkling_native" / "swiglu.f32.zlib"
    path.write_bytes(zlib.compress(factors.tobytes(), 9))


if __name__ == "__main__":
    main()
