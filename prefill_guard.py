"""Check cache state and continuation logits against stock after real prefill."""
import gc
import hashlib

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

from prefill_benchmark import CHUNK, MODEL, PREFIX, fill


def digest(arr):
    return hashlib.sha256(np.asarray(arr.view(mx.uint8)).tobytes()).digest()


def fingerprint(value):
    if isinstance(value, mx.array):
        return value.shape, str(value.dtype), digest(value)
    if isinstance(value, (tuple, list)):
        return tuple(fingerprint(child) for child in value)
    return value


def snapshots():
    model, tok = load(MODEL)
    text = tok.encode(PREFIX.read_text())
    alternate = tok.encode("When the counter reaches zero, record the time. " * 150)
    captured = {}
    for ids, chunk in ((text, CHUNK), (alternate[:1031], 257)):
        cache = make_prompt_cache(model)
        fill(model, mx.array(ids), cache, chunk)
        for i, (kv, conv) in enumerate(cache):
            assert kv.offset == len(ids)
            captured[len(ids), i] = fingerprint((kv.state, conv.state))
        for j, token in enumerate(text[100:104]):
            scores = model(mx.array([[token]]), cache=cache)[0, -1]
            captured[len(ids), "continuation", j] = digest(scores)
    return captured


if __name__ == "__main__":
    stock = snapshots()
    gc.collect()
    mx.clear_cache()
    from inkling_kernel import install
    install()
    actual = snapshots()
    mismatches = [k for k in stock if stock[k] != actual[k]]
    if mismatches:
        raise ValueError(f"{len(mismatches)} mismatches starting at {mismatches[:8]}")
    print("Exact stock cache and continuation matches for both prompts and all 48 visits.")
