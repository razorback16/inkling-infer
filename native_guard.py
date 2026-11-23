"""Check native prefill/decode attention, packed KV storage, and prefix branching."""
import numpy as np

from inkling_native.metal import Metal, bf16, from_bits
from inkling_native.model import VisitCache, softmax


def unpack(array):
    if array.dtype == np.uint16:
        return from_bits(array)
    magnitude = (array & 127).astype(np.int32)
    value = np.where(magnitude < 8, magnitude / 512,
                     (1 + (magnitude % 8) / 8) * 2.0 ** (magnitude // 8 - 7))
    return np.where(array & 128, -value, value).astype(np.float32)


def check(gpu, dtype):
    rng = np.random.default_rng(45)
    cache = VisitCache(gpu, 1024, dtype)
    # Exercise a partial query tile, a one-token decode, and compaction.
    for count in (33, 1, 1024, 32):
        q = bf16(rng.normal(size=(count, 24, 128)))
        k = bf16(rng.normal(size=(count, 8, 128)))
        v = bf16(rng.normal(size=k.shape))
        bias = bf16(rng.normal(size=(count, 24, 1024)))
        sinks = bf16(rng.normal(size=(count, 24)))
        past = cache.append(k, v)
        got = gpu.attend(q, cache.keys, cache.values, bias, sinks, past, cache.window)
        for token in {0, count - 1}:
            end = past + token + 1
            start = max(end-1024, 0)
            for head in (0, 2, 3, 23):
                keys = unpack(cache.keys.array[start:end, head // 3])
                values = unpack(cache.values.array[start:end, head // 3])
                scores = keys @ q[token, head] / np.sqrt(np.float32(128))
                scores += bias[token, head, :end-start][::-1]
                probabilities = softmax(np.append(scores, sinks[token, head]))
                np.testing.assert_allclose(got[token, head], probabilities[:-1] @ values,
                                           atol=.002, rtol=.003)
    saved = cache.fork()
    original = cache.keys.array[:cache.length].copy()
    keys = bf16(rng.normal(size=(65, 8, 128)))
    cache.append(keys, keys)
    np.testing.assert_array_equal(saved.keys.array[:saved.length], original)

    # Two continuations that fit in the same allocation must retain their own
    # suffixes, including when the saved prefix is extended again.
    base = VisitCache(gpu, 0, dtype)
    base.append(keys[:1], keys[:1])
    left, right = base.fork(), base.fork()
    left.append(keys[1:2], keys[1:2])
    expected = left.values.array[:2].copy()
    right.append(keys[2:3], keys[2:3])
    base.append(keys[3:4], keys[3:4])
    np.testing.assert_array_equal(left.values.array[:2], expected)
    assert not np.array_equal(left.values.array[1], right.values.array[1])
    assert not np.array_equal(right.values.array[1], base.values.array[1])

    # The graph batches GPU copies across visits. Include both a nonzero
    # sliding-window offset and a shared prefix that detaches within capacity.
    rolling = VisitCache(gpu, 32, dtype)
    rolling.append(keys, -keys)
    branch = left.fork()
    left.append(keys[4:5], -keys[4:5])
    copies = []
    checks = []
    for saved, count, keep in ((rolling, rolling.keys.shape[0], 31), (branch, 1, 2)):
        extent = slice(saved.length - keep, saved.length)
        old_k = saved.keys.array[extent].copy()
        old_v = saved.values.array[extent].copy()
        assert saved.reserve(count, copies) == keep
        checks.append((saved, old_k, old_v))
    assert len(copies) == 4
    gpu.copy_cache(copies)
    for saved, old_k, old_v in checks:
        np.testing.assert_array_equal(saved.keys.array[:len(old_k)], old_k)
        np.testing.assert_array_equal(saved.values.array[:len(old_v)], old_v)


if __name__ == "__main__":
    device = Metal()
    for dtype in (np.uint8, np.uint16):
        check(device, dtype)
    print("Native FP8 and BF16 attention, rollover and prefix checks passed")
