"""Check FP8 storage, sliding-window rollover and reusable batched prefixes."""

import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import cache

from inkling_ops.fp8_cache import FP8KVCache, FP8RotatingKVCache, install


def roundtrip(x):
    return mx.from_fp8(mx.to_fp8(x), x.dtype)


def check(make_fp8, make_reference):
    compressed = make_fp8()
    reference = make_reference()
    for length in (3, 2, 1, 1, 1, 2):
        keys = mx.random.normal((1, 2, length, 16)).astype(mx.bfloat16)
        values = -keys
        actual = compressed.update_and_fetch(keys, values)
        expected = reference.update_and_fetch(roundtrip(keys), roundtrip(values))
        for x, y in zip(actual, expected):
            assert x.dtype == y.dtype and mx.array_equal(x, y).item()
        assert compressed.keys.dtype == compressed.values.dtype == mx.uint8
        assert compressed.nbytes * 2 == reference.nbytes

    # Both a cold batch and a batch assembled from prefixes must preserve FP8.
    for prefix in (make_fp8(), compressed):
        batch = prefix.merge([prefix, make_fp8()])
        x = mx.random.normal((2, 2, 1, 16)).astype(mx.bfloat16)
        batch.update_and_fetch(x, -x)
        assert batch.keys.dtype == batch.values.dtype == mx.uint8
        for idx in (0, 1):
            single = batch.extract(idx)
            assert type(single) is type(prefix)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "prefix.safetensors")
                cache.save_prompt_cache(path, [cache.CacheList(single)])
                restored = cache.load_prompt_cache(path)[0][0]
            assert type(restored) is type(single)
            actual = restored.update_and_fetch(x[:1], -x[:1])
            expected = single.update_and_fetch(x[:1], -x[:1])
            assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))


if __name__ == "__main__":
    install()
    mx.random.seed(123)
    check(FP8KVCache, cache.KVCache)
    check(lambda: FP8RotatingKVCache(4), lambda: cache.RotatingKVCache(4))
    print("FP8 cache checks passed")
