"""Store Inkling attention keys and values as MLX E4M3 bytes."""

import mlx.core as mx
from mlx_lm.models import cache
from mlx_lm.models.inkling2.model import Model


class _FP8Storage:
    def update_and_fetch(self, keys, values):
        stored_keys, stored_values = super().update_and_fetch(
            mx.to_fp8(keys), mx.to_fp8(values)
        )
        return mx.from_fp8(stored_keys, keys.dtype), mx.from_fp8(stored_values, values.dtype)


class FP8KVCache(_FP8Storage, cache.KVCache):
    @classmethod
    def merge(cls, caches):
        # MLX-LM constructs a plain BatchKVCache for an empty batch.
        if all(c.size() == 0 for c in caches):
            return FP8BatchKVCache([0] * len(caches))
        return FP8BatchKVCache.merge(caches)


class FP8RotatingKVCache(_FP8Storage, cache.RotatingKVCache):
    @classmethod
    def merge(cls, caches):
        return FP8BatchRotatingKVCache.merge(caches)


class FP8BatchKVCache(_FP8Storage, cache.BatchKVCache):
    def extract(self, idx):
        return FP8KVCache.from_state(super().extract(idx).state)


class FP8BatchRotatingKVCache(_FP8Storage, cache.BatchRotatingKVCache):
    def extract(self, idx):
        return FP8RotatingKVCache.from_state(super().extract(idx).state)


_default_make_cache = Model.make_cache


def _make_cache(model):
    caches = _default_make_cache(model)
    for layer in caches:
        attention, conv = layer.caches
        if isinstance(attention, cache.RotatingKVCache):
            attention = FP8RotatingKVCache(attention.max_size, attention.keep)
        elif type(attention) is cache.KVCache:
            attention = FP8KVCache()
        else:
            raise TypeError(f"Unsupported Inkling attention cache: {type(attention).__name__}")
        layer.caches = attention, conv
    return caches


def install():
    # MLX-LM restores nested CacheList members by looking up their class names.
    for cls in (FP8KVCache, FP8RotatingKVCache, FP8BatchKVCache, FP8BatchRotatingKVCache):
        setattr(cache, cls.__name__, cls)
    Model.make_cache = _make_cache
