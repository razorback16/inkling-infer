"""Trace the recurrent layer visits together for one-token fixed-capacity decode."""
import weakref

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.model import TextModel

from . import bank
from .compiled_ring import _DeviceRing

_reference = TextModel._run_loop


def _compile(model):
    owner = weakref.ref(model)

    @mx.compile
    def run(x, states, offset, slots):
        caches = []
        for (keys, values, history), slot in zip(states, slots):
            ring = _DeviceRing(keys, values, offset, slot)
            conv = ArraysCache(4)
            conv.cache = list(history)
            caches.append(CacheList(ring, conv))
        result = bank.run(owner(), x, caches)
        return result, [(c[0].keys, c[0].values, tuple(c[1].cache)) for c in caches]

    return run


def loop(model, x, cache, conv_mask):
    if x.shape[:2] != (1, 1) or conv_mask is not None or cache[0] is None:
        return _reference(model, x, cache, conv_mask)
    offset = cache[0][0].offset
    if type(offset) is not int or offset < 1024:
        return _reference(model, x, cache, conv_mask)
    states, slots = [], []
    for c in cache:
        k, conv = c[0], c[1]
        if (type(k) not in (RotatingKVCache, KVCache) or k.offset != offset or k.keys is None
                or conv.left_padding is not None or conv.lengths is not None
                or any(h is None for h in conv.cache)):
            return _reference(model, x, cache, conv_mask)
        capacity = k.keys.shape[2]
        if type(k) is RotatingKVCache:
            if k.keep or k.max_size != capacity or offset < capacity:
                return _reference(model, x, cache, conv_mask)
            slot = k._idx % capacity
        else:
            if offset >= capacity:
                return _reference(model, x, cache, conv_mask)
            slot = offset
        slots.append(slot)
        states.append((k.keys, k.values, tuple(conv.cache)))
    weights = tuple(v for _, v in tree_flatten(model) if isinstance(v, mx.array))
    saved = getattr(model, "_decode_loop", None)
    if saved is None or len(weights) != len(saved[0]) or any(a is not b for a,b in zip(weights, saved[0])):
        if any(layer.mlp.sharding_group is not None for layer in model.layers):
            return _reference(model, x, cache, conv_mask)
        object.__setattr__(model, "_decode_loop", (weights, _compile(model)))
        # Populate lazy weight packs before tracing the complete recurrent graph.
        return _reference(model, x, cache, conv_mask)
    params = {slot: mx.array([slot], mx.int32) for slot in set(slots + [offset])}
    result, updated = saved[1](x, states, params[offset], [params[i] for i in slots])
    for c, slot, (keys, values, history) in zip(cache, slots, updated):
        k, conv = c[0], c[1]
        k.keys, k.values = keys, values
        k.offset += 1
        if type(k) is RotatingKVCache:
            k._idx = slot + 1
        conv.cache = list(history)
    return result


def install():
    TextModel._run_loop = loop
