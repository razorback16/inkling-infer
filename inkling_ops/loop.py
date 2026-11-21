"""Trace the recurrent layer visits together for one-token fixed-capacity decode."""
import weakref

import mlx.core as mx
from .parameter_guard import CapturedParameters
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.model import TextModel

from .compiled_ring import _DeviceRing
from . import prepared_bank

_reference = TextModel._run_loop


def _compile(model):
    owner = weakref.ref(model)

    @mx.compile
    def run(x, states, offset, slots):
        caches = []
        for (packed, history), slot in zip(states, slots):
            pages = packed if isinstance(packed, tuple) else None
            storage = mx.concatenate(pages, axis=2) if pages is not None else packed
            keys, values = mx.split(storage, 2, axis=1)
            ring = _DeviceRing(keys, values, offset, slot, packed=storage)
            if pages is not None:
                ring.page_prefix, ring.page_tail = pages
            conv = ArraysCache(4)
            conv.cache = list(history)
            caches.append(CacheList(ring, conv))
        result = prepared_bank.run(_reference, owner(), x, caches)
        return result, [
            (c[0].page_tail, tuple(c[1].cache)) if hasattr(c[0], "page_tail")
            else (c[0].packed, c[0].keys, c[0].values, tuple(c[1].cache)) for c in caches
        ]

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
        joined = getattr(k, "_joint_decode_state", None)
        packed = (joined[0] if joined is not None and joined[1] is k.keys and joined[2] is k.values
                  else mx.concatenate((k.keys, k.values), axis=1))
        if (type(k) is KVCache and 1024 <= capacity <= 8192 and capacity % 256 == 0
                and offset >= capacity - 256):
            saved_pages = getattr(k, "_decode_pages", None)
            if saved_pages is not None and saved_pages[0] is k.keys and saved_pages[1] is k.values:
                packed = saved_pages[2]
            else:
                packed = tuple(mx.contiguous(t) for t in mx.split(packed, [capacity-256], axis=2))
        states.append((packed, tuple(conv.cache)))
    saved = getattr(model, "_decode_loop", None)
    if saved is None or not saved[0].matches(model):
        if any(layer.mlp.sharding_group is not None for layer in model.layers):
            return _reference(model, x, cache, conv_mask)
        guard = CapturedParameters()
        object.__setattr__(model, "_decode_loop", (guard, _compile(model)))
        result = _reference(model, x, cache, conv_mask)
        guard.capture(model)
        return result
    params = {slot: mx.array([slot], mx.int32) for slot in set(slots + [offset])}
    result, updated = saved[1](x, states, params[offset], [params[i] for i in slots])
    for c, slot, previous, update in zip(cache, slots, states, updated):
        k, conv = c[0], c[1]
        if len(update) == 2:
            tail, history = update
            pages = (previous[0][0], tail)
            packed = mx.concatenate(pages, axis=2)
            keys, values = mx.split(packed, 2, axis=1)
            k._decode_pages = (keys, values, pages)
        else:
            packed, keys, values, history = update
        k.keys, k.values = keys, values
        k._joint_decode_state = (packed, keys, values)
        k.offset += 1
        if type(k) is RotatingKVCache:
            k._idx = slot + 1
        conv.cache = list(history)
    return result


def install():
    prepared_bank.install()
    TextModel._run_loop = loop
