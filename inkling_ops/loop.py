"""Trace the recurrent layer visits together for one-token fixed-capacity decode."""
import weakref

import mlx.core as mx
from .parameter_guard import CapturedParameters
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.model import TextModel

from .compiled_ring import _DeviceRing

_reference = TextModel._run_loop


def _compile(model, kinds):
    owner = weakref.ref(model)

    @mx.compile
    def run(x, states, offset, after_offset, slots, next_slots):
        caches = []
        for (packed, history), position in zip(states, kinds):
            slot = slots[position]
            keys, values = mx.split(packed, 2, axis=1)
            ring = _DeviceRing(keys, values, offset, slot, packed=packed, after=(after_offset,next_slots[position]))
            conv = ArraysCache(4)
            conv.cache = list(history)
            caches.append(CacheList(ring, conv))
        result = _reference(owner(), x, caches, None)
        return result, [(c[0].packed, c[0].keys, c[0].values, tuple(c[1].cache)) for c in caches]

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
        states.append((packed, tuple(conv.cache)))
    # Caches with matching write positions share one compiled graph parameter.
    # Record the grouping rather than the positions themselves in the trace key.
    unique = list(dict.fromkeys(slots))
    indices = {slot:i for i,slot in enumerate(unique)}
    kinds = tuple(indices[slot] for slot in slots)
    saved = getattr(model, "_decode_loop", None)
    if saved is None or kinds != saved[2] or not saved[0].matches(model):
        if any(layer.mlp.sharding_group is not None for layer in model.layers):
            return _reference(model, x, cache, conv_mask)
        guard = CapturedParameters()
        object.__setattr__(model, "_decode_loop", (guard, _compile(model,kinds),kinds))
        result = _reference(model, x, cache, conv_mask)
        guard.capture(model)
        return result
    numbers = set(unique + [offset])
    params = {n:mx.array([n],mx.int32) for n in numbers | {v+1 for v in numbers}}
    result, updated = saved[1](x, states, params[offset], params[offset+1],
                                [params[n] for n in unique], [params[n+1] for n in unique])
    for c, slot, (packed, keys, values, history) in zip(cache, slots, updated):
        k, conv = c[0], c[1]
        k.keys, k.values = keys, values
        k._joint_decode_state = (packed, keys, values)
        k.offset += 1
        if type(k) is RotatingKVCache:
            k._idx = slot + 1
        conv.cache = list(history)
    return result


def install():
    TextModel._run_loop = loop
