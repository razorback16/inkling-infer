"""Trace a complete sliding decode block once the KV ring has reached its capacity."""
import weakref

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, CacheList, RotatingKVCache
from mlx_lm.models.inkling2 import layers

_reference = layers.DecoderLayer.__call__


class _DeviceRing(RotatingKVCache):
    def __init__(self, keys, values, offset, slot):
        self.keys = keys
        self.values = values
        self.offset = offset
        self._idx = slot

    def update_and_fetch(self, keys, values):
        self.keys = mx.slice_update(self.keys, keys, self._idx, axes=(2,))
        self.values = mx.slice_update(self.values, values, self._idx, axes=(2,))
        self._idx = self._idx + 1
        self.offset = self.offset + 1
        return self.keys, self.values


def _build(layer):
    owner = weakref.ref(layer)
    @mx.compile
    def run(x, blocks, messages, keys, values, offset, slot, history):
        ring = _DeviceRing(keys, values, offset, slot)
        conv = ArraysCache(4)
        conv.cache = list(history)
        y, blocks = _reference(owner(), x, blocks, cache=CacheList(ring, conv), conv_mask=None, messages=messages)
        return y, blocks, ring.keys, ring.values, tuple(conv.cache)
    return run


def decode(layer, hidden_states, block_states, cache=None, conv_mask=None, messages=None):
    kwargs = dict(cache=cache, conv_mask=conv_mask, messages=messages)
    if (hidden_states.shape[:2] != (1, 1) or conv_mask is not None or cache is None
            or type(cache[0]) is not RotatingKVCache or layer.mlp.sharding_group is not None):
        return _reference(layer, hidden_states, block_states, **kwargs)
    kv, conv = cache[0], cache[1]
    if (kv.keys is None or kv.keys.shape[2] != kv.max_size or kv.offset < kv.max_size
            or kv.keep or conv.left_padding is not None or conv.lengths is not None):
        return _reference(layer, hidden_states, block_states, **kwargs)
    # Re-prime packed projections and the compiled graph if weights were replaced.
    weights = tuple(v for _, v in tree_flatten(layer) if isinstance(v, mx.array))
    saved = getattr(layer, "_compiled_ring", None)
    if saved is None or len(weights) != len(saved[0]) or not all(a is b for a,b in zip(weights, saved[0])):
        saved = (weights, set(), _build(layer))
        object.__setattr__(layer, "_compiled_ring", saved)
    has_messages = messages is not None
    if has_messages not in saved[1]:
        saved[1].add(has_messages)
        return _reference(layer, hidden_states, block_states, **kwargs)
    slot = kv._idx % kv.max_size
    y, blocks, keys, values, history = saved[2](
        hidden_states, block_states, messages, kv.keys, kv.values,
        mx.array([kv.offset], mx.int32), mx.array([slot], mx.int32), tuple(conv.cache),
    )
    kv.keys, kv.values = keys, values
    kv.offset += 1
    kv._idx = slot + 1
    conv.cache = list(history)
    return y, blocks


def install():
    layers.DecoderLayer.__call__ = decode
