"""Trace the recurrent layer visits together for one-token fixed-capacity decode."""
import weakref

import mlx.core as mx
from .parameter_guard import CapturedParameters
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.model import TextModel

from .compiled_ring import _DeviceRing
from . import prepared_bank, block_lookahead

_reference = TextModel._run_loop


def _compile(model):
    from . import filtered_output, filtered_kv
    filtered_output.prepare(model)
    filtered_kv.prepare(model)
    from .projections import prepare_query_weights
    prepare_query_weights(model)
    groups=block_lookahead.prepare(model)
    owner = weakref.ref(model)

    @mx.compile
    def run(x, states, offset, slots, slab, ring_slot):
        caches = []
        grouped = []
        for (packed, history), slot in zip(states, slots):
            shared = packed is None
            if slot is None:
                slot = ring_slot if shared else offset
            if shared:
                row = len(grouped)
                packed = slab[row:row+1]
            pages = packed if isinstance(packed, tuple) else None
            storage = mx.concatenate(pages[:2], axis=2) if pages is not None else packed
            keys, values = mx.split(storage, 2, axis=1)
            ring = _DeviceRing(keys, values, offset, slot, packed=storage)
            if shared:
                grouped.append(ring)
                ring.grouped = True
            if pages is not None:
                ring.page_prefix, ring.page_tail = pages[:2]
                if len(pages)==3:
                    ring.interleaved_prefix=pages[2]
            conv = ArraysCache(4)
            conv.cache = list(history)
            caches.append(CacheList(ring, conv))
        result = block_lookahead.run(groups,prepared_bank.run,_reference,owner(),x,caches)
        if grouped:
            updates = []
            for ring in grouped:
                value = getattr(ring, "fresh", None)
                if value is None:
                    value = mx.take(ring.packed,ring.write_coordinates[0],axis=2)
                updates.append(value)
            slab = mx.slice_update(slab, mx.concatenate(updates,axis=0),
                                   grouped[0].write_coordinates[0], axes=(2,))
        return result, slab, [
            (tuple(c[1].cache),) if getattr(c[0],"grouped",False) else
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
            if offset > capacity or (offset == capacity and not
                    (k.step == 256 and 1024 <= capacity <= 7936 and capacity % 256 == 0)):
                return _reference(model, x, cache, conv_mask)
            slot = offset
        slots.append(slot)
        joined = getattr(k, "_joint_decode_state", None)
        packed = (joined[0] if joined is not None and joined[1] is k.keys and joined[2] is k.values
                  else mx.concatenate((k.keys, k.values), axis=1))
        if type(k) is KVCache and offset == capacity:
            shape = (*packed.shape[:2], k.step, packed.shape[3])
            packed = (packed, mx.zeros(shape, dtype=packed.dtype))
        elif (type(k) is KVCache and 1024 <= capacity <= 8192 and capacity % 256 == 0
                and offset >= capacity - 256):
            saved_pages = getattr(k, "_decode_pages", None)
            if saved_pages is not None and saved_pages[0] is k.keys and saved_pages[1] is k.values:
                packed = saved_pages[2]
            else:
                packed = tuple(mx.contiguous(t) for t in mx.split(packed, [capacity-256], axis=2))
        if isinstance(packed,tuple) and packed[0].shape[0]==1 and packed[0].shape[-1]==128:
            prefix=packed[0]
            saved_prefix=getattr(k,"_interleaved_global_prefix",None)
            if saved_prefix is None or saved_prefix[0] is not prefix:
                _,banks,length,channels=prefix.shape
                if length % 128 == 0:
                    adjacent = mx.contiguous(prefix.reshape(1,2,banks//2,length//128,8,16,channels)
                                             .transpose(0,2,4,3,5,1,6))
                else:
                    adjacent = mx.contiguous(prefix.reshape(1,2,banks//2,length,channels).transpose(0,2,3,1,4))
                saved_prefix=(prefix,adjacent)
                k._interleaved_global_prefix=saved_prefix
            packed=(*packed,saved_prefix[1])
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
    rows = [i for i,c in enumerate(cache) if type(c[0]) is RotatingKVCache
            and states[i][0].shape == (1,16,1024,128) and states[i][0].dtype == mx.bfloat16]
    slab = None
    incoming = list(states)
    if rows and all(slots[i] == slots[rows[0]] for i in rows):
        tensors = tuple(states[i][0] for i in rows)
        old = getattr(cache[rows[0]][0], "_shared_rings", None)
        reusable = old is not None and len(tensors)==len(old[1]) and all(
            a is b for a,b in zip(tensors,old[1]))
        slab = old[0] if reusable else mx.concatenate(tensors,axis=0)
        for i in rows:
            incoming[i] = (None,states[i][1])
    ring_slot = None if slab is None else params[slots[rows[0]]]
    positions = [None if state[0] is None or type(c[0]) is KVCache else params[slot]
                 for state,c,slot in zip(incoming,cache,slots)]
    result, slab, updated = saved[1](x, incoming, params[offset], positions, slab, ring_slot)
    views = []
    for c, slot, previous, update in zip(cache, slots, states, updated):
        k, conv = c[0], c[1]
        if len(update) == 1:
            history = update[0]
            row = len(views)
            packed = slab[row:row+1]
            views.append(packed)
            keys, values = mx.split(packed,2,axis=1)
        elif len(update) == 2:
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
    if views:
        cache[rows[0]][0]._shared_rings = (slab, tuple(views))
    return result


def install():
    prepared_bank.install()
    TextModel._run_loop = loop
