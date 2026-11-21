"""Trace the recurrent layer visits together for one-token fixed-capacity decode."""
import weakref

import mlx.core as mx
from .parameter_guard import CapturedParameters
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.model import TextModel

from .compiled_ring import _DeviceRing
from . import prepared_bank, block_lookahead

_reference = TextModel._run_loop


def _commit_rows(storage, rings, start, source):
    if not rings:
        return storage
    updates = []
    for ring in rings:
        value = getattr(ring, "fresh", None)
        if value is None:
            value = mx.take(getattr(ring,source),start,axis=2)
        updates.append(value)
    return mx.slice_update(storage,mx.concatenate(updates,axis=0),start,axes=(2,))


def _reuse_rows(saved, tensors):
    if saved is not None and len(saved[1])==len(tensors) and all(
            a is b for a,b in zip(saved[1],tensors)):
        return saved[0]
    return mx.concatenate(tensors,axis=0)


def _compile(model):
    groups=block_lookahead.prepare(model)
    owner = weakref.ref(model)

    @mx.compile
    def run(x, states, offset, slots, slab, ring_slot, tails):
        caches = []
        grouped, global_rings = [], []
        for (packed, history), slot in zip(states, slots):
            shared = packed is None
            if slot is None:
                slot = ring_slot if shared else offset
            if shared:
                row = len(grouped)
                packed = slab[row:row+1]
            pages = packed if isinstance(packed, tuple) else None
            shared_tail = pages is not None and len(pages)==1
            if shared_tail:
                row = len(global_rings)
                pages = (pages[0],tails[row:row+1])
            storage = mx.concatenate(pages, axis=2) if pages is not None else packed
            keys, values = mx.split(storage, 2, axis=1)
            ring = _DeviceRing(keys, values, offset, slot, packed=storage)
            if shared:
                grouped.append(ring)
                ring.grouped = True
            if shared_tail:
                global_rings.append(ring)
                ring.grouped_tail = True
            if pages is not None:
                ring.page_prefix, ring.page_tail = pages
            conv = ArraysCache(4)
            conv.cache = list(history)
            caches.append(CacheList(ring, conv))
        result = block_lookahead.run(groups,prepared_bank.run,_reference,owner(),x,caches)
        if grouped:
            slab = _commit_rows(slab,grouped,grouped[0].write_coordinates[0],"packed")
        if global_rings:
            start = offset - global_rings[0].page_prefix.shape[2]
            tails = _commit_rows(tails,global_rings,start,"page_tail")
        return result, slab, tails, [
            (None,tuple(c[1].cache)) if getattr(c[0],"grouped_tail",False) else
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
    rows = [i for i,c in enumerate(cache) if type(c[0]) is RotatingKVCache
            and states[i][0].shape == (1,16,1024,128) and states[i][0].dtype == mx.bfloat16]
    slab = None
    incoming = list(states)
    if rows and all(slots[i] == slots[rows[0]] for i in rows):
        tensors = tuple(states[i][0] for i in rows)
        old = getattr(cache[rows[0]][0], "_shared_rings", None)
        slab = _reuse_rows(old,tensors)
        for i in rows:
            incoming[i] = (None,states[i][1])
    global_rows = [i for i,(packed,_) in enumerate(states) if isinstance(packed,tuple)
                   and packed[1].shape==(1,16,256,128) and packed[1].dtype==mx.bfloat16]
    tails = None
    if global_rows and all(states[i][0][0].shape==states[global_rows[0]][0][0].shape for i in global_rows):
        old = getattr(cache[global_rows[0]][0],"_shared_tails",None)
        tails = _reuse_rows(old,tuple(states[i][0][1] for i in global_rows))
        for i in global_rows:
            incoming[i] = ((states[i][0][0],),states[i][1])
    ring_slot = None if slab is None else params[slots[rows[0]]]
    positions = [None if state[0] is None or type(c[0]) is KVCache else params[slot]
                 for state,c,slot in zip(incoming,cache,slots)]
    result, slab, tails, updated = saved[1](x,incoming,params[offset],positions,slab,ring_slot,tails)
    views, tail_views = [], []
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
            if tail is None:
                row = len(tail_views)
                tail = tails[row:row+1]
                tail_views.append(tail)
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
    if tail_views:
        cache[global_rows[0]][0]._shared_tails = (tails,tuple(tail_views))
    if views:
        cache[rows[0]][0]._shared_rings = (slab, tuple(views))
    return result


def install():
    prepared_bank.install()
    TextModel._run_loop = loop
