"""Construct a decode attention mask directly in physical KV cache order."""
import mlx.core as mx
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.inkling2 import layers

_reference = layers.Attention._attention
_mask = mx.fast.metal_kernel(
    name="inkling_relative_decode_mask",
    input_names=["bias", "end", "next_slot"],
    output_names=["out"],
    source="""
        int k = thread_position_in_grid.x;
        int head = thread_position_in_grid.y;
        int count = threads_per_grid.x;
        int distance = next_slot[0] - 1 - k;
        if (distance < 0) distance += count;
        bool valid = distance < end[0] && (WINDOW == 0 || distance < WINDOW);
        out[head * count + k] = valid
            ? (distance < R ? T(bias[head * R + distance]) : T(0))
            : T(-INFINITY);
    """,
)


def attention(layer, query, key, value, position_bias, sinks, cache):
    batch, heads, length, _ = query.shape
    if batch != 1 or length != 1 or cache is None:
        return _reference(layer, query, key, value, position_bias, sinks, cache)
    count = key.shape[2]
    rotated = isinstance(cache, (RotatingKVCache, BatchRotatingKVCache))
    next_slot = cache._idx if rotated else count
    end = mx.array(cache.offset, dtype=mx.int32)
    mask = _mask(
        inputs=[position_bias, end, mx.array(next_slot, dtype=mx.int32)],
        template=[("T", query.dtype), ("R", position_bias.shape[-1]), ("WINDOW", layer.sliding_window)],
        grid=(count, heads, 1), threadgroup=(min(256, count), 1, 1),
        output_shapes=[(1, heads, 1, count)], output_dtypes=[query.dtype],
    )[0]
    out = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=layer.scaling, mask=mask, sinks=sinks[0, 0].astype(query.dtype)
    )
    return out.transpose(0, 2, 1, 3)


def install():
    layers.Attention._attention = attention
