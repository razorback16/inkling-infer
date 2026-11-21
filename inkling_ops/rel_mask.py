"""Project relative logits directly into physical KV mask slots for one-token decode."""
import mlx.core as mx
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

_kernel = mx.fast.metal_kernel(
    name="inkling_projected_mask", input_names=["rel", "w", "end", "slot"], output_names=["out"],
    source="""
        int k = thread_position_in_grid.x;
        int head = thread_position_in_grid.y;
        int n = threads_per_grid.x;
        int d = slot[0] - 1 - k;
        if (d < 0) d += n;
        float score = 0;
        if (d >= end[0] || (WINDOW && d >= WINDOW)) {
            score = -INFINITY;
        } else if (d < R) {
            for (int j = 0; j < WIDTH; ++j)
                score = fma(float(rel[head * WIDTH + j]), float(w[j * R + d]), score);
        }
        out[head * n + k] = T(score);
    """,
)


def attend(layer, q, k, v, rel, sinks, cache):
    w = layer.rel_logits_proj.proj.astype(rel.dtype)
    length = k.shape[2]
    slot = cache._idx if isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)) else length
    mask = _kernel(
        inputs=[rel, w, mx.array(cache.offset, mx.int32).reshape(1), mx.array(slot, mx.int32).reshape(1)],
        template=[("T", q.dtype), ("WIDTH", w.shape[0]), ("R", w.shape[1]), ("WINDOW", layer.sliding_window)],
        grid=(length, q.shape[1], 1), threadgroup=(min(length, 128), 1, 1),
        output_shapes=[(1, q.shape[1], 1, length)], output_dtypes=[q.dtype])[0]
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=layer.scaling, mask=mask, sinks=sinks[0, 0].astype(q.dtype)
    ).transpose(0, 2, 1, 3)
