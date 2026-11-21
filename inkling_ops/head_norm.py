"""Normalize four sets of attention heads in a single decode launch."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar

_kernel = mx.fast.metal_kernel(
    name="inkling_head_norm",
    input_names=["q", "k", "v", "r", "qs", "rs", "threshold", "rel_threshold", "eps"],
    output_names=["qo", "kvo", "ro"],
    source="""
        const uint head = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const bool query = head < H;
        const bool key = head >= H && head < H + KV;
        const bool val = head >= H + KV && head < H + 2 * KV;
        const bool rel = head >= H + 2 * KV;
        const uint width = rel ? R : D;
        const uint local_head = query ? head : key ? head-H : val ? head-H-KV : head-H-2*KV;
        const uint base = local_head * width;
        float states[D / 32];
        float sum = 0;
        for (uint d = lane; d < width; d += 32) {
            uint i = base + d;
            float x = query ? float(q[i]) * float(qs[i]) :
                      key ? float(k[i]) : val ? float(v[i]) : float(r[i]) * float(rs[i]);
            states[d/32] = x;
            sum += x*x;
        }
        sum = simd_sum(sum);
        float scale = val ? metal::precise::rsqrt(sum / float(D) + eps) :
            min(1.0f, (rel ? rel_threshold : threshold) / (metal::precise::sqrt(sum) + 1e-6f));
        for (uint d = lane; d < width; d += 32) {
            T x = T(states[d/32] * scale);
            if (query) qo[base+d] = x;
            else if (key) kvo[base+d] = x;
            else if (val) kvo[KV*D+base+d] = x;
            else ro[base+d] = x;
        }
    """,
)


def normalize(layer, query, key, value, relative, return_pair=False):
    heads, kv, dim = layer.num_heads, layer.num_key_value_heads, layer.head_dim
    rel = relative.shape[-1] // heads
    q, pair, r = _kernel(
        inputs=[query, key, value, relative, layer.q_l2_clip.scale, layer.rel_l2_clip.scale,
                scalar(layer.q_l2_clip.threshold), scalar(layer.rel_l2_clip.threshold), scalar(layer.eps)],
        template=[("T", query.dtype), ("H", heads), ("KV", kv), ("D", dim), ("R", rel)],
        grid=(32 * (2 * heads + 2 * kv), 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, heads, dim), (1, 1, 2*kv, dim), (1, 1, heads, rel)],
        output_dtypes=[query.dtype] * 3,
    )
    k, v = mx.split(pair,2,axis=2)
    return (q, k, v, r, pair) if return_pair else (q, k, v, r)
