"""Distribute single-token depth attention across queries and residual sources."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers._depth_attention
_scores = mx.fast.metal_kernel(
    name="inkling_depth_source_score", input_names=["blocks", "partial", "queries"],
    output_names=["scores"],
    source="""
        uint source = threadgroup_position_in_grid.x;
        uint query = threadgroup_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        const device float *x = source == K ? partial : blocks + source * 2304;
        float dot = 0;
        for (uint d = tid; d < 2304; d += 288)
            dot += x[d] * queries[query * 2304 + d];
        threadgroup float sums[9];
        dot = simd_sum(dot);
        if (lane == 0) sums[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0;
            for (uint i = 0; i < 9; ++i) total += sums[i];
            scores[query * (K + 1) + source] = total * (1.f / 48.f);
        }
    """,
)
_mix = mx.fast.metal_kernel(
    name="inkling_depth_source_mix", input_names=["blocks", "partial", "scores"],
    output_names=["out"],
    source="""
        uint d = thread_position_in_grid.x;
        uint q = thread_position_in_grid.y;
        auto s = scores + q * (K + 1);
        float peak = s[0];
        for (uint i = 1; i <= K; ++i) peak = max(peak, s[i]);
        float weights[K + 1], z = 0;
        for (uint i = 0; i <= K; ++i) {
            weights[i] = metal::exp(s[i] - peak);
            z += weights[i];
        }
        float value = 0;
        for (uint i = 0; i < K; ++i) value += (weights[i] / z) * blocks[i * 2304 + d];
        out[q * 2304 + d] = value + (weights[K] / z) * partial[d];
    """,
)


def depth(blocks, partial, queries):
    if (partial.shape != (1, 1, 2304) or partial.dtype != mx.float32
            or blocks.dtype != mx.float32 or queries.dtype != mx.float32):
        return _reference(blocks, partial, queries)
    k, q = blocks.shape[0], queries.shape[0]
    template = [("K", k)]
    scores = _scores(inputs=[blocks, partial, queries], template=template,
                     grid=(288 * (k + 1), q, 1), threadgroup=(288, 1, 1),
                     output_shapes=[(q, k + 1)], output_dtypes=[mx.float32])[0]
    return _mix(inputs=[blocks, partial, scores], template=template,
                grid=(2304, q, 1), threadgroup=(128, 1, 1),
                output_shapes=[(1, 1, q, 2304)], output_dtypes=[mx.float32])[0]


def install():
    layers._depth_attention = depth
