"""Vectorized one-token dense projections."""
import mlx.core as mx

_matvec = mx.fast.metal_kernel(
    name="inkling_multi_input_matvec", input_names=["query", "key", "value", "w"], output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint row = (threadgroup_position_in_grid.x * GROUPS + simdgroup_index_in_threadgroup) * ROWS;
        if (row >= N) return;
        auto x = row < KSTART ? query : (row < VSTART ? key : value);
        float sums[ROWS] = {};
        const device vec<T,4> *wv = (const device vec<T,4>*)w;
        for (uint i = lane; i < K / 4; i += 32) {
            float4 v = float4(float(x[i*4]), float(x[i*4+1]), float(x[i*4+2]), float(x[i*4+3]));
            for (uint r = 0; r < ROWS; ++r) {
                float4 wr = float4(wv[(row+r)*(K/4)+i]);
                sums[r] += dot(v, wr);
            }
        }
        for (uint r = 0; r < ROWS; ++r) {
            float total = simd_sum(sums[r]);
            if (lane == 0) out[row + r] = T(total);
        }
    """,
)


def matvec(query, key, value, w, key_start, value_start):
    n, k = w.shape
    rows, groups = 2, 8
    tiles = (n + rows * groups - 1) // (rows * groups)
    return _matvec(
        inputs=[query, key, value, w],
        template=[("K", k), ("N", n), ("ROWS", rows), ("GROUPS", groups), ("T", w.dtype),
                  ("KSTART", key_start), ("VSTART", value_start)],
        grid=(tiles * groups * 32, 1, 1), threadgroup=(groups * 32, 1, 1),
        output_shapes=[query.shape[:-1] + (n,)], output_dtypes=[w.dtype])[0]
