"""Normalize the residual and evaluate the learned depth queries in one launch."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_residual_depth", input_names=["x", "blocks", "queries"], output_names=["mixed", "normalized"],
    source="""
        uint tid = thread_position_in_threadgroup.x, sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup, query = threadgroup_position_in_grid.x;
        threadgroup float norm[2304], sums[18], scores[K+1][9], coeff[K+1];
        float sum = 0;
        for (uint j = 0; j < 4; ++j) {
            float v = float(x[tid*4+j]);
            sum += v*v;
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = precise::rsqrt(simd_sum(lane < 18 ? sums[lane] : 0.f)/2304.f + 1.e-6f);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            norm[d] = float(x[d])*scale;
            if (!query) normalized[d] = norm[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 288) {
            for (uint i = 0; i <= K; ++i) {
                float dot = 0;
                for (uint d = tid; d < 2304; d += 288)
                    dot += (i == K ? norm[d] : blocks[i*2304+d])*queries[query*2304+d];
                dot = simd_sum(dot);
                if (lane == 0) scores[i][sg] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float peak = -INFINITY, z = 0;
            for (uint i = 0; i <= K; ++i) {
                float dot = 0;
                for (uint j = 0; j < 9; ++j) dot += scores[i][j];
                coeff[i] = dot * (1.f/48.f);
                peak = max(peak, coeff[i]);
            }
            for (uint i = 0; i <= K; ++i) {
                coeff[i] = metal::exp(coeff[i]-peak);
                z += coeff[i];
            }
            for (uint i = 0; i <= K; ++i) coeff[i] /= z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            float v = 0;
            for (uint i = 0; i < K; ++i) v += blocks[i*2304+d]*coeff[i];
            mixed[query*2304+d] = v + norm[d]*coeff[K];
        }
    """,
)


def evaluate(module, blocks, x):
    q = module.query.size // 2304
    return _kernel(inputs=[x,blocks,module.query], template=[("K",blocks.shape[0])],
                   grid=(q*576,1,1),threadgroup=(576,1,1),
                   output_shapes=[(1,1,q,2304),x.shape],output_dtypes=[mx.float32,mx.float32])
