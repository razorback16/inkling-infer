"""Normalize the residual and evaluate the learned depth queries in one launch."""
import mlx.core as mx

_filter = """
            uint d = tid*4+j;
            float conv = 0;
            for (uint t = 0; t < 3; ++t) {
                float old = history[t*2304+d];
                conv = fma(float(W(old)), float(w[d*4+t]), conv);
                window[t*2304+d] = old;
            }
            window[3*2304+d] = v;
            conv = fma(float(W(v)), float(w[d*4+3]), conv);
            float filtered = float(T((float(W(conv))+v) * 0.4472135954999579f));
            v = BOUNDARY ? filtered : float(hidden[d]) + filtered;
            residual[d] = v;
"""

_source = """
        uint tid = thread_position_in_threadgroup.x, sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup, query = threadgroup_position_in_grid.x;
        threadgroup float norm[2304], sums[18], scores[K+1][9], coeff[K+1];
        float4 values;
        float sum = 0;
        for (uint j = 0; j < 4; ++j) {
            float v = float(x[tid*4+j]);
            FILTER
            values[j] = v;
            sum += v*v;
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = precise::rsqrt(simd_sum(lane < 18 ? sums[lane] : 0.f)/2304.f + 1.e-6f);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            norm[d] = values[j]*scale;
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
    """


def make_kernel(filtered):
    inputs = ["x", "blocks", "queries"]
    outputs = ["mixed", "normalized"]
    if filtered:
        inputs.extend(["history", "w", "hidden"])
        outputs.extend(["residual", "window"])
    return mx.fast.metal_kernel(
        name="inkling_filtered_depth" if filtered else "inkling_residual_depth",
        input_names=inputs, output_names=outputs,
        source=_source.replace("FILTER", _filter if filtered else ""),
    )


_kernel = make_kernel(False)
_filtered = make_kernel(True)


def evaluate(module, blocks, x):
    q = module.query.size // 2304
    return _kernel(inputs=[x,blocks,module.query], template=[("K",blocks.shape[0])],
                   grid=(q*576,1,1),threadgroup=(576,1,1),
                   output_shapes=[(1,1,q,2304),x.shape],output_dtypes=[mx.float32,mx.float32])


def post_attention(layer, blocks, attention, history, hidden):
    weights = layer.post_attention_convolution.conv1d.weight
    mixed, _, residual, window = _filtered(
        inputs=[attention, blocks, layer.mlp_input.query, history, weights, hidden],
        template=[("K", blocks.shape[0]), ("T", attention.dtype), ("W", weights.dtype),
                  ("BOUNDARY", layer.is_block_boundary)],
        grid=(576,1,1), threadgroup=(576,1,1),
        output_shapes=[(1,1,1,2304), hidden.shape, hidden.shape, (1,4,2304)],
        output_dtypes=[mx.float32]*4)
    return mixed, residual, window
